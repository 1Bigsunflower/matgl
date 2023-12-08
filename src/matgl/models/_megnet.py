"""Implementation of MatErials Graph Network (MEGNet) model.

Graph networks are a new machine learning (ML) paradigm that supports both relational reasoning and combinatorial
generalization. For more details on MEGNet, please refer to::

    Chen, C.; Ye, W.; Zuo, Y.; Zheng, C.; Ong, S. P. _Graph Networks as a Universal Machine Learning Framework for
    Molecules and Crystals._ Chem. Mater. 2019, 31 (9), 3564-3572. DOI: 10.1021/acs.chemmater.9b01294.
"""
from __future__ import annotations
import csv
import logging
from typing import TYPE_CHECKING

import torch
from dgl.nn.pytorch import Set2Set
from torch import nn

from matgl.config import DEFAULT_ELEMENTS
from matgl.graph.compute import compute_pair_vector_and_distance
from matgl.layers import MLP, ActivationFunction, BondExpansion, EdgeSet2Set, EmbeddingBlock, MEGNetBlock
from matgl.utils.io import IOMixIn
from pymatgen.core import Element
if TYPE_CHECKING:
    import dgl

    from matgl.graph.converters import GraphConverter

logger = logging.getLogger(__file__)


class MEGNet(nn.Module, IOMixIn):
    """DGL implementation of MEGNet."""

    __version__ = 1

    def __init__(
        self,
        dim_node_embedding: int = 16,
        dim_edge_embedding: int = 100,
        dim_state_embedding: int = 2,
        ntypes_state: int | None = None,
        nblocks: int = 3,
        hidden_layer_sizes_input: tuple[int, ...] = (64, 32),
        hidden_layer_sizes_conv: tuple[int, ...] = (64, 64, 32),
        hidden_layer_sizes_output: tuple[int, ...] = (32, 16),
        nlayers_set2set: int = 1,
        niters_set2set: int = 2,
        activation_type: str = "softplus2",
        is_classification: bool = False,
        include_state: bool = True,
        dropout: float = 0.0,
        element_types: tuple[str, ...] = DEFAULT_ELEMENTS,
        bond_expansion: BondExpansion | None = None,
        cutoff: float = 4.0,
        gauss_width: float = 0.5,
        **kwargs,
    ):
        """Useful defaults for all arguments have been specified based on MEGNet formation energy model.

        Args:
            dim_node_embedding: Dimension of node embedding.
            dim_edge_embedding: Dimension of edge embedding.
            dim_state_embedding: Dimension of state embedding.
            ntypes_state: Number of state types.
            nblocks: Number of blocks.
            hidden_layer_sizes_input: Architecture of dense layers before the graph convolution
            hidden_layer_sizes_conv: Architecture of dense layers for message and update functions
            nlayers_set2set: Number of layers in Set2Set layer
            niters_set2set: Number of iterations in Set2Set layer
            hidden_layer_sizes_output: Architecture of dense layers for concatenated features after graph convolution
            activation_type: Activation used for non-linearity
            is_classification: Whether this is classification task or not
            layer_node_embedding: Architecture of embedding layer for node attributes
            layer_edge_embedding: Architecture of embedding layer for edge attributes
            layer_state_embedding: Architecture of embedding layer for state attributes
            include_state: Whether the state embedding is included
            dropout: Randomly zeroes some elements in the input tensor with given probability (0 < x < 1) according to
                a Bernoulli distribution. Defaults to 0, i.e., no dropout.
            element_types: Elements included in the training set
            bond_expansion: Gaussian expansion for edge attributes
            cutoff: cutoff for forming bonds
            gauss_width: width of Gaussian function for bond expansion
            **kwargs: For future flexibility. Not used at the moment.
        """
        super().__init__()

        self.save_args(locals(), kwargs)

        self.element_types = element_types or DEFAULT_ELEMENTS
        self.cutoff = cutoff
        self.bond_expansion = bond_expansion or BondExpansion(
            rbf_type="Gaussian", initial=0.0, final=cutoff + 1.0, num_centers=dim_edge_embedding, width=gauss_width
        )

        node_dims = [dim_node_embedding, *hidden_layer_sizes_input]
        edge_dims = [dim_edge_embedding, *hidden_layer_sizes_input]
        state_dims = [dim_state_embedding, *hidden_layer_sizes_input]

        try:
            activation: nn.Module = ActivationFunction[activation_type].value()
        except KeyError:
            raise ValueError(
                f"Invalid activation type, please try using one of {[af.name for af in ActivationFunction]}"
            ) from None

        self.embedding = EmbeddingBlock(
            degree_rbf=dim_edge_embedding,
            dim_node_embedding=dim_node_embedding,
            ntypes_node=len(self.element_types),
            ntypes_state=ntypes_state,
            include_state=include_state,
            dim_state_embedding=dim_state_embedding,
            activation=activation,
        )

        self.edge_encoder = MLP(edge_dims, activation, activate_last=True)
        self.node_encoder = MLP(node_dims, activation, activate_last=True)
        self.state_encoder = MLP(state_dims, activation, activate_last=True)

        dim_blocks_in = hidden_layer_sizes_input[-1]
        dim_blocks_out = hidden_layer_sizes_conv[-1]
        block_args = {
            "conv_hiddens": hidden_layer_sizes_conv,
            "dropout": dropout,
            "act": activation,
            "skip": True,
        }
        # first block
        blocks = [MEGNetBlock(dims=[dim_blocks_in], **block_args)] + [  # type: ignore
            MEGNetBlock(dims=[dim_blocks_out, *hidden_layer_sizes_input], **block_args)  # type: ignore
            for _ in range(nblocks - 1)
        ]

        self.blocks = nn.ModuleList(blocks)

        s2s_kwargs = {"n_iters": niters_set2set, "n_layers": nlayers_set2set}
        self.edge_s2s = EdgeSet2Set(dim_blocks_out, **s2s_kwargs)
        self.node_s2s = Set2Set(dim_blocks_out, **s2s_kwargs)

        self.output_proj = MLP(
            # S2S cats q_star to output producing double the dim
            dims=[2 * 2 * dim_blocks_out + dim_blocks_out, *hidden_layer_sizes_output, 1],
            activation=activation,
            activate_last=False,
        )

        self.dropout = nn.Dropout(dropout) if dropout else None

        self.is_classification = is_classification
        self.include_state_embedding = include_state

    def forward(
        self,
        graph: dgl.DGLGraph,
        edge_feat: torch.Tensor,
        node_feat: torch.Tensor,
        state_feat: torch.Tensor,
    ):
        """Forward pass of MEGnet. Executes all blocks.

        Args:
            graph: Input graph
            edge_feat: Edge features
            node_feat: Node features
            state_feat: State features.

        Returns:
            Prediction
        """
        # 1维还需要embedding，多维直接替换掉embedding
        # print("embedding前",node_feat,node_feat.shape)
        node_feat = self.modify_node_embedding(node_feat)
        node_feat, edge_feat, state_feat = self.embedding(node_feat, edge_feat, state_feat)

        # print("embedding后", node_feat, node_feat.shape)
        edge_feat = self.edge_encoder(edge_feat)
        node_feat = self.node_encoder(node_feat)
        # print("encoder后",node_feat,node_feat.shape)
        state_feat = self.state_encoder(state_feat)

        for block in self.blocks:
            output = block(graph, edge_feat, node_feat, state_feat)
            edge_feat, node_feat, state_feat = output

        node_vec = self.node_s2s(graph, node_feat)
        edge_vec = self.edge_s2s(graph, edge_feat)

        node_vec = torch.squeeze(node_vec)
        edge_vec = torch.squeeze(edge_vec)
        state_feat = torch.squeeze(state_feat)

        vec = torch.hstack([node_vec, edge_vec, state_feat])

        if self.dropout:
            vec = self.dropout(vec)  # pylint: disable=E1102

        output = self.output_proj(vec)
        if self.is_classification:
            output = torch.sigmoid(output)

        return torch.squeeze(output)

    def predict_structure(
        self,
        structure,
        state_feats: torch.Tensor | None = None,
        graph_converter: GraphConverter | None = None,
    ):
        """Convenience method to directly predict property from structure.

        Args:
            structure: An input crystal/molecule.
            state_feats (torch.tensor): Graph attributes
            graph_converter: Object that implements a get_graph_from_structure.

        Returns:
            output (torch.tensor): output property
        """
        if graph_converter is None:
            from matgl.ext.pymatgen import Structure2Graph

            graph_converter = Structure2Graph(element_types=self.element_types, cutoff=self.cutoff)
        g, lat, state_feats_default = graph_converter.get_graph(structure)
        g.edata["pbc_offshift"] = torch.matmul(g.edata["pbc_offset"], lat[0])
        g.ndata["pos"] = g.ndata["frac_coords"] @ lat[0]
        if state_feats is None:
            state_feats = torch.tensor(state_feats_default)
        bond_vec, bond_dist = compute_pair_vector_and_distance(g)
        g.edata["edge_attr"] = self.bond_expansion(bond_dist)
        return self(g, g.edata["edge_attr"], g.ndata["node_type"], state_feats).detach()

    def generate_element_map_from_csv(self, csv_filepath):
        """
        从CSV文件生成一个将化学元素名称映射到映射值的字典

        参数:
        csv_filepath (str): CSV文件的路径

        返回:
        dict: 包含化学元素名称映射到映射值的字典
        """
        element_map = {}
        with open(csv_filepath, mode='r', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)  # 跳过标题行
            for row in reader:
                if len(row) == 2:  # 确保每行有两个元素
                    element, value = row
                    # 假设value是字符串形式的列表，我们需要将其转换为实际的列表对象
                    # 这里使用eval来将字符串转换成Python对象，但在实际应用中需要确保安全性
                    try:
                        value = eval(value)
                    except (SyntaxError, NameError):
                        continue  # 如果转换失败，跳过这一行
                    if isinstance(value, list) and len(value) == 1:
                        element_map[element] = value[0]  # 只取列表中的单一元素
        return element_map

    def atomic_numbers_to_symbols(self, atomic_numbers):  # 输入node_feat，转化为化学元素
        element_symbols = []
        for atomic_number in atomic_numbers:
            element = Element.from_Z(atomic_number.item()+1)
            element_symbol = element.symbol
            element_symbols.append(element_symbol)
        return element_symbols

    def modify_node_embedding(self, node_feat):
        # 定义元素符号到数字的映射
        symbol_to_number = self.generate_element_map(103)
        # 调用函数将原子序数转换为元素符号
        element_symbols = self.atomic_numbers_to_symbols(node_feat)
        # 将元素符号转换为数字
        element_numbers = [symbol_to_number[symbol] for symbol in element_symbols]
        # 将数字列表转换为二维tensor
        element_tensor = torch.tensor(element_numbers).unsqueeze(1)
        return element_tensor