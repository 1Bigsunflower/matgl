from __future__ import annotations
import sys
import os
import shutil
import warnings
import zipfile
from typing import List
from matgl.config import DEFAULT_ELEMENTS
import dgl
import matplotlib.pyplot as plt
import pandas as pd
import pytorch_lightning as pl
import torch
from dgl.data.utils import split_dataset
from pymatgen.core import Structure
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import CSVLogger
from tqdm import tqdm

from matgl.ext.pymatgen import Structure2Graph, get_element_list
from matgl.graph.data import MEGNetDataset, MGLDataLoader, collate_fn
from matgl.layers import BondExpansion
from _megnet import MEGNet
from matgl.utils.io import RemoteFile
from matgl.utils.training import ModelLightningModule
from pymatgen.core import Element


# To suppress warnings for clearer output
warnings.simplefilter("ignore")


def get_periodic_table_elements(element_number) -> tuple[str, ...]:  # element_number表示最多表示多少个原子序数
    """Get the tuple of elements in the periodic table.

    Returns:
        Tuple of elements in the periodic table (atomic numbers 1-103)
    """
    elements = [Element.from_Z(z).symbol for z in range(1, element_number+1)]
    return tuple(elements)


def generate_element_map(max_atomic_number):
    """
    生成一个将化学元素名称映射到原子序数的字典

    参数:
    max_atomic_number (int): 要生成的原子序数的最大值

    返回:
    dict: 包含化学元素名称映射到原子序数的字典
    """
    element_map = {}
    for i in range(1, max_atomic_number+1):
        el = Element.from_Z(i)
        if el.symbol:
            element_map[el.symbol] = i+0.12
    return element_map


def atomic_numbers_to_symbols(atomic_numbers):  # 将原子序数tensor转化为元素。输入：一个tensor list的原子序数
    element_symbols = []
    for atomic_number in atomic_numbers:
        element = Element.from_Z(atomic_number.item())
        element_symbol = element.symbol
        element_symbols.append(element_symbol)
    return element_symbols


# Dataset Preparation
def load_dataset() -> tuple[list[Structure], list[str], list[float]]:
    """Raw data loading function.

    Returns:
        tuple[list[Structure], list[str], list[float]]: structures, mp_id, Eform_per_atom
    """
    # 下载数据集
    if not os.path.exists("mp.2018.6.1.json"):
        f = RemoteFile("https://figshare.com/ndownloader/files/15087992")
        with zipfile.ZipFile(f.local_path) as zf:
            zf.extractall(".")
    # 读取数据集（完整）
    data = pd.read_json("mp.2018.6.1.json")
    # data = pd.read_json("first_10_data.json")
    structures = []
    mp_ids = []

    for mid, structure_str in tqdm(zip(data["material_id"], data["structure"])):
        struct = Structure.from_str(structure_str, fmt="cif")
        structures.append(struct)
        mp_ids.append(mid)

    return structures, mp_ids, data["formation_energy_per_atom"].tolist()


# class MyMEGNET(MEGNet):
#     def __int__(
#             self,
#             dim_node_embedding: int = 1,
#             node_embeddings=None,
#             **kwargs
#     ):
#         super().__init__(dim_node_embedding=dim_node_embedding, **kwargs)
#         self.node_embeddings = node_embeddings
#
#     def forward(
#             self,
#             graph: dgl.DGLGraph,
#             edge_feat: torch.Tensor,
#             node_feat: torch.Tensor,
#             state_feat: torch.Tensor, ):
#
#         """Forward pass of MEGnet. Executes all blocks.
#
#                 Args:
#                     graph: Input graph
#                     edge_feat: Edge features
#                     node_feat: Node features
#                     state_feat: State features.
#
#                 Returns:
#                     Prediction
#                 """
#         # print("embedding前",node_feat)
#         _, edge_feat, state_feat = self.embedding(node_feat, edge_feat, state_feat)
#         # 修改node_feat的embedding
#         node_feat = self.modify_node_embedding(node_feat)
#         # print("embedding后", node_feat)
#         edge_feat = self.edge_encoder(edge_feat)
#         node_feat = self.node_encoder(node_feat)
#         state_feat = self.state_encoder(state_feat)
#
#         for block in self.blocks:
#             output = block(graph, edge_feat, node_feat, state_feat)
#             edge_feat, node_feat, state_feat = output
#
#         node_vec = self.node_s2s(graph, node_feat)
#         edge_vec = self.edge_s2s(graph, edge_feat)
#
#         node_vec = torch.squeeze(node_vec)
#         edge_vec = torch.squeeze(edge_vec)
#         state_feat = torch.squeeze(state_feat)
#
#         vec = torch.hstack([node_vec, edge_vec, state_feat])
#
#         if self.dropout:
#             vec = self.dropout(vec)  # pylint: disable=E1102
#
#         output = self.output_proj(vec)
#         if self.is_classification:
#             output = torch.sigmoid(output)
#
#         return torch.squeeze(output)
#
#     # embedding嵌入的方法
#     def modify_node_embedding(self, node_feat: torch.Tensor) -> torch.Tensor:
#         if self.node_embeddings is not None:
#             return self.node_embeddings[node_feat.long()]
#
#         # 如果没有传递嵌入表示，则使用默认的处理方式
#         modified_node_feat = []
#         for atomic_number in node_feat:
#             modified_node_feat.append(self.default_embedding())
#
#         return torch.tensor(modified_node_feat)
#
#     def default_embedding(self) -> List[float]:
#         # 默认的嵌入表示
#         return [0.0] * 1


if __name__ == '__main__':

    if torch.cuda.is_available():
        torch.cuda.set_device(0)

    structures, mp_ids, eform_per_atom = load_dataset()

    # 从整套结构中选择 100 个结构
    # structures
    # structures = structures[:100]
    # # label
    # eform_per_atom = eform_per_atom[:100]
    # get element types in the dataset
    elem_list = get_element_list(structures)
    # # 数据集中所有的元素类型
    # elem_list = get_periodic_table_elements(103)  # 。z
    # 结构转化为图
    converter = Structure2Graph(element_types=elem_list, cutoff=4.0)
    # 把原数据集转化为megnet数据集
    mp_dataset = MEGNetDataset(
        structures=structures,  # 结构
        labels={"Eform": eform_per_atom},  # 标签
        converter=converter,  # 图
        initial=0.0,  # 高斯扩展的初始距离
        final=5.0,  # 高斯扩展的最终距离
        num_centers=100,  # 高斯函数的数量
        width=0.5,  # 高斯函数的宽度
    )
    # 拆分数据集为训练、验证、测试集
    train_data, val_data, test_data = split_dataset(
        mp_dataset,
        frac_list=[0.8, 0.1, 0.1],  # 比例
        shuffle=True,
        random_state=42,
    )

    train_loader, val_loader, test_loader = MGLDataLoader(
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        collate_fn=collate_fn,
        batch_size=128,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    # 设置嵌入层
    # node_embed = torch.nn.Embedding(len(elem_list), 1)

    # define the bond expansion
    bond_expansion = BondExpansion(rbf_type="Gaussian", initial=0.0, final=5.0, num_centers=100, width=0.5)

    # setup the architecture of MEGNet model
    model = MEGNet(
        dim_node_embedding=16,
        dim_edge_embedding=100,
        dim_state_embedding=2,
        nblocks=3,
        hidden_layer_sizes_input=(64, 32),
        hidden_layer_sizes_conv=(64, 64, 32),
        nlayers_set2set=1,
        niters_set2set=2,
        hidden_layer_sizes_output=(32, 16),
        is_classification=False,
        activation_type="softplus2",
        element_types=DEFAULT_ELEMENTS,  # 更改
        # layer_node_embedding=node_embed,
        bond_expansion=bond_expansion,
        cutoff=4.0,
        gauss_width=0.5,
    )
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model.to(device)
    # setup the MEGNetTrainer
    lit_module = ModelLightningModule(model=model)

    early_stop_callback = EarlyStopping(monitor="val_MAE", min_delta=0.00, patience=20, verbose=True, mode="min")
    # Training
    logger = CSVLogger("logs", name="MEGNet_org_16dimension")
    trainer = pl.Trainer(max_epochs=1000, logger=logger, callbacks=[early_stop_callback])  # 指定gpus参数为1表示使用一块GPU进行训练
    trainer.fit(model=lit_module, train_dataloaders=train_loader, val_dataloaders=val_loader)

    # 保存模型
    save_path = "saved_models/node_original-16"
    metadata = {"description": "MEGNet trained using original node embedding 16 dimension",
                "training_set": "node embedding dimension = 16"}
    model.save(save_path, metadata=metadata)
    # 测试部分
    model.eval()
    predict = trainer.test(model=lit_module, dataloaders=test_loader)
    # print(predict)
    # This code just performs cleanup for this notebook.
    for fn in ("dgl_graph.bin", "lattice.pt", "dgl_line_graph.bin", "state_attr.pt", "labels.json"):
        try:
            os.remove(fn)
        except FileNotFoundError:
            pass

    # shutil.rmtree("logs")