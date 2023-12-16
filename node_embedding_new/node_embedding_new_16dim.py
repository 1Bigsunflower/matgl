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
from _megnet_new import MEGNet
from matgl.utils.io import RemoteFile
from matgl.utils.training import ModelLightningModule
from pymatgen.core import Element
import os

# To suppress warnings for clearer output
warnings.simplefilter("ignore")

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['CUDA_DEBUG'] = '1'


def get_periodic_table_elements(element_number) -> tuple[str, ...]:  # element_number表示最多表示多少个原子序数
    """Get the tuple of elements in the periodic table.

    Returns:
        Tuple of elements in the periodic table (atomic numbers 1-103)
    """
    elements = [Element.from_Z(z).symbol for z in range(1, element_number+1)]
    return tuple(elements)


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
    # 读取数据集（完整）
    data = pd.read_json("../node_embedding/mp.2018.6.1.json")
    # data = pd.read_json("../node_embedding/first_10_data.json")
    structures = []
    mp_ids = []

    for mid, structure_str in tqdm(zip(data["material_id"], data["structure"])):
        struct = Structure.from_str(structure_str, fmt="cif")
        structures.append(struct)
        mp_ids.append(mid)

    return structures, mp_ids, data["formation_energy_per_atom"].tolist()


if __name__ == '__main__':

    if torch.cuda.is_available():
        torch.cuda.set_device(0)

    # 结构,_,标签
    structures, mp_ids, eform_per_atom = load_dataset()

    # 数据集中所有的元素类型
    elem_list = get_periodic_table_elements(103)  # 。z
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

    # define the bond expansion
    bond_expansion = BondExpansion(rbf_type="Gaussian", initial=0.0, final=5.0, num_centers=100, width=0.5)

    # 生成原子序数从1到103的元素对象，并将其转换为元素符号
    elements = [Element.from_Z(z).symbol for z in range(1, 104)]

    # 将元素符号构建成元组
    elements_tuple = tuple(elements)

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
        element_types=elements_tuple,  # 更改
        # layer_node_embedding=node_embed,
        bond_expansion=bond_expansion,
        cutoff=4.0,
        gauss_width=0.5,
    )

    # setup the MEGNetTrainer
    lit_module = ModelLightningModule(model=model)

    early_stop_callback = EarlyStopping(monitor="val_MAE", min_delta=0.00, patience=20, verbose=True, mode="min")
    # Training
    logger = CSVLogger("logs", name="MEGNet_training_16dim_mds")

    trainer = pl.Trainer(max_epochs=1000, logger=logger, callbacks=[early_stop_callback])  # 指定gpus参数为1表示使用一块GPU进行训练
    trainer.fit(model=lit_module, train_dataloaders=train_loader, val_dataloaders=val_loader)

    # 保存模型
    save_path = "saved_models/16dim_unembed"
    metadata = {"description": "MEGNet trained using 16 dimension mds",
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
