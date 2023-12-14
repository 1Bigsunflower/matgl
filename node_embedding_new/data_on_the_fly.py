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
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import pandas as pd
from tqdm import tqdm
from pymatgen.core import Structure

# To suppress warnings for clearer output
warnings.simplefilter("ignore")

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['CUDA_DEBUG'] = '1'


def get_periodic_table_elements(element_number) -> tuple[str, ...]:  # element_number表示最多表示多少个原子序数
    """Get the tuple of elements in the periodic table.

    Returns:
        Tuple of elements in the periodic table (atomic numbers 1-103)
    """
    elements = [Element.from_Z(z).symbol for z in range(1, element_number + 1)]
    return tuple(elements)


def atomic_numbers_to_symbols(atomic_numbers):  # 将原子序数tensor转化为元素。输入：一个tensor list的原子序数
    element_symbols = []
    for atomic_number in atomic_numbers:
        element = Element.from_Z(atomic_number.item())
        element_symbol = element.symbol
        element_symbols.append(element_symbol)
    return element_symbols


def load_dataset_chunked(chunk_size=1000) -> tuple[list[Structure], list[str], list[float]]:
    """分块加载数据的函数。

    Returns:
        tuple[list[Structure], list[str], list[float]]: 结构体，mp_id，Eform_per_atom
    """
    data_reader = pd.read_json("../node_embedding/mp.2018.6.1.json", lines=True, chunksize=chunk_size)

    structures = []
    mp_ids = []
    eform_per_atom = []

    for chunk in tqdm(data_reader):
        for mid, structure_str, eform in zip(chunk["material_id"], chunk["structure"],
                                             chunk["formation_energy_per_atom"]):
            struct = Structure.from_str(structure_str, fmt="cif")
            structures.append(struct)
            mp_ids.append(mid)
            eform_per_atom.append(eform)

    return structures, mp_ids, eform_per_atom


if __name__ == '__main__':
    if torch.cuda.is_available():
        torch.cuda.set_device(0)

    structures, mp_ids, eform_per_atom = load_dataset_chunked()

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
    train_loader, val_loader, test_loader = DataLoader(
        dataset=mp_dataset,
        batch_size=128,
        num_workers=4,  # 根据你的系统配置调整此值
        collate_fn=collate_fn,
        shuffle=True,
        pin_memory=torch.cuda.is_available()
    )

    # ...（你的代码的其余部分保持不变）

    trainer = pl.Trainer(max_epochs=1000, logger=logger, callbacks=[early_stop_callback],
                         gpus=1 if torch.cuda.is_available() else None)
    trainer.fit(model=lit_module, train_dataloaders=train_loader, val_dataloaders=val_loader)

    # ...（你的代码的其余部分保持不变）

    model.eval()
    predict = trainer.test(model=lit_module, dataloaders=test_loader)
