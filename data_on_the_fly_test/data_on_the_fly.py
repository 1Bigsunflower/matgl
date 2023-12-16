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
from torch.utils.data import Dataset
import ijson
import pandas as pd
from sklearn.model_selection import train_test_split
from pymatgen.core.structure import Structure
from pymatgen.analysis.graphs import MoleculeGraph
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

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


class CustomDataset(Dataset):
    def __init__(self, data_path):
        self.data = pd.read_json(data_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        structure_str = self.data.iloc[idx]["structure"]
        mid = self.data.iloc[idx]["material_id"]
        formation_energy = self.data.iloc[idx]["formation_energy_per_atom"]

        struct = Structure.from_str(structure_str, fmt="cif")

        return {"structure": struct, "material_id": mid, "formation_energy": formation_energy}


class CustomDataModule(pl.LightningDataModule):
    def __init__(self, data_path, batch_size, converter, initial=0.0, final=5.0, num_centers=100, width=0.5):
        super(CustomDataModule, self).__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.converter = converter
        self.initial = initial
        self.final = final
        self.num_centers = num_centers
        self.width = width

    def setup(self, stage=None):
        self.struct, self.mid, self.formation_energy = CustomDataset(self.data_path)
        dataset = MEGNetDataset(
            structures=self.struct,
            labels={"Eform": self.formation_energy},
            converter=self.converter,
            initial=self.initial,
            final=self.final,
            num_centers=self.num_centers,
            width=self.width,
        )

        # Split the dataset into train, val, and test sets 80训练，10验证，10测试
        train_data, val_test_data = train_test_split(dataset, test_size=0.2, random_state=42)
        val_data, test_data = train_test_split(val_test_data, test_size=0.5, random_state=42)

        self.train_dataset = train_data
        self.val_dataset = val_data
        self.test_dataset = test_data

    def train_dataloader(self):
        train_loader, _, _ = MGLDataLoader(self.train_dataset, self.val_dataset, self.test_dataset,
                                           collate_fn=collate_fn, batch_size=self.batch_size, num_workers=0)
        return train_loader

    def val_dataloader(self):
        _, val_loader, _ = MGLDataLoader(self.train_dataset, self.val_dataset, self.test_dataset, collate_fn=collate_fn,
                                         batch_size=self.batch_size, num_workers=0)
        return val_loader

    def test_dataloader(self):
        _, _, test_loader = MGLDataLoader(self.train_dataset, self.val_dataset, self.test_dataset, collate_fn=collate_fn,
                                          batch_size=self.batch_size, num_workers=0)
        return test_loader


if __name__ == '__main__':
    if torch.cuda.is_available():
        torch.cuda.set_device(0)

    number_of_elements = 95

    # 数据集中所有的元素类型
    elem_list = get_periodic_table_elements(number_of_elements)  # 。z
    # 结构转化为图
    converter = Structure2Graph(element_types=elem_list, cutoff=4.0)

    # data_path = "../node_embedding/mp.2018.6.1.json"
    data_path = "../node_embedding/first_10_data.json"
    batch_size = 128
    # 数据集加载
    data_module = CustomDataModule(data_path, batch_size, converter)

    # define the bond expansion
    bond_expansion = BondExpansion(rbf_type="Gaussian", initial=0.0, final=5.0, num_centers=100, width=0.5)

    # 生成原子序数从1到103的元素对象，并将其转换为元素符号
    elements = [Element.from_Z(z).symbol for z in range(1, number_of_elements + 1)]

    # 将元素符号构建成元组
    elements_tuple = tuple(elements)

    # setup the architecture of MEGNet model
    model = MEGNet(
        dim_node_embedding=1,
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
    logger = CSVLogger("logs", name="MEGNet_training_no_nf__different_distance_dataonthefly_test")

    trainer = pl.Trainer(max_epochs=1000, logger=logger, callbacks=[early_stop_callback])  # 指定gpus参数为1表示使用一块GPU进行训练
    trainer.fit(model=lit_module, datamodule=data_module)

    # 保存模型
    save_path = "saved_models/dataonthefly_mds_unembed_unequal_distance"
    metadata = {"description": "MEGNet trained using 1 dim mds node embedding with unequal distance data on the fly",
                "training_set": "node embedding dimension = 1"}
    model.save(save_path, metadata=metadata)
    # 测试部分
    model.eval()
    predict = trainer.test(model=lit_module, datamodule=data_module)
    # print(predict)
    # This code just performs cleanup for this notebook.
    for fn in ("dgl_graph.bin", "lattice.pt", "dgl_line_graph.bin", "state_attr.pt", "labels.json"):
        try:
            os.remove(fn)
        except FileNotFoundError:
            pass

    # shutil.rmtree("logs")
