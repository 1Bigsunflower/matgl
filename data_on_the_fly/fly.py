from __future__ import annotations

import os
import warnings

import pandas as pd
import pytorch_lightning as pl
import torch
from dgl.data.utils import split_dataset
from matgl.ext.pymatgen import Structure2Graph
from matgl.graph.data import MEGNetDataset, MGLDataLoader, collate_fn
from matgl.layers import BondExpansion
from matgl.utils.training import ModelLightningModule
from pymatgen.core import Element
from pymatgen.core import Structure
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import CSVLogger
from tqdm import tqdm

from _megnet_new import MEGNet
from custom_dataset import Custom_Dataset

# To suppress warnings for clearer output
warnings.simplefilter("ignore")

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['CUDA_DEBUG'] = '1'


# 检查目录下文件数量
def count_files_in_folder(folder_path):

    files_iterator = os.scandir(folder_path)
    file_count = sum(entry.is_file() for entry in files_iterator)
    return file_count


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


# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# 存储数据集的路径
folder = "data_save"


def main():

    data_path = 'data_save/train'
    train_set = Custom_Dataset(data_path, count_files_in_folder(data_path))
    data_path = 'data_save/val'
    val_set = Custom_Dataset(data_path, count_files_in_folder(data_path))
    data_path = 'data_save/test'
    test_set = Custom_Dataset(data_path, count_files_in_folder(data_path))
    train_loader, val_loader, test_loader = MGLDataLoader(
        train_data=train_set,
        val_data=val_set,
        test_data=test_set,
        collate_fn=collate_fn,
        batch_size=128,
        num_workers=0,
    )

    # define the bond expansion
    bond_expansion = BondExpansion(rbf_type="Gaussian", initial=0.0, final=5.0, num_centers=100, width=0.5)

    # 生成原子序数从1到103的元素对象，并将其转换为元素符号
    elements = [Element.from_Z(z).symbol for z in range(1, 104)]

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

    early_stop_callback = EarlyStopping(monitor="val_MAE", min_delta=0.00, patience=10, verbose=True, mode="min")
    # Training
    logger = CSVLogger("logs", name="Fly_test")

    trainer = pl.Trainer(max_epochs=100, logger=logger, callbacks=[early_stop_callback])  # 指定gpus参数为1表示使用一块GPU进行训练
    trainer.fit(model=lit_module, train_dataloaders=train_loader, val_dataloaders=val_loader)

    # # 保存模型
    # save_path = "saved_models/1dim_fly_test"
    # metadata = {"description": "MEGNet trained using mds embedding with 1 dimension and unequal distance",
    #             "training_set": "node embedding dimension = 1"}
    # model.save(save_path, metadata=metadata)
    # 测试部分
    model.eval()
    predict = trainer.test(model=lit_module, dataloaders=test_loader)
    print(predict)


if __name__ == '__main__':
    main()
