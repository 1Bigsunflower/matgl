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
# from custom_dataset import Custom_Dataset

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

folder = "data_save"


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


def gen_and_dump(name, dataset):
    io_buffer = []  # 用于存储生成的数据字典的缓冲区
    io_cnt = 0  # 计数生成的文件数量
    for i in range(len(dataset)):
        io_buffer.append(dataset[i])
        if len(io_buffer) > 100 or i == len(dataset) - 1:
            for data in io_buffer:
                save_path = "{}/{}/{}.pth".format(folder, name, io_cnt)
                # Check if the directory exists, and create it if not
                if not os.path.exists(os.path.dirname(save_path)):
                    os.makedirs(os.path.dirname(save_path))

                # Save the data to the specified path
                torch.save(data, save_path)
                io_cnt += 1
            io_buffer.clear()


def main():
    if not os.path.exists(os.path.dirname(folder)):
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
        gen_and_dump(name='train', dataset=train_data)
        gen_and_dump(name='val', dataset=val_data)
        gen_and_dump(name='test', dataset=test_data)

        for fn in ("dgl_graph.bin", "lattice.pt", "dgl_line_graph.bin", "state_attr.pt", "labels.json"):
            try:
                os.remove(fn)
            except FileNotFoundError:
                pass

    # data_path = 'data_save/train'
    # print(count_files_in_folder(data_path))
    # train_set = Custom_Dataset(data_path, count_files_in_folder(data_path))
    # data_path = 'data_save/val'
    # val_set = Custom_Dataset(data_path, count_files_in_folder(data_path))
    # data_path = 'data_save/test'
    # test_set = Custom_Dataset(data_path, count_files_in_folder(data_path))
    # train_loader, val_loader, test_loader = MGLDataLoader(
    #     train_data=train_set,
    #     val_data=val_set,
    #     test_data=test_set,
    #     collate_fn=collate_fn,
    #     batch_size=182,
    #     num_workers=0,
    # )
    #
    # # define the bond expansion
    # bond_expansion = BondExpansion(rbf_type="Gaussian", initial=0.0, final=5.0, num_centers=100, width=0.5)
    #
    # # 生成原子序数从1到103的元素对象，并将其转换为元素符号
    # elements = [Element.from_Z(z).symbol for z in range(1, 104)]
    #
    # # 将元素符号构建成元组
    # elements_tuple = tuple(elements)
    #
    # # setup the architecture of MEGNet model
    # model = MEGNet(
    #     dim_node_embedding=1,
    #     dim_edge_embedding=100,
    #     dim_state_embedding=2,
    #     nblocks=3,
    #     hidden_layer_sizes_input=(64, 32),
    #     hidden_layer_sizes_conv=(64, 64, 32),
    #     nlayers_set2set=1,
    #     niters_set2set=2,
    #     hidden_layer_sizes_output=(32, 16),
    #     is_classification=False,
    #     activation_type="softplus2",
    #     element_types=elements_tuple,  # 更改
    #     # layer_node_embedding=node_embed,
    #     bond_expansion=bond_expansion,
    #     cutoff=4.0,
    #     gauss_width=0.5,
    # )
    #
    # # setup the MEGNetTrainer
    # lit_module = ModelLightningModule(model=model)
    #
    # early_stop_callback = EarlyStopping(monitor="val_MAE", min_delta=0.00, patience=20, verbose=True, mode="min")
    # # Training
    # logger = CSVLogger("logs", name="MEGNet_training_no_nf_different_distance")
    #
    # trainer = pl.Trainer(max_epochs=2)  # , logger=logger, callbacks=[early_stop_callback])  # 指定gpus参数为1表示使用一块GPU进行训练
    # trainer.fit(model=lit_module, train_dataloaders=train_loader, val_dataloaders=val_loader)


if __name__ == '__main__':
    main()
