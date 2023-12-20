from tofunet.utils import Custom_Dataset, pad_collate
from tofunet.pl_modules import LitTrainer, LitTTA
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
import lightning.pytorch as pl
import hydra
from omegaconf import DictConfig
import torch
from torch.utils.data import DataLoader

import lightning as L
L.seed_everything(1)
import json

# Ensure that all operations are deterministic on GPU (if used) for
# reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# import os
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:5000"

def tensor_to_list(nested_tensor):
    if isinstance(nested_tensor, torch.Tensor):
        return nested_tensor.tolist()
    elif isinstance(nested_tensor, list):
        return [tensor_to_list(item) for item in nested_tensor]
    else:
        raise ValueError("Unsupported type.")

def cut_tofu(x):
    batch_size = x.size(0)
    x_lin_ipt = []
    xi = x # [:, 0, :, :, :]
    xi_slices = []
    for i in range(xi.size(2)):
        xi_slices.append(xi[:, :, i, :, :])
    for i in range(xi.size(3)):
        xi_slices.append(xi[:, :, :, i, :])
    for i in range(xi.size(4)):
        xi_slices.append(xi[:, :, :, :, i])
    
    batch_size, n = xi.shape[0], xi.shape[-1]
    mask_i_j = torch.eye(n, dtype=torch.bool).unsqueeze(-1).expand(n, n, n).unsqueeze(0).expand(batch_size, n, n, n).unsqueeze(1).expand(batch_size, 1, n, n, n)
    mask_j_k = torch.eye(n, dtype=torch.bool).unsqueeze(0).expand(n, n, n).unsqueeze(0).expand(batch_size, n, n, n).unsqueeze(1).expand(batch_size, 1, n, n, n)
    mask_i_k = torch.eye(n, dtype=torch.bool).unsqueeze(1).expand(n,n,n).unsqueeze(0).expand(batch_size, n, n, n).unsqueeze(1).expand(batch_size, 1, n, n, n)
    mask_i_n1_j = torch.eye(n, dtype=torch.bool).flip(0).unsqueeze(-1).expand(n, n, n).unsqueeze(0).expand(batch_size, n, n, n).unsqueeze(1).expand(batch_size, 1, n, n, n)
    mask_j_n1_k = torch.eye(n, dtype=torch.bool).flip(0).unsqueeze(0).expand(n, n, n).unsqueeze(0).expand(batch_size, n, n, n).unsqueeze(1).expand(batch_size, 1, n, n, n)
    mask_i_n1_k = torch.eye(n, dtype=torch.bool).flip(0).unsqueeze(1).expand(n, n, n).unsqueeze(0).expand(batch_size, n, n, n).unsqueeze(1).expand(batch_size, 1, n, n, n)
    
    result_i_j = xi[mask_i_j].reshape(batch_size, 1, n, n)
    result_j_k = xi[mask_j_k].reshape(batch_size, 1, n, n)
    result_i_k = xi[mask_i_k].reshape(batch_size, 1, n, n)
    result_i_n1_j = xi[mask_i_n1_j].reshape(batch_size, 1, n, n)
    result_j_n1_k = xi[mask_j_n1_k].reshape(batch_size, 1, n, n)
    result_i_n1_k = xi[mask_i_n1_k].reshape(batch_size, 1, n, n)
    
    xi_slices.append(result_i_j)
    xi_slices.append(result_j_k)
    xi_slices.append(result_i_k)
    xi_slices.append(result_i_n1_j)
    xi_slices.append(result_j_n1_k)
    xi_slices.append(result_i_n1_k)
    
    xi_stacked = torch.stack(xi_slices, dim=1)
    return xi_stacked


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    print(cfg)
# for task in mb.tasks:
    # task.load()
    # for fold in task.folds:
    for fold in range(1):
        model = LitTTA.load_from_checkpoint(cfg.task.load_model_name[fold], map_location=torch.device('cpu'))
        print("Scale_factor:", model.encoder.pointwise.scale_factors)
        print("biases: ", model.encoder.pointwise.biases)
        # print(model)
        # Current implement of TTA only need 1 GPU, more GPU will cause order error
        # trainer = pl.Trainer(devices=1, precision=cfg.task.precision)
        data_number = cfg.task.test_numbers[fold] * 1
        aug_times = 1 

        data_path = cfg.task.data_folder + "/test"
        test_set = Custom_Dataset(
            data_path,
            fold,
            data_number,
            shuffle_atoms=True,
            d_proj=cfg.task.d_projection,
            aug_times=aug_times)
        model.freeze()
        
        batch_size = 1
        dataloader=DataLoader(
            test_set,
            batch_size=batch_size,
            num_workers=1,
            collate_fn=pad_collate)
        last_lst = []
        after_gru_fc_lst= []
        after_gru_fc_conv2d_lst = []
        for i, (X, Y, length) in enumerate(dataloader):
            # ret0 = model(X, length)
            # ret1 = model.encoder(X, length)
            x = model.encoder.pointwise(X, length)
            after_gru_fc_lst.append([tensor_to_list(x.view(batch_size, -1)), tensor_to_list(Y)])
            x = cut_tofu(x)
            xi_out = model.encoder.conv2lin_0(x.view(-1, 1, 8, 8))
            x_lin_ipt = []
            x_lin_ipt.append(xi_out.view(batch_size, -1, 1, 1))
            x = torch.cat(x_lin_ipt, dim=1)
            x = x.view(batch_size, -1)
            after_gru_fc_conv2d_lst.append([tensor_to_list(x), tensor_to_list(Y)])
            ret2 = model.encoder.lin(x)
            last_lst.append([tensor_to_list(ret2), tensor_to_list(Y)])
            
            if len(last_lst) % 100 == 0:
                print(len(last_lst))
            # if len(last_lst) % 1000 == 0:
            #     break
        with open('after_gru_fc_conv2d_lst.json', 'w') as f:
            json.dump(after_gru_fc_conv2d_lst, f)
        with open('after_gru_fc_lst.json', 'w') as f:
            json.dump(after_gru_fc_lst, f)
        with open('last_lst.json', 'w') as f:
            json.dump(last_lst, f)
   
if __name__ == "__main__":
    main()
