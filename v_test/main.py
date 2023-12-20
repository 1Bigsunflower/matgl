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

# Ensure that all operations are deterministic on GPU (if used) for
# reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# import os
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:5000"

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    print(cfg)
# for task in mb.tasks:
    # task.load()
    # for fold in task.folds:
    for fold in range(5):
        if fold not in cfg.fold:
            continue
        data_path = cfg.task.data_folder + "/train"
        batch_size = cfg.task.batch_size
        train_set = Custom_Dataset(
            data_path,
            fold,
            cfg.task.train_numbers[fold],
            shuffle_atoms=True,
            d_proj=cfg.task.d_projection)
        train_loader = DataLoader(
            train_set,
            batch_size=batch_size,
            drop_last=True,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
            collate_fn=pad_collate)

        batch_size = cfg.task.batch_size
        data_path = cfg.task.data_folder + "/vali"
        vali_set = Custom_Dataset(
            data_path,
            fold,
            cfg.task.vali_numbers[fold] *
            cfg.task.vali_aug_times,
            shuffle_atoms=True,
            d_proj=cfg.task.d_projection,
            aug_times=cfg.task.vali_aug_times)
        valid_loader = DataLoader(
            vali_set,
            batch_size=batch_size,
            num_workers=4,
            collate_fn=pad_collate)
        if cfg.train_or_tta == "train":
            model = LitTrainer(cfg=cfg, cur_fold=fold)
            trainer = pl.Trainer(max_epochs=cfg.task.max_epoch, detect_anomaly=True, precision=cfg.task.precision,
                                 num_sanity_val_steps=-1,
                                 devices=cfg.task.pl_devices,
                                 num_nodes=cfg.task.num_nodes,
                                 reload_dataloaders_every_n_epochs=20,
                                 sync_batchnorm=True,
                                 strategy='ddp',  # _find_unused_parameters_true',
                                 # gradient_clip_val=10,
                                 # gradient_clip_algorithm="value",
                                 callbacks=[LearningRateMonitor("epoch"),
                                     #        EarlyStopping(
                                     # monitor=cfg.task.monitor,
                                     # mode=cfg.task.monitor_mode,
                                     # patience=cfg.task.early_stopping_patience),
                                     ModelCheckpoint(
                                     save_weights_only=True,
                                     monitor=cfg.task.monitor,
                                     mode=cfg.task.monitor_mode,
                                     save_top_k=1,
                                     filename='{epoch}-{val_loss:.4f}-{' + \
                                     cfg.task.monitor + ':.4f}'
                                 )
                                 ]
                                 )

            # If True, we plot the computation graph in tensorboard
            trainer.logger._log_graph = False
            # trainer.logger._version = "version_latt_at_last"
            trainer.fit(model, train_loader, valid_loader)
            checkpoint = torch.load(trainer.checkpoint_callback.best_model_path)
            model.load_state_dict(checkpoint['state_dict'])
            data_number = cfg.task.test_numbers[fold]
            aug_times = 1
        else:
            print("Load model:", cfg.task.load_model_name[fold])
            model = LitTTA.load_from_checkpoint(cfg.task.load_model_name[fold])
            model.cfg = cfg
            # Current implement of TTA only need 1 GPU, more GPU will cause order error
            trainer = pl.Trainer(devices=1, precision=cfg.task.precision)
            batch_size = cfg.task.test_aug_times
            data_number = cfg.task.test_numbers[fold] * cfg.task.test_aug_times
            aug_times = cfg.task.test_aug_times

        data_path = cfg.task.data_folder + "/test"
        test_set = Custom_Dataset(
            data_path,
            fold,
            data_number,
            shuffle_atoms=True,
            d_proj=cfg.task.d_projection,
            aug_times=aug_times)
        
        trainer.test(
            model,
            dataloaders=DataLoader(
                test_set,
                batch_size=batch_size,
                num_workers=4,
                collate_fn=pad_collate))


if __name__ == "__main__":
    main()
