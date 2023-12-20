import os
from hydra.utils import instantiate
import lightning.pytorch as pl
import time
import json
import torch
from torch import nn
from sklearn.metrics import roc_auc_score
import numpy as np
import lightning as L
L.seed_everything(1)

# Ensure that all operations are deterministic on GPU (if used) for
# reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class LitTrainer(pl.LightningModule):
    def __init__(self, cfg, cur_fold=0):
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters()
        self.encoder = instantiate(cfg.task.model)
        # self.example_input_array = torch.zeros((64, 3, 16, 16, 16), dtype=torch.float32)
        self.validation_step_outputs = []
        self.test_step_outputs = []
        # self.grad_max = 0
        # self.grad_min = 0
        # self.grad_mean = 0
        self.cur_fold = cur_fold

    def inverse_yeo_johnson(self, y, lam):
        """Inverse Yeo-Johnson transformation."""
        pos = y >= 0
        x_inv = torch.zeros_like(y, dtype=y.dtype)
    
        # For positive y
        pos_val = y[pos]
        if lam != 0:
            x_inv[pos] = (torch.exp(torch.log(lam * pos_val + 1) / lam) - 1).to(dtype=x_inv.dtype)
        else:
            x_inv[pos] = (torch.exp(pos_val) - 1).to(dtype=x_inv.dtype)

        # For negative y
        neg_val = y[~pos]
        if lam != 2:
            x_inv[~pos] = (1 - torch.exp(torch.log(-((2 - lam) * neg_val) + 1) / (2 - lam))).to(dtype=x_inv.dtype)
        else:
            x_inv[~pos] = (1 - torch.exp(-neg_val)).to(dtype=x_inv.dtype)
    
        return x_inv
    
    def forward(self, x, length):
        # Forward function that is run when visualizing the graph
        return self.encoder(x, length)

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        x, y, lengths = batch
        z = self.encoder(x, lengths.cpu())
        if self.cfg.task.task_type == "cls":
            bce_with_logits_loss = torch.nn.BCEWithLogitsLoss(reduction='none')
            loss = bce_with_logits_loss(z, y.float().view(-1, 1))
            # roc_auc = roc_auc_score(
            #     y.cpu().detach().numpy(),
            #     torch.sigmoid(z).cpu().detach().float().numpy())
            # self.log("train_roc_auc", roc_auc)
            self.log("train_loss", loss.mean())
            return loss.sum()
        elif self.cfg.task.task_type == "reg":
            criterion = nn.MSELoss()
            loss = criterion(z, y.float().view(-1, 1))
            self.log("train_loss", loss)
            return loss
        else:
            pass

    def test_step(self, batch, batch_idx):
        # this is the test loop
        x, y, lengths = batch

        z = self.encoder(x, lengths.cpu())
        if self.cfg.task.task_type == "cls":
            z = torch.sigmoid(z)
        else:
            # reverse the coxbox and norm
            z = (z * self.cfg.task.train_std[self.cur_fold]) +self.cfg.task.train_mean[self.cur_fold]
            if self.cfg.task.lambda_best_fit[self.cur_fold] is None:
                pass
            else:
                z = self.inverse_yeo_johnson(z, self.cfg.task.lambda_best_fit[self.cur_fold])
                # if self.cfg.task.lambda_best_fit[self.cur_fold] != 0:
                #     z = (self.cfg.task.lambda_best_fit[self.cur_fold] * z + 1) ** (1 / self.cfg.task.lambda_best_fit[self.cur_fold])
                # else:
                #     z = np.exp(z)

        self.test_step_outputs.append([z, y])

    def on_test_epoch_end(self):
        all_preds = self.test_step_outputs
        all_predictions = []
        all_labels = []
        for output in all_preds:
            all_predictions.extend(output[0].cpu().detach().float().numpy())
            all_labels.extend(output[1].cpu().detach().numpy())

        if self.cfg.task.task_type == "cls":
            roc_auc = roc_auc_score(all_labels, all_predictions)
            self.log("test_roc_auc", roc_auc)
        elif self.cfg.task.task_type == "reg":
            criterion = nn.L1Loss()
            loss = criterion(torch.tensor(all_predictions), torch.tensor(all_labels).float().view(-1,1))
            self.log("test_loss", loss)
        else:
            pass
        
        time.sleep(1)
        self.test_step_outputs.clear()  # free memory

    def validation_step(self, batch, batch_idx):
        # this is the validation loop
        x, y, lengths = batch

        z = self.encoder(x, lengths.cpu())
        if self.cfg.task.task_type == "cls":
            bce_with_logits_loss = torch.nn.BCEWithLogitsLoss(reduction='none')
            val_loss = bce_with_logits_loss(z, y.float().view(-1, 1))

            self.log("val_loss", val_loss.mean(), sync_dist=True)
            self.validation_step_outputs.append([z, y])
        elif self.cfg.task.task_type == "reg":
            # reverse the coxbox and norm 
            z = (z * self.cfg.task.train_std[self.cur_fold]) +self.cfg.task.train_mean[self.cur_fold]
            if self.cfg.task.lambda_best_fit[self.cur_fold] is None:
                pass
            else:
                z = self.inverse_yeo_johnson(z, self.cfg.task.lambda_best_fit[self.cur_fold])
                # if self.cfg.task.lambda_best_fit[self.cur_fold] != 0:
                #     z = (self.cfg.task.lambda_best_fit[self.cur_fold] * z + 1) ** (1 / self.cfg.task.lambda_best_fit[self.cur_fold])
                # else:
                #     z = np.exp(z)
            
            criterion = nn.L1Loss()
            loss = criterion(z, y.float().view(-1, 1))
            self.log("val_loss", loss, sync_dist=True)
            self.validation_step_outputs.append([z, y])

    def on_validation_epoch_end(self):
        all_preds = self.validation_step_outputs
        all_predictions = []
        all_labels = []
        for output in all_preds:
            all_predictions.extend(output[0].cpu().detach().float().numpy())
            all_labels.extend(output[1].cpu().detach().numpy())


        if self.cfg.task.task_type == "cls":
            bce_with_logits_loss = torch.nn.BCEWithLogitsLoss(reduction='none')
            loss = bce_with_logits_loss(torch.tensor(np.array(all_predictions)), torch.tensor(np.array(all_labels)).float().view(-1,1)).mean()
            # loss = roc_auc_score(all_labels, all_predictions)
        elif self.cfg.task.task_type == "reg":
            criterion = nn.L1Loss()
            loss = criterion(torch.tensor(np.array(all_predictions)), torch.tensor(np.array(all_labels)).float().view(-1,1))
            
        self.log(self.cfg.task.monitor, loss, sync_dist=True)
        self.validation_step_outputs.clear()
        torch.cuda.empty_cache()

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
        optimizer.step(closure=optimizer_closure)
        # record grad information
        # if batch_idx == 0:
        #     gradients = []
        #     for param in self.parameters():
        #         if param.grad is not None:
        #             gradients.append(param.grad.view(-1))
        #     gradients = torch.cat(gradients)
        #     gradients = torch.abs(gradients)
        #     self.grad_max = gradients.max()
        #     self.grad_min = gradients.min()
        #     self.grad_mean = gradients.mean()
        # self.log("grad_max", self.grad_max)
        # self.log("grad_min", self.grad_min)
        # self.log("grad_mean", self.grad_mean)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.cfg.task.lr, weight_decay=0.01)
        # optimizer = torch.optim.AdamW([
        #         {"params": self.encoder.pointwise.gru.parameters()},
        #         {"params": self.encoder.pointwise.scale_factors, "weight_decay": 0.0},
        #         {"params": self.encoder.pointwise.biases, "weight_decay": 0.0},
        #         {"params": self.encoder.pointwise.fc.parameters()},
        #         {"params": self.encoder.conv2lin_0.parameters()},
        #         {"params": self.encoder.lin.parameters()},
        #     ], lr=self.cfg.task.lr, weight_decay=0.01)
        # Using a scheduler is optional but can be helpful.
        # The scheduler reduces the LR if the validation performance hasn't
        # improved for the last N epochs
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=self.cfg.task.monitor_mode,
            factor=0.5,
            patience=self.cfg.task.lr_decay_patience,
            min_lr=self.cfg.task.min_lr)
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": self.cfg.task.monitor}


class LitTTA(LitTrainer):
    def test_step(self, batch, batch_idx):
        # this is the test loop
        x, y, lengths = batch
        assert y.eq(y[0]).all(), "All elements in y should be the same across the batch"
        assert lengths.eq(lengths[0]).all(), "All elements in lengths should be the same across the batch"

        z = self.encoder(x, lengths.cpu())
        if self.cfg.task.task_type == "cls":
            z = torch.sigmoid(z)
        else:
            # reverse the coxbox and norm
            z = (z * self.cfg.task.train_std[self.cur_fold]) +self.cfg.task.train_mean[self.cur_fold]
            if self.cfg.task.lambda_best_fit[self.cur_fold] is None:
                pass
            else:
                z = self.inverse_yeo_johnson(z, self.cfg.task.lambda_best_fit[self.cur_fold])
                # if self.cfg.task.lambda_best_fit[self.cur_fold] != 0:
                #     z = (self.cfg.task.lambda_best_fit[self.cur_fold] * z + 1) ** (1 / self.cfg.task.lambda_best_fit[self.cur_fold])
                # else:
                #     z = np.exp(z)

        self.test_step_outputs.append([z, y])

    def on_test_epoch_end(self):
        aug_times = self.cfg.task.test_aug_times
        all_preds = self.test_step_outputs
        all_predictions = []
        all_labels = []
        for output in all_preds:
            all_predictions.extend(output[0].cpu().detach().float().numpy())
            all_labels.extend(output[1].cpu().detach().numpy())

        ret_p = []
        ret_l = []
        ret_tta_std = []
        for i in range(0, len(all_predictions), aug_times):
            batch_preds = all_predictions[i:i + aug_times]
            batch_labels = all_labels[i:i + aug_times]
            ret_p.append(np.mean(batch_preds))
            ret_tta_std.append(np.std(batch_preds))
            # all should be same in batch_labels
            ret_l.append(batch_labels[0])
        if self.cfg.task.task_type == "cls":
            ret_l = [int(x) for x in ret_l]
            loss = roc_auc_score(ret_l, ret_p)
        elif self.cfg.task.task_type == "reg":
            criterion = nn.L1Loss()
            # for reg task loss should be same with loss1
            loss = criterion(torch.tensor(ret_p).float().view(-1,1), torch.tensor(ret_l).float().view(-1,1))
            # loss1 = criterion(torch.tensor(all_predictions), torch.tensor(all_labels).float().view(-1,1))
        
        directory = os.path.dirname(self.cfg.task.load_model_name[self.cur_fold])
        directory = os.path.dirname(directory)
        log_dir = directory
        list_path = f"{log_dir}/predictions.json"
        print("Dump file:", list_path)
        ret_p = [float(item) for item in ret_p]
        with open(list_path, 'w') as file:
            json.dump(ret_p, file)
        self.log("test_result", loss)
        self.log("tta_std", np.mean(ret_tta_std))
        time.sleep(1)
        self.test_step_outputs.clear()  # free memory
