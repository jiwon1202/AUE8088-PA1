# Python packages
from termcolor import colored
from typing import Dict
import copy

# PyTorch & Pytorch Lightning
from lightning.pytorch import LightningModule
from lightning.pytorch.loggers.wandb import WandbLogger
from torch import nn
from torchvision import models
from torchvision.models.alexnet import AlexNet
import torch

# Custom packages
from src.metric import MyAccuracy, MyF1Score
import src.config as cfg
from src.util import show_setting
import torch
import timm
import torch.nn as nn
from einops.layers.torch import Rearrange
import torch.nn.functional as F
import timm

class SimpleClassifier(LightningModule):
    def __init__(self,
                 model_name: str = 'resnet18',
                 num_classes: int = 200,
                 optimizer_params: Dict = dict(),
                 scheduler_params: Dict = dict(),
        ):
        super().__init__()


        # Network
        if model_name == 'Astroformer':
            self.model = timm.models.create_model('astroformer_0',  pretrained=False)
        else:
            models_list = models.list_models()
            assert model_name in models_list, f'Unknown model name: {model_name}. Choose one from {", ".join(models_list)}'
            self.model = models.get_model(model_name, num_classes=num_classes)

        # Loss function
        # self.loss_fn = nn.CrossEntropyLoss()
        self.loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

        # Metric
        self.accuracy = MyAccuracy()
        self.f1score = MyF1Score(num_classes=200)

        # Hyperparameters
        self.save_hyperparameters()

    def on_train_start(self):
        show_setting(cfg)

    def configure_optimizers(self):
        optim_params = copy.deepcopy(self.hparams.optimizer_params)
        optim_type = optim_params.pop('type')
        optimizer = getattr(torch.optim, optim_type)(self.parameters(), **optim_params)

        # scheduler_params = copy.deepcopy(self.hparams.scheduler_params)
        # scheduler_type = scheduler_params.pop('type')
        # scheduler = getattr(torch.optim.lr_scheduler, scheduler_type)(optimizer, **scheduler_params)

        from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

        total_epochs    = 100         # t_initial
        warmup_epochs   = 5           # warmup_t
        base_lr         = optimizer.defaults['lr']   # optimizer에 설정된 초기 lr (예: 0.01)
        warmup_lr_init  = 0.005       # warmup_lr_init
        min_lr          = 1e-4        # lr_min (0.01 * 1e-2)
        T_max           = total_epochs - warmup_epochs  # 코사인 감소 기간 (95 에폭)

        warmup_scheduler = LinearLR(
            optimizer,
            start_factor = warmup_lr_init / base_lr,
            total_iters  = warmup_epochs
        )

        cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max   = T_max,
            eta_min = min_lr
        )

        scheduler = SequentialLR(
            optimizer,
            schedulers = [warmup_scheduler, cosine_scheduler],
            milestones = [warmup_epochs] 
        )
        
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        loss, scores, y = self._common_step(batch)
        accuracy = self.accuracy(scores, y)
        f1 = self.f1score(scores, y)
        self.log_dict({'loss/train': loss, 'accuracy/train': accuracy, 'f1/train': f1},
                      on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, scores, y = self._common_step(batch)
        accuracy = self.accuracy(scores, y)
        f1 = self.f1score(scores, y)
        self.log_dict({'loss/val': loss, 'accuracy/val': accuracy, 'f1/val': f1},
                      on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self._wandb_log_image(batch, batch_idx, scores, frequency = cfg.WANDB_IMG_LOG_FREQ)

    def _common_step(self, batch):
        x, y = batch
        scores = self.forward(x)
        loss = self.loss_fn(scores, y)
        return loss, scores, y

    def _wandb_log_image(self, batch, batch_idx, preds, frequency = 100):
        if not isinstance(self.logger, WandbLogger):
            if batch_idx == 0:
                self.print(colored("Please use WandbLogger to log images.", color='blue', attrs=('bold',)))
            return

        if batch_idx % frequency == 0:
            x, y = batch
            preds = torch.argmax(preds, dim=1)
            self.logger.log_image(
                key=f'pred/val/batch{batch_idx:5d}_sample_0',
                images=[x[0].to('cpu')],
                caption=[f'GT: {y[0].item()}, Pred: {preds[0].item()}'])
