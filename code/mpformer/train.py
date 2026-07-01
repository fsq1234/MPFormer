import os
import datetime
import torch
from mpformer.data_provider import datasets_factory
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from mpformer.models.model_factory import Model
from torch.cuda.amp import autocast
from mpformer.utils.data_normalization import ZNorm
from torchinfo import summary
from pytorch_lightning.loggers import WandbLogger

def train_pytorch_loader(configs):
    norm = ZNorm()
    wandb_logger = WandbLogger(
        project=getattr(configs, 'wandb_project', 'mpformer'),
        name=getattr(configs, 'wandb_name', None),
        save_dir=getattr(configs, 'wandb_save_dir', 'wandb_logs'),
        log_model=False,
    )
    wandb_logger.experiment.config.update(vars(configs), allow_val_change=True)

    train_loader = datasets_factory.data_provider(configs)
    # print(len(train_loader))
    val_loader = datasets_factory.data_provider_val(configs)
    # print(len(val_loader))

    # 是否标准化
    # train_loader.dataset.transform = norm
    # val_loader.dataset.transform = norm

    model = Model(configs)
    
    # 如果配置中有预训练权重路径，加载预训练权重
    if configs.pretrained_model and os.path.exists(configs.pretrained_model):
        print(f"Loading pretrained weights from {configs.pretrained_model}")
        checkpoint = torch.load(configs.pretrained_model)
        model.load_state_dict(checkpoint['state_dict'], strict=False)  # 加载预训练权重
        print("Pretrained weights loaded successfully.")

        # # 添加 adapter 结构到模型中
        # model = add_adapter(model, configs)
        # print("Adapter added to the model.")
        
    if getattr(configs, 'show_summary', False):
        input_size = (1, configs.input_length, configs.img_height, configs.img_width, configs.img_ch)
        print("Model Architecture:")
        summary(model, input_size=input_size, col_names=("input_size", "output_size", "num_params", "trainable"), depth=6)
    lr_monitor = LearningRateMonitor(logging_interval='step')
    checkpoint_callback = ModelCheckpoint(
        dirpath=configs.checkpoint_dir,
        filename='mpformer-{epoch:02d}-{val_neigh_csi2:.2f}',
        save_top_k=1, # 保存最好的一个模型
        mode='max',
        monitor='val_neigh_csi2'
    )

    trainer_kwargs = dict(
        max_epochs=configs.epochs,
        num_nodes=1,
        callbacks=[checkpoint_callback, lr_monitor],
        log_every_n_steps=configs.log_interval,
        sync_batchnorm=True if torch.cuda.device_count() > 1 else False,
        logger=wandb_logger,
    )
    if torch.cuda.device_count() > 1:
        trainer_kwargs['strategy'] = 'ddp_find_unused_parameters_true'

    trainer = Trainer(**trainer_kwargs)
    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader
    )

