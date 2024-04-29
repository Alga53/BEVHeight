from argparse import ArgumentParser, Namespace

import os
import mmcv
import pytorch_lightning as pl
import torch
import torch.nn.parallel
import torch.utils.data
import torch.utils.data.distributed
import torchvision.models as models
from pytorch_lightning.core import LightningModule
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.optim.lr_scheduler import MultiStepLR

from dataset.nusc_mv_det_dataset import NuscMVDetDataset, collate_fn
from evaluators.det_evaluators import RoadSideEvaluator
from models.bev_height import BEVHeight
from utils.torch_dist import all_gather_object, get_rank, synchronize
from utils.backup_files import backup_codebase



def main():
    ckpt_name = os.listdir(args.ckpt_path)[0]
    model_pth = os.path.join(args.ckpt_path, ckpt_name)
    model = BEVHeightLightningModel.load_from_checkpoint(model_pth)
    output = model(imgs, mats)

if __name__ == '__main__':
    main()