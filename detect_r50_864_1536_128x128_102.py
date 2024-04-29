# Copyright (c) Megvii Inc. All rights reserved.
from argparse import ArgumentParser, Namespace

import os
import mmcv
import pytorch_lightning as pl
import cv2
import json
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
from evaluators.result2kitti import *
from scripts.data_converter.visual_utils import *
from tqdm import tqdm
from exps.dair_v2x.bev_height_lss_r101_864_1536_256x256_140 import BEVHeightLightningModel


# Questions:
# 1. Solve the GPU out of memory problem
# 2. Check how to set batch size
# 3. Use customized dataset

data_root = "data/dair-v2x-i/"
gt_label_path = "data/dair-v2x-i-kitti/training/label_2"


def main(args: Namespace) -> None:
    torch.cuda.empty_cache()

    ckpt_name = os.listdir(ckpt_path)[0]
    model_pth = os.path.join(ckpt_path, ckpt_name)
    BEVHeight = BEVHeightLightningModel(**vars(args)).load_from_checkpoint(model_pth)

    # img_metas: Point cloud and image's meta info
    dataloader = BEVHeight.val_dataloader()      # !!!
    (sweep_imgs, mats, _, img_metas, _, _) = next(iter(dataloader))  # !!!
    print(sweep_imgs.shape)

    # Customize for CARLA images
    custom_img_name = 'carla01'
    # The parameters of 'mats' are used in LSSFPN _forward_single_sweep
    # Transformation matrix from camera to ego
    mats['sensor2ego_mats'][1] = torch.tensor([[[[-0.0414, -0.2259,  0.9736, -0.1962],
                                                 [-1.0011, -0.0151, -0.0364, -2.0562],
                                                 [ 0.0072, -1.0842, -0.2493,  6.4794],
                                                 [ 0.0000,  0.0000,  0.0000,  1.0000]]]])
    # Intrinsic matrix
    mats['intrin_mats'][1] = torch.tensor([[[[2.1834e+03, 0.0000e+00, 9.4059e+02, 0.0000e+00],
                                             [0.0000e+00, 2.3293e+03, 5.6757e+02, 0.0000e+00],
                                             [0.0000e+00, 0.0000e+00, 1.0000e+00, 0.0000e+00],
                                             [0.0000e+00, 0.0000e+00, 0.0000e+00, 1.0000e+00]]]])
    mats['reference_heights'][1] = torch.tensor([[5.8241]])
    # Transformation matrix from key frame camera to sweep frame camera
    mats['sensor2sensor_mats'][1] = torch.tensor([[[[ 1.0000e+00, -9.3132e-10, -3.7253e-09,  2.0862e-07],
                                                    [-6.9849e-10,  1.0000e+00,  1.6851e-08, -2.1545e-15],
                                                    [-2.9680e-18,  4.2492e-09,  1.0000e+00, -1.1921e-07],
                                                    [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]]]])
    # Rotation matrix for bda
    mats['bda_mat'][1] = torch.tensor([[1., 0., 0., 0.],
                                       [0., 1., 0., 0.],
                                       [0., 0., 1., 0.],
                                       [0., 0., 0., 1.]])
    # Transformation matrix for ida
    mats['ida_mats'][1] = torch.tensor([[[[0.8000, 0.0000, 0.0000, 0.0000],
                                          [0.0000, 0.8000, 0.0000, 0.0000],
                                          [0.0000, 0.0000, 1.0000, 0.0000],
                                          [0.0000, 0.0000, 0.0000, 1.0000]]]])
    mats['sensor2virtual_mats'][1] = torch.tensor([[[[ 9.9998e-01,  6.4451e-03,  7.3137e-04,  0.0000e+00],
                                                     [-6.4451e-03,  9.7455e-01,  2.2407e-01,  0.0000e+00],
                                                     [ 7.3137e-04, -2.2407e-01,  9.7457e-01,  0.0000e+00],
                                                     [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]]]])


    # img_metas[1]['token'] = f'image/{custom_img_name}.png'

    # Input:
    # sweep_imgs
    # mats
    # img_metas
    # virtuallidar_to_camera (Why LiDAR?)
    with torch.no_grad():
        for key, value in mats.items():
            mats[key] = value.cuda()
        sweep_imgs = sweep_imgs.cuda()
        BEVHeight.model.cuda()
        preds = BEVHeight(sweep_imgs, mats)

        results = BEVHeight.model.get_bboxes(preds, img_metas)
        for i in range(len(results)):
            results[i][0] = results[i][0].tensor.detach().cpu().numpy()
            results[i][1] = results[i][1].detach().cpu().numpy()
            results[i][2] = results[i][2].detach().cpu().numpy()
            results[i].append(img_metas[i])
        print(results[0])

        all_pred_results = list()
        all_img_metas = list()
        for validation_step_output in results:
            all_pred_results.append(validation_step_output[:3])
            all_img_metas.append(validation_step_output[3])
        synchronize()
        len_dataset = len(BEVHeight.val_dataloader().dataset)
        all_pred_results = sum(
            map(list, zip(*all_gather_object(all_pred_results))),     # !!!
            [])[:len_dataset]
        all_img_metas = sum(map(list, zip(*all_gather_object(all_img_metas))),
                            [])[:len_dataset]
        print(all_img_metas)

        result_files, tmp_dir = BEVHeight.evaluator.format_results(all_pred_results, all_img_metas,
                                                                   result_names=['img_bbox'],
                                                                   jsonfile_prefix=None)     # !!!
        print(result_files, tmp_dir)
        results_path = "outputs"

        results_file = result_files["img_bbox"]
        dair_root = data_root
        demo = True

        category_map_dair = {"car": "Car", "van": "Car", "truck": "Car", "bus": "Car", "pedestrian": "Pedestrian",
                             "bicycle": "Cyclist", "trailer": "Cyclist", "motorcycle": "Cyclist"}

        with open(results_file, 'r', encoding='utf8') as fp:
            results = json.load(fp)["results"]
        for sample_token in tqdm(results.keys()):
            sample_id = int(sample_token.split("/")[1].split(".")[0])
            virtuallidar_to_camera_file = os.path.join(dair_root, "calib/virtuallidar_to_camera",    # Input!
                                                       "{:06d}".format(sample_id) + ".json")
            Tr_velo_to_cam, r_velo2cam, t_velo2cam = get_lidar2cam(virtuallidar_to_camera_file)
            preds = results[sample_token]
            pred_lines = []
            bboxes = []
            for pred in preds:
                loc = pred["translation"]
                dim = pred["size"]
                yaw_lidar = pred["box_yaw"]
                detection_score = pred["detection_score"]
                class_name = pred["detection_name"]

                w, l, h = dim[0], dim[1], dim[2]
                x, y, z = loc[0], loc[1], loc[2]
                bottom_center = [x, y, z]
                obj_size = [l, w, h]
                bottom_center_in_cam = r_velo2cam * np.matrix(bottom_center).T + t_velo2cam
                _, yaw = get_camera_3d_8points(
                    obj_size, yaw_lidar, bottom_center, bottom_center_in_cam, r_velo2cam, t_velo2cam
                )
                yaw = 0.5 * np.pi - yaw_lidar

                cam_x, cam_y, cam_z = convert_point(np.array([x, y, z, 1]).T, Tr_velo_to_cam)
                box = get_lidar_3d_8points([w, l, h], yaw_lidar, [x, y, z + h / 2])
                if detection_score > 0.45 and class_name in category_map_dair.keys():
                    i1 = category_map_dair[class_name]
                    i9, i11, i10 = str(round(h, 4)), str(round(w, 4)), str(round(l, 4))
                    i12, i13, i14 = str(round(cam_x, 4)), str(round(cam_y, 4)), str(round(cam_z, 4))
                    i15 = str(round(yaw, 4))
                    line = [i1, '0', '0', '0', '0', '0', '0', '0', i9, i10, i11, i12, i13, i14, i15,
                            str(round(detection_score, 4))]
                    pred_lines.append(line)
                    bboxes.append(box)
            os.makedirs(os.path.join(results_path, "data"), exist_ok=True)
            write_kitti_in_txt(pred_lines, os.path.join(results_path, "data", "{:06d}".format(sample_id) + ".txt"))
            if demo:
                os.makedirs(os.path.join(results_path, "demo"), exist_ok=True)
                label_path = os.path.join(gt_label_path, "{:06d}".format(sample_id) + ".txt")
                demo_file = os.path.join(results_path, "demo", "{:06d}".format(sample_id) + ".jpg")
                pcd_vis(bboxes, demo_file, label_path, Tr_velo_to_cam)

        data_kitti_root = 'data/dair-v2x-i-kitti'
        filename = '000021'
        image_path = os.path.join(data_kitti_root, "training/image_2", f"{filename}.jpg")
        calib_path = os.path.join(data_kitti_root, "training/calib", f"{filename}.txt")
        label_path = os.path.join(results_path, "data", f"{filename}.txt")

        image = cv2.imread(image_path)
        _, P2, denorm = load_calib(calib_path)

        for line in pred_lines:
            object_type = line[0]
            if object_type not in color_map.keys(): continue
            dim = np.array(line[8:11]).astype(float)
            location = np.array(line[11:14]).astype(float)
            rotation_y = float(line[14])
            box_3d = compute_box_3d_camera(dim, location, rotation_y, denorm)
            box_2d = project_to_image(box_3d, P2)
            image = draw_box_3d(image, box_2d, c=color_map[object_type])

        detection_path = os.path.join(results_path, "detections")
        if not os.path.exists(detection_path):
            os.mkdir(detection_path)
        # cv2.imwrite(os.path.join(detection_path, f"{filename}_detection.jpg"), image)
        cv2.imshow('Detection Results', image)
        k = cv2.waitKey(0)
        if k == 27:
            cv2.destroyAllWindows()


def run_cli():
    parent_parser = ArgumentParser(add_help=False)
    # parent_parser = pl.Trainer.add_argparse_args(parent_parser)
    parent_parser.add_argument('-b', '--batch_size_per_device',
                               default=1,
                               type=int)
    parent_parser.add_argument('--seed',
                               type=int,
                               default=0,
                               help='seed for initializing training.')
    parent_parser.add_argument('--ckpt_path',
                               default='./ckpt',
                               type=str)
    parser = BEVHeightLightningModel.add_model_specific_args(parent_parser)
    parser.set_defaults(
        profiler='simple',
        deterministic=False,
        max_epochs=30,
        accelerator='ddp',
        num_sanity_val_steps=0,
        gradient_clip_val=5,
        limit_val_batches=0,
        enable_checkpointing=True,
        precision=32,
        default_root_dir='./outputs/bev_height_lss_r101_864_1536_256x256')
    args = parser.parse_args()
    main(args)

ckpt_path = './ckpt'
bz = 1
if __name__ == '__main__':
    run_cli()
