# Copyright (c) Gorilla-Lab. All rights reserved.
import os
from os.path import join as opj
import numpy as np
from torch.utils.data import Dataset
import h5py
import json
# from utils.provider import rotate_point_cloud_SO3, rotate_point_cloud_y
import pickle as pkl
import trimesh
import matplotlib

import os
import os.path as osp
import torch
import numpy as np
import trimesh
import random
import argparse
import copy
from scipy.spatial.transform import Rotation
from collections import defaultdict
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from ndf_robot.utils import path_util
import ndf_robot.model.vnn_occupancy_net_pointnet_dgcnn as vnn_occupancy_network
from ndf_robot.eval.ndf_alignment import NDFAlignmentCheck
import ndf_robot.utils.transformations as tra
from ndf_robot.utils import torch_util, trimesh_util
from sklearn.metrics import classification_report

os.environ["NDF_SOURCE_DIR"] = ".."
os.environ["PB_PLANNING_SOURCE_DIR"] = "../../pybullet-planning"


def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc, centroid, m


def pc_scale_max(pc):
    dimension = np.abs(np.max(pc, axis=0) - np.min(pc, axis=0))
    # print("dimension:", dimension)
    long_side = np.max(dimension)
    # Scales all dimensions equally.
    pc = pc / long_side
    return pc


def semi_points_transform(points):
    spatialExtent = np.max(points, axis=0) - np.min(points, axis=0)
    eps = 2e-3*spatialExtent[np.newaxis, :]
    jitter = eps*np.random.randn(points.shape[0], points.shape[1])
    points_ = points + jitter
    return points_


class AffordNetDataset(Dataset):
    def __init__(self, data_dir, split, partial=False, rotate='None', semi=False,
                 keep_object_classes=None, ndf_scale=False, single_affordance=None, regression=True):
        super().__init__()
        self.data_dir = data_dir
        self.split = split

        self.partial = partial
        self.rotate = rotate
        self.semi = semi
        self.ndf_scale = ndf_scale
        self.regression = regression

        self.load_data()

        # {'Chair', 'Hat', 'Bag', 'Laptop', 'Bottle', 'Dishwasher', 'Microwave', 'Clock', 'Door', 'StorageFurniture', 'Earphone', 'Bowl', 'Display', 'Scissors', 'Refrigerator', 'TrashCan', 'Mug', 'Table', 'Knife', 'Keyboard', 'Bed', 'Faucet', 'Vase'}
        self.object_classes = keep_object_classes
        if self.object_classes:
            all_data = []
            for d in self.all_data:
                if d["semantic class"] in self.object_classes:
                    all_data.append(d)
            self.all_data = all_data
        else:
            self.object_classes = list(sorted(set([d["semantic class"] for d in self.all_data])))
        print("{} data points".format(len(self.all_data)))

        self.affordance = self.all_data[0]["affordance"]
        print("{} affordances: {}".format(len(self.affordance), self.affordance))

        self.single_affordance = single_affordance
        if self.single_affordance:
            self.single_affordance_idx = self.affordance.index(self.single_affordance)

        return

    def load_data(self):
        self.all_data = []
        if self.semi:
            with open(opj(self.data_dir, 'semi_label_1.pkl'), 'rb') as f:
                temp_data = pkl.load(f)
        else:
            if self.partial:
                with open(opj(self.data_dir, 'partial_%s_data.pkl' % self.split), 'rb') as f:
                    temp_data = pkl.load(f)
            elif self.rotate != "None" and self.split != 'train':
                with open(opj(self.data_dir, 'rotate_%s_data.pkl' % self.split), 'rb') as f:
                    temp_data_rotate = pkl.load(f)
                with open(opj(self.data_dir, 'full_shape_%s_data.pkl' % self.split), 'rb') as f:
                    temp_data = pkl.load(f)
            else:
                with open(opj(self.data_dir, 'full_shape_%s_data.pkl' % self.split), 'rb') as f:
                    temp_data = pkl.load(f)
        for index, info in enumerate(temp_data):
            if self.partial:
                partial_info = info["partial"]
                for view, data_info in partial_info.items():
                    temp_info = {}
                    temp_info["shape_id"] = info["shape_id"]
                    temp_info["semantic class"] = info["semantic class"]
                    temp_info["affordance"] = info["affordance"]
                    temp_info["view_id"] = view
                    temp_info["data_info"] = data_info
                    self.all_data.append(temp_info)
            elif self.split != 'train' and self.rotate != 'None':
                rotate_info = temp_data_rotate[index]["rotate"][self.rotate]
                full_shape_info = info["full_shape"]
                for r, r_data in rotate_info.items():
                    temp_info = {}
                    temp_info["shape_id"] = info["shape_id"]
                    temp_info["semantic class"] = info["semantic class"]
                    temp_info["affordance"] = info["affordance"]
                    temp_info["data_info"] = full_shape_info
                    temp_info["rotate_matrix"] = r_data.astype(np.float32)
                    self.all_data.append(temp_info)
            else:
                temp_info = {}
                temp_info["shape_id"] = info["shape_id"]
                temp_info["semantic class"] = info["semantic class"]
                temp_info["affordance"] = info["affordance"]
                temp_info["data_info"] = info["full_shape"]
                self.all_data.append(temp_info)


    def __getitem__(self, index):

        data_dict = self.all_data[index]
        modelid = data_dict["shape_id"]
        modelcat = data_dict["semantic class"]

        data_info = data_dict["data_info"]
        model_data = data_info["coordinate"].astype(np.float32)
        labels = data_info["label"]
        for aff in self.affordance:
            temp = labels[aff].astype(np.float32).reshape(-1, 1)
            model_data = np.concatenate((model_data, temp), axis=1)

        datas = model_data[:, :3]
        targets = model_data[:, 3:]

        if self.rotate != 'None':
            if self.split == 'train':
                if self.rotate == 'so3':
                    datas = rotate_point_cloud_SO3(
                        datas[np.newaxis, :, :]).squeeze()
                elif self.rotate == 'z':
                    datas = rotate_point_cloud_y(
                        datas[np.newaxis, :, :]).squeeze()
            else:
                r_matrix = data_dict["rotate_matrix"]
                datas = (np.matmul(r_matrix, datas.T)).T

        # important: use ndf scale
        if self.ndf_scale:
            datas = pc_scale_max(datas)
            datas = datas * 0.3

            rand_idx = np.random.permutation(len(datas))[:1500]
            datas = datas[rand_idx]
            targets = targets[rand_idx]

        else:
            datas, _, _ = pc_normalize(datas)

        datas = trimesh.transform_points(datas, matrix=tra.random_rotation_matrix()).astype(datas.dtype)

        if self.single_affordance:
            # num_pts, 1
            if self.regression:
                label = targets[:, self.single_affordance_idx][:, None]
            else:
                label = targets[:, self.single_affordance_idx][:, None] > 0.3
                label = label.astype(np.float)
        else:
            # num_pts, num_aff
            label = targets

        datum = {"coords": datas, "point_cloud": datas, "shapenet_id": modelid, "object_class": modelcat}
        return datum, label
        # datum = {"coords": ref_query_pts, "point_cloud": ref_shape_pcd}

    def __len__(self):
        return len(self.all_data)


# class AffordNetDataset_Unlabel(Dataset):
#     def __init__(self, data_dir):
#         super().__init__()
#         self.data_dir = data_dir
#         self.load_data()
#         self.affordance = self.all_data[0]["affordance"]
#         return
#
#     def load_data(self):
#         self.all_data = []
#         with open(opj(self.data_dir, 'semi_unlabel_1.pkl'), 'rb') as f:
#             temp_data = pkl.load(f)
#         for info in temp_data:
#             temp_info = {}
#             temp_info["shape_id"] = info["shape_id"]
#             temp_info["semantic class"] = info["semantic class"]
#             temp_info["affordance"] = info["affordance"]
#             temp_info["data_info"] = info["full_shape"]
#             self.all_data.append(temp_info)
#
#     def __getitem__(self, index):
#         data_dict = self.all_data[index]
#         modelid = data_dict["shape_id"]
#         modelcat = data_dict["semantic class"]
#
#         data_info = data_dict["data_info"]
#         datas = data_info["coordinate"].astype(np.float32)
#
#         datas, _, _ = pc_normalize(datas)
#
#         return datas, datas, modelid, modelcat
#
#     def __len__(self):
#         return len(self.all_data)


def visualize_part_segmentation(pc, seg_labels, all_labels=None, threshold=0.3, return_vis_pc=False):

    vis_pc = trimesh.PointCloud(pc[:, :3], colors=[0,0,0,255])

    colors = get_rgb_colors()

    if seg_labels.ndim == 1:
        unique_labels = np.unique(seg_labels)
        for li, l in enumerate(unique_labels):
            color = colors[li][1]
            color = [int(c) * 255 for c in color] + [255]
            vis_pc.colors[seg_labels == l] = color
        vis_pc.show()
    elif seg_labels.ndim == 2:
        vis_pcs = []
        num_aff = seg_labels.shape[1]
        for i in range(num_aff):
            vis_pc.colors[:] = [0, 0, 0, 255]
            if np.any(seg_labels[:, i] > threshold):
                aff = all_labels[i]
                color = colors[i][1]
                color = [int(c) * 255 for c in color] + [255]
                aff_idxs = seg_labels[:, i] > threshold
                aff_colors = np.zeros([sum(aff_idxs), 4])
                aff_colors[:] = color
                aff_colors[:, 3] = seg_labels[aff_idxs][:, i] * 255
                # print(aff_colors)
                vis_pc.colors[aff_idxs] = aff_colors
                # print(vis_pc.colors[seg_labels[:, i] > threshold][:, 3])
                print("affordance:", aff)
                if return_vis_pc:
                    vis_pcs.append(copy.deepcopy(vis_pc))
                else:
                    print("affordance:", aff)
                    vis_pc.show()

    if return_vis_pc:
        return vis_pcs


def get_rgb_colors():
    rgb_colors = []
    # each color is a tuple of (name, (r,g,b))
    for name, hex in matplotlib.colors.cnames.items():
        rgb_colors.append((name, matplotlib.colors.to_rgb(hex)))

    rgb_colors = sorted(rgb_colors, key=lambda x: x[0])

    priority_colors = [('red', (1.0, 0.0, 0.0)), ('green', (0.0, 1.0, 0.0)), ('blue', (0.0, 0.0, 1.0)),
                       ('orange', (1.0, 0.6470588235294118, 0.0)),
                       ('purple', (0.5019607843137255, 0.0, 0.5019607843137255)), ('magenta', (1.0, 0.0, 1.0)), ]
    rgb_colors = priority_colors + rgb_colors

    return rgb_colors


####################################################################################################
# NDFClassifier class
####################################################################################################
class NDFPointClassifier(pl.LightningModule):
    def __init__(self, model_path, regression=True):
        super().__init__()

        self.ndf_model = vnn_occupancy_network.VNNOccNet(
            latent_dim=256, model_type="pointnet", return_features=True, sigmoid=True
        )
        self.ndf_model.load_state_dict(torch.load(model_path))
        for param in self.ndf_model.parameters():
            param.requires_grad = False

        self.layers = nn.Sequential(
            nn.Linear(2049, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

        self.regression = regression
        if self.regression:
            self.loss = torch.nn.MSELoss(reduction="mean")
        else:
            self.loss = torch.nn.BCEWithLogitsLoss(reduction="mean", pos_weight=torch.tensor(15))

    def forward(self, x):
        self.ndf_model.eval()
        with torch.no_grad():
            reference_latent = self.ndf_model.extract_latent(x)
            reference_act_hat = self.ndf_model.forward_latent(
                reference_latent, x["coords"]
            )
            # # ToDo: try other pooling methods for creating pose feature
            # reference_act_hat = torch.mean(reference_act_hat, dim=1)

        y_hat = self.layers(reference_act_hat)
        return y_hat

    def training_step(self, batch, batch_idx):
        x, y = batch

        # self.ndf_model.eval()
        # with torch.no_grad():
        #     reference_latent = self.ndf_model.extract_latent(x)
        #     reference_act_hat = self.ndf_model.forward_latent(
        #         reference_latent, x["coords"]
        #     )
        #     # reference_act_hat = torch.mean(reference_act_hat, dim=1)
        # y_hat = self.layers(reference_act_hat)
        y_hat = self.forward(x)

        loss = self.loss(y_hat, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch

        # self.ndf_model.eval()
        # with torch.no_grad():
        #     reference_latent = self.ndf_model.extract_latent(x)
        #     reference_act_hat = self.ndf_model.forward_latent(
        #         reference_latent, x["coords"]
        #     )
        #     # reference_act_hat = torch.mean(reference_act_hat, dim=1)
        # y_hat = self.layers(reference_act_hat)
        y_hat = self.forward(x)

        loss = self.loss(y_hat, y)
        self.log("val_loss", loss, prog_bar=True)
        print(f"\nprediction: {torch.sigmoid(y_hat).item()}, gt: {y.item()}")
        # input("next?")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer


####################################################################################################
# Functions for training an MLP and splitting the dataset into training and validation splits
####################################################################################################
def train_single_affordance_mlp(mode, random_seed=42, checkpoint_path=None, max_epochs=5, single_affordance="pourable", regression=True):
    device = (
        torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    )

    if mode == "train":

        pl.seed_everything(random_seed)
        train_dataset = AffordNetDataset("/home/weiyu/data_drive/3daffordancenet", 'train', partial=False, rotate='None', semi=False,
                                         keep_object_classes=["Bowl", "Bottle", "Mug"], ndf_scale=True, single_affordance=single_affordance, regression=regression)

        model_path = osp.join(path_util.get_ndf_model_weights(), "multi_category_weights.pth")
        mlp = NDFPointClassifier(model_path=model_path, regression=regression)

        trainer = pl.Trainer(gpus=1, deterministic=True, max_epochs=max_epochs)
        trainer.fit(mlp, train_dataloaders=DataLoader(train_dataset, batch_size=8))
        # trainer.validate(dataloaders=DataLoader(val_dataset, batch_size=1))

    elif mode == "valid":

        assert os.path.exists(checkpoint_path), "checkpoint path {} not valid".format(checkpoint_path)

        pl.seed_everything(random_seed)
        val_dataset = AffordNetDataset("/home/weiyu/data_drive/3daffordancenet", 'val', partial=False,
                                         rotate='None', semi=False,
                                         keep_object_classes=["Bowl", "Bottle", "Mug"], ndf_scale=True,
                                         single_affordance=single_affordance, regression=regression)

        model_path = osp.join(path_util.get_ndf_model_weights(), "multi_category_weights.pth")
        model = NDFPointClassifier.load_from_checkpoint(checkpoint_path, model_path=model_path, regression=regression)

        # trainer = pl.Trainer(gpus=1, deterministic=True, max_epochs=max_epochs)
        # trainer.validate(model=model, dataloaders=DataLoader(val_dataset, batch_size=1))
        model.to(device)
        model.eval()
        for x, y in val_dataset:
            tensor_x = {}
            for k in ["point_cloud", "coords"]:
                tensor_x[k] = torch.from_numpy(x[k]).float().to(device).unsqueeze(0)
            with torch.no_grad():
                y_hat = model(tensor_x)
            y_hat = torch.sigmoid(y_hat)[0].cpu().numpy()
            # y_hat, y: num_pts, 1
            print("-"*30)
            print(f"gt: {y[:, 0]}")
            print(f"prediction: {y_hat[:, 0]}")
            # visualize_part_segmentation(x["point_cloud"], seg_labels=y, all_labels=[single_affordance], threshold=0.3)
            # visualize_part_segmentation(x["point_cloud"], seg_labels=y_hat, all_labels=[single_affordance], threshold=0.55)

            vis_pcs = visualize_part_segmentation(x["point_cloud"], seg_labels=y, all_labels=[single_affordance], threshold=0.3, return_vis_pc=True)
            vis_pcs2 = visualize_part_segmentation(x["point_cloud"], seg_labels=y_hat, all_labels=[single_affordance], threshold=0.55, return_vis_pc=True)
            for vis_pc in vis_pcs2:
                vis_pc.apply_translation([0.5, 0, 0])

            if len(vis_pcs + vis_pcs2):
                vis_scene = trimesh.Scene()
                vis_scene.add_geometry(vis_pcs + vis_pcs2)
                vis_scene.show()

        # labels = []
        # predictions = []
        # for x, y, _ in tqdm(val_dataset):
        #     for k in x:
        #         x[k] = x[k].to(device).unsqueeze(0)
        #     with torch.no_grad():
        #         y_hat = model(x)
        #     y_hat = y_hat[0]
        #     labels.append(y.cpu().numpy()[0])
        #     predictions.append(y_hat.cpu().numpy()[0] > 0.5)
        # print(labels)
        # print(predictions)
        # print(classification_report(labels, predictions))


if __name__ == "__main__":

    # train_single_affordance_mlp("train", random_seed=42, checkpoint_path=None, max_epochs=100, single_affordance="pourable")
    # train_single_affordance_mlp("valid", random_seed=42, checkpoint_path="/home/weiyu/Research/ndf_robot/src/ndf_robot/eval/lightning_logs/version_6/checkpoints/epoch=99-step=7000.ckpt", single_affordance="pourable")

    np.random.seed(0)

    # # load model
    # device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    # model_path = osp.join(path_util.get_ndf_model_weights(), 'multi_category_weights.pth')
    # model = vnn_occupancy_network.VNNOccNet(latent_dim=256, model_type='pointnet', return_features=True,
    #                                         sigmoid=True).to(device)
    # model.load_state_dict(torch.load(model_path))


    # dataset
    dataset = AffordNetDataset("/home/weiyu/data_drive/3daffordancenet", 'train', partial=False, rotate='None', semi=False,
                               keep_object_classes=["Bowl", "Bottle", "Mug"], ndf_scale=True, single_affordance="pourable")

    # for pc, _, aff_label, shapenet_id, obj_cls in dataset:
    ratios = []
    for i in np.random.permutation(len(dataset)):
        datum, aff_label = dataset[i]

        pc = datum["point_cloud"]
        shapenet_id = datum["shapenet_id"]
        obj_cls = datum["object_class"]

        print(pc.shape)
        print(aff_label.shape)
        print(shapenet_id)
        print(obj_cls)
        print(np.min(pc, axis=0), np.max(pc, axis=0))

        print("ratio", np.sum(aff_label == 0)/np.sum(aff_label == 1))
        # visualize_part_segmentation(pc, seg_labels=aff_label, all_labels=dataset.affordance)
        vis_pcs = visualize_part_segmentation(pc, seg_labels=aff_label, all_labels=["pourable"], return_vis_pc=True)
        vis_pcs2 = visualize_part_segmentation(pc, seg_labels=aff_label, all_labels=["pourable"], return_vis_pc=True)

        for vis_pc in vis_pcs2:
            vis_pc.apply_translation([0.5, 0, 0])

        if len(vis_pc + vis_pcs2):
            vis_scene = trimesh.Scene()
            vis_scene.add_geometry(vis_pcs + vis_pcs2)
            vis_scene.show()

        if np.any(aff_label==1):
            ratios.append(np.sum(aff_label == 0)/np.sum(aff_label == 1))

    print(ratios)
    print(len(ratios))
    print(np.mean(ratios))

    # # reference_model_input = {}
    # # ref_query_pts = torch.from_numpy(pc).float().to(device)
    # # ref_shape_pcd = torch.from_numpy(pc).float().to(device)
    # datum["coords"] = torch.from_numpy(datum["coords"]).float().to(device)[None, :, :]
    # datum["point_cloud"] = torch.from_numpy(datum["point_cloud"]).float().to(device)[None, :, :]
    #
    # model.eval()
    # with torch.no_grad():
    #     # get the descriptors for these reference query points
    #     reference_latent = model.extract_latent(datum).detach()
    #     print(reference_latent.shape)
    #     reference_act_hat = model.forward_latent(
    #         reference_latent, datum["coords"]
    #     ).detach()
    #     print(reference_act_hat.shape)