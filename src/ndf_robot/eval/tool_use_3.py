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

from ndf_robot.utils import path_util
import ndf_robot.model.vnn_occupancy_net_pointnet_dgcnn as vnn_occupancy_network
from ndf_robot.eval.ndf_alignment import NDFAlignmentCheck
import ndf_robot.utils.transformations as tra
from ndf_robot.utils import torch_util, trimesh_util

from sklearn import svm
from sklearn.metrics import classification_report


import os

os.environ["NDF_SOURCE_DIR"] = ".."
os.environ["PB_PLANNING_SOURCE_DIR"] = "../../pybullet-planning"

######################################################
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl


class NDFClassifier(pl.LightningModule):
    def __init__(self, model_path):
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
        self.loss = torch.nn.BCEWithLogitsLoss(reduction="mean")

    def forward(self, x):
        self.ndf_model.eval()
        with torch.no_grad():
            reference_latent = self.ndf_model.extract_latent(x)
            reference_act_hat = self.ndf_model.forward_latent(
                reference_latent, x["coords"]
            )
            reference_act_hat = torch.mean(reference_act_hat, dim=1)

        y_hat = self.layers(reference_act_hat)
        return y_hat

    def training_step(self, batch, batch_idx):
        x, y = batch

        self.ndf_model.eval()
        with torch.no_grad():
            reference_latent = self.ndf_model.extract_latent(x)
            reference_act_hat = self.ndf_model.forward_latent(
                reference_latent, x["coords"]
            )
            reference_act_hat = torch.mean(reference_act_hat, dim=1)

        y_hat = self.layers(reference_act_hat)
        loss = self.loss(y_hat, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch

        self.ndf_model.eval()
        with torch.no_grad():
            reference_latent = self.ndf_model.extract_latent(x)
            reference_act_hat = self.ndf_model.forward_latent(
                reference_latent, x["coords"]
            )
            reference_act_hat = torch.mean(reference_act_hat, dim=1)

        y_hat = self.layers(reference_act_hat)
        loss = self.loss(y_hat, y)
        self.log("val_loss", loss, prog_bar=True)
        print(f"\nprediction: {torch.sigmoid(y_hat).item()}, gt: {y.item()}")
        # input("next?")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer


def train_mlp(mode, random_seed=42, checkpoint_path=None, max_epochs=5):

    device = (
        torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    )

    if mode == "train":

        pl.seed_everything(random_seed)
        train_obj_models, val_obj_models = split_objects(train_ratio=0.8)
        print(train_obj_models)
        print(val_obj_models)
        train_dataset = SemanticPoseDataset(train_obj_models, debug=False, semantic_type="top")

        model_path = osp.join(path_util.get_ndf_model_weights(), "multi_category_weights.pth")
        mlp = NDFClassifier(model_path=model_path)

        trainer = pl.Trainer(gpus=1, deterministic=True, max_epochs=max_epochs)
        trainer.fit(mlp, train_dataloaders=DataLoader(train_dataset, batch_size=8))
        # trainer.validate(dataloaders=DataLoader(val_dataset, batch_size=1))

    elif mode == "valid":

        assert os.path.exists(checkpoint_path), "checkpoint path {} not valid".format(checkpoint_path)

        pl.seed_everything(random_seed)
        train_obj_models, val_obj_models = split_objects(train_ratio=0.8)
        print(train_obj_models)
        print(val_obj_models)
        val_dataset = SemanticPoseDataset(val_obj_models, debug=False, semantic_type="top", valid_mode=True)

        model_path = osp.join(path_util.get_ndf_model_weights(), "multi_category_weights.pth")
        model = NDFClassifier.load_from_checkpoint(checkpoint_path, model_path=model_path)

        # trainer = pl.Trainer(gpus=1, deterministic=True, max_epochs=max_epochs)
        # trainer.validate(model=model, dataloaders=DataLoader(val_dataset, batch_size=1))
        model.to(device)
        model.eval()
        for x, y, scene in val_dataset:
            for k in x:
                x[k] = x[k].to(device).unsqueeze(0)
            with torch.no_grad():
                y_hat = model(x)
            print(f"\nprediction: {torch.sigmoid(y_hat).item()}, gt: {y.item()}")
            scene.show()


######################################################


def split_objects(
    object_base_dir=path_util.get_ndf_obj_descriptions(), train_ratio=0.7
):
    all_obj_models = []
    for obj_class in os.listdir(object_base_dir):
        if "centered_obj_normalized" in obj_class:
            for obj_model in os.listdir(os.path.join(object_base_dir, obj_class)):
                all_obj_models.append(obj_model)
    # print(all_obj_models)
    rix = np.random.permutation(len(all_obj_models))
    train_num = int(len(all_obj_models) * train_ratio)
    train_obj_models = [all_obj_models[i] for i in rix[:train_num]]
    val_obj_models = [all_obj_models[i] for i in rix[train_num:]]
    return train_obj_models, val_obj_models


class SemanticPoseDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        split_obj_models=None,
        object_base_dir=path_util.get_ndf_obj_descriptions(),
        debug=False,
        semantic_type="bottom",
        num_pos_poses=3,
        num_neg_poses=3,
        random_noise_sigma=0.001,
        bottom_scale_ratio=0.01,
        top_scale_ratio=0.05,
        scale=0.25,
        feat_sigma=0.025,
        feat_n_opt_pts=500,
        feat_n_pts=1500,
        random_opening_pose_scaling=0.9,
        valid_mode=False
    ):

        # params
        self.random_noise_sigma = random_noise_sigma
        self.bottom_scale_ratio = bottom_scale_ratio
        self.top_scale_ratio = top_scale_ratio
        self.scale = scale
        assert semantic_type in ["bottom", "top"]
        self.semantic_type = semantic_type
        self.random_opening_pose_scaling = random_opening_pose_scaling

        # params for ndf features
        self.feat_sigma = feat_sigma
        self.feat_n_opt_pts = feat_n_opt_pts
        self.feat_n_pts = feat_n_pts

        self.debug = debug
        self.num_pos_poses = num_pos_poses
        self.num_neg_poses = num_neg_poses
        self.valid_mode = valid_mode

        self.obj_paths = []
        self.obj_path_to_data = {}

        for obj_class in os.listdir(object_base_dir):
            # if obj_class != "bottle_centered_obj_normalized":
            if "centered_obj_normalized" in obj_class:
                for obj_model in os.listdir(os.path.join(object_base_dir, obj_class)):
                    if split_obj_models is None or obj_model in split_obj_models:
                        obj_path = os.path.join(
                            object_base_dir,
                            obj_class,
                            obj_model,
                            "models/model_normalized.obj",
                        )
                        self.obj_paths.append(obj_path)
                        self.obj_path_to_data[obj_path] = {"class": obj_class}

        # create data index
        self.data = []
        for obj_path in self.obj_paths:
            self.data.extend([(obj_path, True)] * num_pos_poses)
            self.data.extend([(obj_path, False)] * num_neg_poses)

        # extract object pcd and semantic poses
        self.preprocess()

    def __len__(self):
        return len(self.data)

    def preprocess(self):
        print("sample object point cloud from meshes and semantic mask")
        for obj_path in tqdm(self.obj_paths):
            if self.semantic_type == "bottom":
                pcd, semantic_mask = self.get_obj_pcd_and_bottom_mask(obj_path)
            elif self.semantic_type == "top":
                pcd, semantic_mask = self.get_obj_pcd_and_top_mask(obj_path)

            self.obj_path_to_data[obj_path]["pcd"] = pcd
            self.obj_path_to_data[obj_path]["semantic_mask"] = semantic_mask

    def get_obj_pcd_and_bottom_mask(self, obj_model_path):

        mesh = trimesh.load(obj_model_path, process=False)
        mesh.apply_scale(self.scale)
        # convert to pc
        pcd = mesh.sample(5000)

        bounds = mesh.bounding_box.bounds
        y_min, y_max = bounds[:, 1]
        x_range, y_range, z_range = bounds[1, :] - bounds[0, :]

        bottom_mask = pcd[:, 1] < y_min + y_range * self.bottom_scale_ratio

        return pcd, bottom_mask

    def get_obj_pcd_and_top_mask(self, obj_model_path):

        mesh = trimesh.load(obj_model_path, process=False)
        mesh.apply_scale(self.scale)
        # convert to pc
        pcd = mesh.sample(5000)

        bounds = mesh.bounding_box.bounds
        y_min, y_max = bounds[:, 1]
        x_range, y_range, z_range = bounds[1, :] - bounds[0, :]

        top_mask = pcd[:, 1] > y_max - y_range * self.top_scale_ratio

        return pcd, top_mask

    def get_raw_data(self, idx):
        """
        retrieve one data point
        :param idx:
        :return:
        """

        obj_path, label = self.data[idx]

        pcd, semantic_mask, obj_class = (
            self.obj_path_to_data[obj_path]["pcd"],
            self.obj_path_to_data[obj_path]["semantic_mask"],
            self.obj_path_to_data[obj_path]["class"]
        )
        # print(obj_class)
        pcd_semantic = pcd[semantic_mask]

        if label:
            pose = tra.euler_matrix(np.pi, np.random.uniform(high=np.pi * 2), 0)
            pose[:3, 3] = pcd_semantic[np.random.choice(list(range(len(pcd_semantic)))), :]
            pose[:3, 3] += np.random.normal(scale=self.random_noise_sigma, size=3)
        elif self.semantic_type == "bottom":
            pose = self.random_pose(pcd, semantic_mask)
        elif self.semantic_type == "top":
            if np.random.uniform() > 0.5 and obj_class != "bottle_centered_obj_normalized":
                pose = self.random_opening_pose(pcd, semantic_mask)
            else:
                pose = self.random_pose(pcd, semantic_mask)

        if self.debug or self.valid_mode:
            shape_pcd = trimesh.PointCloud(pcd)
            semantic_pcd = trimesh.PointCloud(pcd_semantic)
            semantic_pcd.colors = np.tile((255, 0, 0), (semantic_pcd.vertices.shape[0], 1))
            local_frame_positive = trimesh.creation.axis(
                origin_size=0.002,
                transform=pose,
                origin_color=None,
                axis_radius=0.002,
                axis_length=0.05,
            )
            scene = trimesh.Scene()
            scene.add_geometry([shape_pcd, local_frame_positive, semantic_pcd])
            if self.debug:
                scene.show()

        # sample query points
        query_pts = np.random.normal(
            0.0, self.feat_sigma, size=(self.feat_n_opt_pts, 3)
        )

        reference_query_pts = torch_util.transform_pcd(query_pts, pose)

        reference_model_input = {}
        ref_query_pts = torch.from_numpy(
            reference_query_pts[: self.feat_n_opt_pts]
        ).float()
        ref_shape_pcd = torch.from_numpy(pcd[: self.feat_n_pts]).float()

        # put all the input data in a dictionary
        datum = {"coords": ref_query_pts, "point_cloud": ref_shape_pcd}

        if self.valid_mode:
            return datum, torch.FloatTensor([label]), scene

        return datum, torch.FloatTensor([label])

    def random_pose(self, pcd, semantic_mask):
        random_idx = np.random.randint(pcd[~semantic_mask].shape[0])
        random_pos = pcd[~semantic_mask][random_idx]
        random_pose = tra.random_rotation_matrix()
        random_pose[:3, 3] = random_pos
        pose = random_pose
        return pose
    
    def random_opening_pose(self, pcd, semantic_mask):
        scaling = self.random_opening_pose_scaling
        height = np.mean(pcd[semantic_mask], axis=0)[1]
        x_min, x_max = np.min(pcd[semantic_mask], axis=0)[0], np.max(pcd[semantic_mask], axis=0)[0]
        
        random_angle = np.random.uniform(low=-np.pi, high=np.pi)
        random_dist = np.random.uniform(low=x_min*scaling, high=x_max*scaling)
        x_perturb, z_perturb = random_dist*np.cos(random_angle), random_dist*np.sin(random_angle)
        
        random_pos = np.array([x_perturb, height ,z_perturb])
        random_pose = tra.random_rotation_matrix()
        random_pose[:3, 3] = random_pos
        return random_pose


    def __getitem__(self, idx):

        # datum = self.convert_to_tensors(self.get_raw_data(idx))
        datum = self.get_raw_data(idx)

        return datum


def build_simple_semantic_classifier():

    device = (
        torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    )
    model_path = osp.join(
        path_util.get_ndf_model_weights(), "multi_category_weights.pth"
    )
    model = vnn_occupancy_network.VNNOccNet(
        latent_dim=256, model_type="pointnet", return_features=True, sigmoid=True
    ).to(device)
    model.load_state_dict(torch.load(model_path))

    feats = []
    labels = []

    spd = SemanticPoseDataset(debug=False)
    for datum in spd:
        pcd = datum["pcd"]
        pose_feature = compute_pose_feature(
            model, datum["pcd"], datum["pose"], visualize=False, device=device
        )

        print(pose_feature.shape)

        feats.append(pose_feature.cpu().numpy())
        labels.append(int(datum["label"]))
        input("here")


#####################################################################################################


def sample_bottom_pose(
    pcd,
    mesh,
    num_poses=1,
    bottom_scale_ratio=0.01,
    random_noise_sigma=0.001,
    visualize=False,
):

    bounds = mesh.bounding_box.bounds
    y_min, y_max = bounds[:, 1]
    x_range, y_range, z_range = bounds[1, :] - bounds[0, :]

    bottom_mask = pcd[:, 1] < y_min + y_range * bottom_scale_ratio
    pcd_bottom = pcd[bottom_mask]

    poses = []
    for i in range(num_poses):
        pose = tra.euler_matrix(np.pi, np.random.uniform(high=np.pi * 2), 0)
        pose[:3, 3] = pcd_bottom[np.random.choice(list(range(len(pcd_bottom)))), :]
        pose[:3, 3] += np.random.normal(scale=random_noise_sigma, size=3)
        poses.append(pose)

        if visualize:
            shape_pcd = trimesh.PointCloud(pcd)
            bottom_pcd = trimesh.PointCloud(pcd_bottom)
            bottom_pcd.colors = np.tile((255, 0, 0), (bottom_pcd.vertices.shape[0], 1))
            local_frame_positive = trimesh.creation.axis(
                origin_size=0.002,
                transform=pose,
                origin_color=None,
                axis_radius=0.002,
                axis_length=0.05,
            )
            scene = trimesh.Scene()
            scene.add_geometry([shape_pcd, local_frame_positive, bottom_pcd])
            scene.show()

    return poses, bottom_mask


def sample_top_pose(
    pcd,
    mesh,
    num_poses=1,
    top_scale_ratio=0.05,
    random_noise_sigma=0.001,
    visualize=False,
):

    bounds = mesh.bounding_box.bounds
    y_min, y_max = bounds[:, 1]
    x_range, y_range, z_range = bounds[1, :] - bounds[0, :]

    top_mask = pcd[:, 1] > y_max - y_range * top_scale_ratio
    pcd_top = pcd[top_mask]

    poses = []
    for i in range(num_poses):
        pose = tra.euler_matrix(np.pi, np.random.uniform(high=np.pi * 2), 0)
        pose[:3, 3] = pcd_top[np.random.choice(list(range(len(pcd_top)))), :]
        pose[:3, 3] += np.random.normal(scale=random_noise_sigma, size=3)
        poses.append(pose)

        if visualize:
            shape_pcd = trimesh.PointCloud(pcd)
            top_pcd = trimesh.PointCloud(pcd_top)
            top_pcd.colors = np.tile((255, 0, 0), (top_pcd.vertices.shape[0], 1))
            local_frame_positive = trimesh.creation.axis(
                origin_size=0.002,
                transform=pose,
                origin_color=None,
                axis_radius=0.002,
                axis_length=0.05,
            )
            scene = trimesh.Scene()
            scene.add_geometry([shape_pcd, local_frame_positive, top_pcd])
            scene.show()

    return poses, top_mask


def sample_random_pose(pcd, mask, num_poses=1, visualize=False):

    # create negative example
    poses = []
    for i in range(num_poses):
        random_idx = np.random.randint(pcd[~mask].shape[0])
        random_pos = pcd[~mask][random_idx]
        random_pose = tra.random_rotation_matrix()
        random_pose[:3, 3] = random_pos
        poses.append(random_pose)

        if visualize:
            shape_pcd = trimesh.PointCloud(pcd)
            local_frame_negative = trimesh.creation.axis(
                origin_size=0.002,
                transform=random_pose,
                origin_color=None,
                axis_radius=0.002,
                axis_length=0.05,
            )
            scene = trimesh.Scene()
            scene.add_geometry([shape_pcd, local_frame_negative])
            scene.show()

    return poses


def sample_opening_pose(pcd, mask, center_radius=None, num_poses=1, scaling=0.9, visualize=False):

    height = np.mean(pcd[mask], axis=0)[1]
    x_max = np.max(pcd[mask], axis=0)[0] * scaling

    # create negative example
    poses = []
    for i in range(num_poses):

        random_angle = np.random.uniform(low=-np.pi, high=np.pi)
        # if center_radius is specified, excluding the center region
        random_dist = np.random.uniform(low=center_radius if center_radius else 0, high=x_max)
        x_perturb, z_perturb = random_dist*np.cos(random_angle), random_dist*np.sin(random_angle)
        
        random_pos = np.array([x_perturb, height, z_perturb])
        random_pose = tra.random_rotation_matrix()
        random_pose[:3, 3] = random_pos

        poses.append(random_pose)

        if visualize:
            shape_pcd = trimesh.PointCloud(pcd)
            center_region_vis_transform = tra.euler_matrix(np.pi/2, 0, 0)
            center_region_vis_transform[1, 3] += height
            center_region_vis = trimesh.creation.cylinder(radius=center_radius, height=0.005, transform=center_region_vis_transform)
            local_frame_negative = trimesh.creation.axis(
                origin_size=0.002,
                transform=random_pose,
                origin_color=None,
                axis_radius=0.002,
                axis_length=0.05,
            )
            scene = trimesh.Scene()
            scene.add_geometry([shape_pcd, local_frame_negative, center_region_vis])
            scene.show()

    return poses


def sample_center_opening_pose(pcd, mask, num_poses=1, scaling=0.2, visualize=False):

    height = np.mean(pcd[mask], axis=0)[1]
    center_radius = np.max(pcd[mask], axis=0)[0] * scaling

    # create negative example
    poses = []
    for i in range(num_poses):

        random_angle = np.random.uniform(low=-np.pi, high=np.pi)
        random_dist = np.random.uniform(low=0, high=center_radius)
        x_perturb, z_perturb = random_dist * np.cos(random_angle), random_dist * np.sin(random_angle)

        height_perturb = height + np.random.uniform(low=-center_radius, high=center_radius)

        random_pos = np.array([x_perturb, height_perturb, z_perturb])
        random_pose = tra.euler_matrix(0, np.random.uniform(high=np.pi * 2), 0)
        random_pose[:3, 3] = random_pos

        poses.append(random_pose)

        if visualize:
            shape_pcd = trimesh.PointCloud(pcd)
            local_frame_negative = trimesh.creation.axis(
                origin_size=0.002,
                transform=random_pose,
                origin_color=None,
                axis_radius=0.002,
                axis_length=0.05,
            )
            scene = trimesh.Scene()
            scene.add_geometry([shape_pcd, local_frame_negative])
            scene.show()

    return poses, center_radius


def sample_negative_top_pose(pcd, mask, center_x_bounds, num_poses=1, scaling=0.2, visualize=False):

    height = np.mean(pcd[mask], axis=0)[1]
    x_min, x_max = np.min(pcd[mask], axis=0)[0], np.max(pcd[mask], axis=0)[0]
    center_x_bounds = (x_min * scaling, x_max * scaling)

    # create negative example
    poses = []
    for i in range(num_poses):

        random_angle = np.random.uniform(low=-np.pi, high=np.pi)
        random_dist = np.random.uniform(low=center_x_bounds[0], high=center_x_bounds[1])
        x_perturb, z_perturb = random_dist * np.cos(random_angle), random_dist * np.sin(random_angle)

        height_perturb = height + np.random.uniform(low=center_x_bounds[0], high=center_x_bounds[1])

        random_pos = np.array([x_perturb, height_perturb, z_perturb])
        random_pose = tra.euler_matrix(0, np.random.uniform(high=np.pi * 2), 0)
        random_pose[:3, 3] = random_pos

        poses.append(random_pose)

        if visualize:
            shape_pcd = trimesh.PointCloud(pcd)
            local_frame_negative = trimesh.creation.axis(
                origin_size=0.002,
                transform=random_pose,
                origin_color=None,
                axis_radius=0.002,
                axis_length=0.05,
            )
            scene = trimesh.Scene()
            scene.add_geometry([shape_pcd, local_frame_negative])
            scene.show()

    return poses, center_x_bounds


def compute_pose_feature(
    model,
    pcd,
    pose,
    sigma=0.025,
    n_opt_pts=500,
    n_pts=1500,
    device="cpu",
    visualize=False,
):

    # sample query points
    query_pts = np.random.normal(0.0, sigma, size=(n_opt_pts, 3))

    reference_query_pts = torch_util.transform_pcd(query_pts, pose)

    if visualize:
        shape_pcd = trimesh.PointCloud(pcd[:n_pts])
        reference_pts_pcd = trimesh.PointCloud(reference_query_pts)
        reference_pts_pcd.colors = np.tile(
            (255, 0, 0), (reference_pts_pcd.vertices.shape[0], 1)
        )

        local_frame_negative = trimesh.creation.axis(
            origin_size=0.002,
            transform=pose,
            origin_color=None,
            axis_radius=0.002,
            axis_length=0.05,
        )
        scene = trimesh.Scene()
        scene.add_geometry([shape_pcd, local_frame_negative, reference_pts_pcd])
        scene.show()

    reference_model_input = {}
    ref_query_pts = torch.from_numpy(reference_query_pts[:n_opt_pts]).float().to(device)
    ref_shape_pcd = torch.from_numpy(pcd[:n_pts]).float().to(device)
    reference_model_input["coords"] = ref_query_pts[None, :, :]
    reference_model_input["point_cloud"] = ref_shape_pcd[None, :, :]

    # get the descriptors for these reference query points
    reference_latent = model.extract_latent(reference_model_input).detach()
    reference_act_hat = model.forward_latent(
        reference_latent, reference_model_input["coords"]
    ).detach()

    return reference_act_hat


def load_objects(model, obj_model_path, device="cpu", visualize=False):

    scale1 = 0.25
    mesh1 = trimesh.load(obj_model_path, process=False)
    mesh1.apply_scale(scale1)
    # convert to pc
    pcd1 = mesh1.sample(5000)

    # pos_poses, bottom_mask = sample_bottom_pose(
    #     pcd1, mesh1, num_poses=3, visualize=visualize
    # )

    pos_poses, top_mask = sample_top_pose(pcd1,mesh1, num_poses=3, visualize=visualize)
    # ToDo: add hard examples where only orientations are wrong
    # ToDo: add hard examples where only positions are wrong
    # neg_poses = sample_random_pose(pcd1, bottom_mask, num_poses=3, visualize=visualize)
    # neg_poses = sample_random_pose(pcd1, top_mask, num_poses=3, visualize=visualize)
    neg_poses = sample_opening_pose(pcd1, top_mask, num_poses=3, visualize=visualize)

    pos_feats = []
    for pos_pose in pos_poses:
        pose_feature = compute_pose_feature(
            model, pcd1, pos_pose, visualize=visualize, device=device
        )
        pos_feats.append(pose_feature)
    neg_feats = []
    for neg_pose in neg_poses:
        pose_feature = compute_pose_feature(
            model, pcd1, neg_pose, visualize=visualize, device=device
        )
        neg_feats.append(pose_feature)

    pos_feats = torch.cat(pos_feats, dim=0).cpu().numpy().reshape((len(pos_feats), -1))
    neg_feats = torch.cat(neg_feats, dim=0).cpu().numpy().reshape((len(neg_feats), -1))

    pos_dists = np.linalg.norm(
        pos_feats[:, None, :] - pos_feats[None, :, :], axis=-1, ord=1
    )
    print(pos_dists)
    neg_dists = np.linalg.norm(
        pos_feats[:, None, :] - neg_feats[None, :, :], axis=-1, ord=1
    )
    print(neg_dists)


if __name__ == "__main__":
    # train_mlp("valid", max_epochs=1, checkpoint_path="/home/weiyu/Research/ndf_robot/src/ndf_robot/eval/lightning_logs/version_0/checkpoints/epoch=0-step=539.ckpt")

    # device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    # model_path = osp.join(path_util.get_ndf_model_weights(), 'multi_category_weights.pth')
    # model = vnn_occupancy_network.VNNOccNet(latent_dim=256, model_type='pointnet', return_features=True,
    #                                         sigmoid=True).to(device)
    # model.load_state_dict(torch.load(model_path))

    # print(path_util.get_ndf_obj_descriptions())
    object_base_dir = path_util.get_ndf_obj_descriptions()
    class_to_obj_paths = defaultdict(list)
    for obj_class in os.listdir(object_base_dir):
        if "centered_obj_normalized" in obj_class:
            for obj_model in os.listdir(os.path.join(object_base_dir, obj_class)):
                class_to_obj_paths[obj_class].append(os.path.join(object_base_dir, obj_class, obj_model, "models/model_normalized.obj"))

    for obj_class in class_to_obj_paths:
        # print(obj_class, len(class_to_obj_paths[obj_class]))
        # load_objects(class_to_obj_paths[obj_class][0])
        print(obj_class)
        if obj_class == "bowl_centered_obj_normalized":
            count = 0
            for obj_path in class_to_obj_paths[obj_class]:

                scale1 = 0.25
                mesh1 = trimesh.load(obj_path, process=False)
                mesh1.apply_scale(scale1)
                # convert to pc
                pcd1 = mesh1.sample(5000)
                _, top_mask = sample_top_pose(pcd1, mesh1, num_poses=0)
                pos_poses, center_radius = sample_center_opening_pose(pcd1, top_mask, scaling=0.2, num_poses=5, visualize=True)
                neg_poses = sample_opening_pose(pcd1, top_mask, center_radius=center_radius, num_poses=5, scaling=0.9, visualize=True)

                # print(obj_path)
                # load_objects(model, obj_path, device, visualize=True)
                count += 1
                # if count == 1:
                    # break

        # break