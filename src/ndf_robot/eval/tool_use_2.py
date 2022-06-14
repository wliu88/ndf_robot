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

        self.ndf_model = vnn_occupancy_network.VNNOccNet(latent_dim=256, model_type='pointnet', return_features=True,
                                                sigmoid=True)
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
            nn.Linear(32, 1)
        )
        self.loss = torch.nn.BCEWithLogitsLoss(reduction="mean")

    def forward(self, x):
        self.ndf_model.eval()
        with torch.no_grad():
            reference_latent = self.ndf_model.extract_latent(x)
            reference_act_hat = self.ndf_model.forward_latent(reference_latent, x['coords'])
            reference_act_hat = torch.mean(reference_act_hat, dim=1)

        y_hat = self.layers(reference_act_hat)
        return y_hat

    def training_step(self, batch, batch_idx):
        x, y = batch

        self.ndf_model.eval()
        with torch.no_grad():
            reference_latent = self.ndf_model.extract_latent(x)
            reference_act_hat = self.ndf_model.forward_latent(reference_latent, x['coords'])
            reference_act_hat = torch.mean(reference_act_hat, dim=1)

        y_hat = self.layers(reference_act_hat)
        loss = self.loss(y_hat, y)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch

        self.ndf_model.eval()
        with torch.no_grad():
            reference_latent = self.ndf_model.extract_latent(x)
            reference_act_hat = self.ndf_model.forward_latent(reference_latent, x['coords'])
            reference_act_hat = torch.mean(reference_act_hat, dim=1)

        y_hat = self.layers(reference_act_hat)
        loss = self.loss(y_hat, y)
        self.log('val_loss', loss, prog_bar=True)
        print("prediction:", y_hat)
        print("gt:", y)
        input("next?")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer


def train_mlp():
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    model_path = osp.join(path_util.get_ndf_model_weights(), 'multi_category_weights.pth')

    train_obj_models, val_obj_models = split_objects(train_ratio=0.8)
    train_dataset = SemanticPoseDataset(train_obj_models, debug=False)
    val_dataset = SemanticPoseDataset(val_obj_models, debug=True)

    pl.seed_everything(42)
    mlp = NDFClassifier(model_path=model_path)

    trainer = pl.Trainer(gpus=1, deterministic=True, max_epochs=5)
    trainer.fit(mlp, train_dataloaders=DataLoader(train_dataset, batch_size=8))
    trainer.validate(dataloaders=DataLoader(val_dataset, batch_size=1))

    # for x, y in DataLoader(val_dataset, batch_size=1):
    #     for k in x:
    #         x[k] = x[k].to(device)
    #     score = mlp.forward(x)
    #     print(score)
    #     input("next?")


######################################################


def split_objects(object_base_dir = path_util.get_ndf_obj_descriptions(), train_ratio=0.7):
    all_obj_models = []
    for obj_class in os.listdir(object_base_dir):
        if "centered_obj_normalized" in obj_class:
            for obj_model in os.listdir(os.path.join(object_base_dir, obj_class)):
                all_obj_models.append(obj_model)
    print(all_obj_models)
    rix = np.random.permutation(len(all_obj_models))
    train_num = int(len(all_obj_models) * train_ratio)
    train_obj_models = [all_obj_models[i] for i in rix[:train_num]]
    val_obj_models = [all_obj_models[i] for i in rix[train_num:]]
    return train_obj_models, val_obj_models

class SemanticPoseDataset(torch.utils.data.Dataset):

    def __init__(self, split_obj_models=None, object_base_dir=path_util.get_ndf_obj_descriptions(), debug=False,
                 num_pos_poses=3, num_neg_poses=3, random_noise_sigma=0.001, bottom_scale_ratio=0.01, scale=0.25,
                 feat_sigma=0.025, feat_n_opt_pts=500, feat_n_pts=1500):

        # params
        self.random_noise_sigma = random_noise_sigma
        self.bottom_scale_ratio = bottom_scale_ratio
        self.scale = scale

        # params for ndf features
        self.feat_sigma = feat_sigma
        self.feat_n_opt_pts = feat_n_opt_pts
        self.feat_n_pts = feat_n_pts

        self.debug = debug
        self.num_pos_poses = num_pos_poses
        self.num_neg_poses = num_neg_poses

        self.obj_paths = []
        self.obj_path_to_data = {}

        for obj_class in os.listdir(object_base_dir):
            if "centered_obj_normalized" in obj_class:
                for obj_model in os.listdir(os.path.join(object_base_dir, obj_class)):
                    if split_obj_models is None or obj_model in split_obj_models:
                        obj_path = os.path.join(object_base_dir, obj_class, obj_model, "models/model_normalized.obj")
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
            pcd, bottom_mask = self.get_obj_pcd_and_bottom_mask(obj_path)
            self.obj_path_to_data[obj_path]["pcd"] = pcd
            self.obj_path_to_data[obj_path]["bottom_mask"] = bottom_mask

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

    def get_raw_data(self, idx):
        """
        retrieve one data point
        :param idx:
        :return:
        """

        obj_path, label = self.data[idx]

        pcd, bottom_mask = self.obj_path_to_data[obj_path]["pcd"], self.obj_path_to_data[obj_path]["bottom_mask"]
        pcd_bottom = pcd[bottom_mask]

        if label:
            pose = tra.euler_matrix(np.pi, np.random.uniform(high=np.pi * 2), 0)
            pose[:3, 3] = pcd_bottom[np.random.choice(list(range(len(pcd_bottom)))), :]
            pose[:3, 3] += np.random.normal(scale=self.random_noise_sigma, size=3)
        else:
            random_idx = np.random.randint(pcd[~bottom_mask].shape[0])
            random_pos = pcd[~bottom_mask][random_idx]
            random_pose = tra.random_rotation_matrix()
            random_pose[:3, 3] = random_pos
            pose = random_pose

        if self.debug:
            shape_pcd = trimesh.PointCloud(pcd)
            bottom_pcd = trimesh.PointCloud(pcd_bottom)
            bottom_pcd.colors = np.tile((255, 0, 0), (bottom_pcd.vertices.shape[0], 1))
            local_frame_positive = trimesh.creation.axis(origin_size=0.002, transform=pose, origin_color=None,
                                                         axis_radius=0.002,
                                                         axis_length=0.05)
            scene = trimesh.Scene()
            scene.add_geometry([shape_pcd, local_frame_positive, bottom_pcd])
            scene.show()

        # sample query points
        query_pts = np.random.normal(0.0, self.feat_sigma, size=(self.feat_n_opt_pts, 3))

        # put the query points at the position of the provided pose
        # q_offset = pose[:3, 3]
        # q_offset *= 1.2
        # reference_query_pts = query_pts + q_offset

        reference_query_pts = torch_util.transform_pcd(query_pts, pose)

        # if visualize:
        #     shape_pcd = trimesh.PointCloud(pcd[:n_pts])
        #     reference_pts_pcd = trimesh.PointCloud(reference_query_pts)
        #     reference_pts_pcd.colors = np.tile((255, 0, 0), (reference_pts_pcd.vertices.shape[0], 1))
        #
        #     local_frame_negative = trimesh.creation.axis(origin_size=0.002, transform=pose, origin_color=None,
        #                                                  axis_radius=0.002,
        #                                                  axis_length=0.05)
        #     scene = trimesh.Scene()
        #     scene.add_geometry([shape_pcd, local_frame_negative, reference_pts_pcd])
        #     scene.show()

        reference_model_input = {}
        ref_query_pts = torch.from_numpy(reference_query_pts[:self.feat_n_opt_pts]).float()
        ref_shape_pcd = torch.from_numpy(pcd[:self.feat_n_pts]).float()

        # put all the input data in a dictionary
        datum = {"coords": ref_query_pts,
                 "point_cloud": ref_shape_pcd}

        return datum, torch.FloatTensor([label])

    def __getitem__(self, idx):

        # datum = self.convert_to_tensors(self.get_raw_data(idx))
        datum = self.get_raw_data(idx)

        return datum

    # @staticmethod
    # def collate_fn(data):
    #     """
    #     used to specify how to combine data points into a batch
    #     :param data:
    #     :return:
    #     """
    #
    #     # torch.cat() and torch.stack() can be useful
    #     batched_data_dict = {}
    #     for key in ["xyzs"]:
    #         batched_data_dict[key] = torch.cat([dict[key] for dict in data], dim=0)
    #     for key in ["obj_x_inputs"]:
    #         batched_data_dict[key] = torch.stack([dict[key] for dict in data], dim=0)
    #
    #     return batched_data_dict


def build_simple_semantic_classifier():

    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    model_path = osp.join(path_util.get_ndf_model_weights(), 'multi_category_weights.pth')
    model = vnn_occupancy_network.VNNOccNet(latent_dim=256, model_type='pointnet', return_features=True,
                                            sigmoid=True).to(device)
    model.load_state_dict(torch.load(model_path))

    feats = []
    labels = []

    spd = SemanticPoseDataset(debug=False)
    for datum in spd:
        pcd = datum["pcd"]
        pose_feature = compute_pose_feature(model, datum["pcd"], datum["pose"], visualize=False, device=device)

        print(pose_feature.shape)

        feats.append(pose_feature.cpu().numpy())
        labels.append(int(datum["label"]))
        input("here")

    # clf = svm.SVC()
    # clf.fit(feats, labels)
    #
    # preds = clf.predict(feats)
    # print(classification_report(labels, preds))

#####################################################################################################





def sample_bottom_pose(pcd, mesh, num_poses=1, bottom_scale_ratio=0.01, random_noise_sigma=0.001, visualize=False):

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
            local_frame_positive = trimesh.creation.axis(origin_size=0.002, transform=pose, origin_color=None,
                                                         axis_radius=0.002,
                                                         axis_length=0.05)
            scene = trimesh.Scene()
            scene.add_geometry([shape_pcd, local_frame_positive, bottom_pcd])
            scene.show()

    return poses, bottom_mask


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
            local_frame_negative = trimesh.creation.axis(origin_size=0.002, transform=random_pose, origin_color=None,
                                                         axis_radius=0.002,
                                                         axis_length=0.05)
            scene = trimesh.Scene()
            scene.add_geometry([shape_pcd, local_frame_negative])
            scene.show()

    return poses


def compute_pose_feature(model, pcd, pose, sigma=0.025, n_opt_pts=500, n_pts=1500, device="cpu", visualize=False):

    # sample query points
    query_pts = np.random.normal(0.0, sigma, size=(n_opt_pts, 3))

    # put the query points at the position of the provided pose
    # q_offset = pose[:3, 3]
    # q_offset *= 1.2
    # reference_query_pts = query_pts + q_offset

    reference_query_pts = torch_util.transform_pcd(query_pts, pose)

    if visualize:
        shape_pcd = trimesh.PointCloud(pcd[:n_pts])
        reference_pts_pcd = trimesh.PointCloud(reference_query_pts)
        reference_pts_pcd.colors = np.tile((255, 0, 0), (reference_pts_pcd.vertices.shape[0], 1))

        local_frame_negative = trimesh.creation.axis(origin_size=0.002, transform=pose, origin_color=None,
                                                     axis_radius=0.002,
                                                     axis_length=0.05)
        scene = trimesh.Scene()
        scene.add_geometry([shape_pcd, local_frame_negative, reference_pts_pcd])
        scene.show()

    reference_model_input = {}
    ref_query_pts = torch.from_numpy(reference_query_pts[:n_opt_pts]).float().to(device)
    ref_shape_pcd = torch.from_numpy(pcd[:n_pts]).float().to(device)
    reference_model_input['coords'] = ref_query_pts[None, :, :]
    reference_model_input['point_cloud'] = ref_shape_pcd[None, :, :]

    # get the descriptors for these reference query points
    reference_latent = model.extract_latent(reference_model_input).detach()
    reference_act_hat = model.forward_latent(reference_latent, reference_model_input['coords']).detach()

    return reference_act_hat


def load_objects(model, obj_model_path, device="cpu", visualize=False):

    scale1 = 0.25
    mesh1 = trimesh.load(obj_model_path, process=False)
    mesh1.apply_scale(scale1)
    # convert to pc
    pcd1 = mesh1.sample(5000)

    pos_poses, bottom_mask = sample_bottom_pose(pcd1, mesh1, num_poses=3, visualize=visualize)
    # ToDo: add hard examples where only orientations are wrong
    # ToDo: add hard examples where only positions are wrong
    neg_poses = sample_random_pose(pcd1, bottom_mask, num_poses=3, visualize=visualize)

    pos_feats = []
    for pos_pose in pos_poses:
        pose_feature = compute_pose_feature(model, pcd1, pos_pose, visualize=True, device=device)
        pos_feats.append(pose_feature)
    neg_feats = []
    for neg_pose in neg_poses:
        pose_feature = compute_pose_feature(model, pcd1, neg_pose, visualize=True, device=device)
        neg_feats.append(pose_feature)

    pos_feats = torch.cat(pos_feats, dim=0).cpu().numpy().reshape((len(pos_feats), -1))
    neg_feats = torch.cat(neg_feats, dim=0).cpu().numpy().reshape((len(neg_feats), -1))

    pos_dists = np.linalg.norm(pos_feats[:, None, :] - pos_feats[None, :, :], axis=-1, ord=1)
    print(pos_dists)
    neg_dists = np.linalg.norm(pos_feats[:, None, :] - neg_feats[None, :, :], axis=-1, ord=1)
    print(neg_dists)


    # local_frame_1 = trimesh.creation.axis(origin_size=0.002, transform=frame1_tf, origin_color=None, axis_radius=0.002,
    #                                       axis_length=0.05)
    # local_frame_2 = trimesh.creation.axis(origin_size=0.002, transform=rand_mat_np, origin_color=None,
    #                                       axis_radius=0.002, axis_length=0.05)
    # local_frame_3 = trimesh.creation.axis(origin_size=0.002, transform=frame2_tf, origin_color=None, axis_radius=0.002,
    #                                       axis_length=0.05)
    #
    # best_scene = trimesh_util.trimesh_show([vpcd1, vquery1, self.pcd2, best_X, pcd_traj_list[best_idx]], show=False)
    # best_scene.add_geometry([local_frame_1, local_frame_2, local_frame_3])
    # best_scene.show()







if __name__ == '__main__':

    #############################################################################################
    # device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    # model_path = osp.join(path_util.get_ndf_model_weights(), 'multi_category_weights.pth')
    # model = vnn_occupancy_network.VNNOccNet(latent_dim=256, model_type='pointnet', return_features=True,
    #                                         sigmoid=True).to(device)
    # model.load_state_dict(torch.load(model_path))
    #
    # print(path_util.get_ndf_obj_descriptions())
    # object_base_dir = path_util.get_ndf_obj_descriptions()
    # class_to_obj_paths = defaultdict(list)
    # for obj_class in os.listdir(object_base_dir):
    #     if "centered_obj_normalized" in obj_class:
    #         for obj_model in os.listdir(os.path.join(object_base_dir, obj_class)):
    #             class_to_obj_paths[obj_class].append(os.path.join(object_base_dir, obj_class, obj_model, "models/model_normalized.obj"))
    #
    # for obj_class in class_to_obj_paths:
    #     print(obj_class, len(class_to_obj_paths[obj_class]))
    #     # load_objects(class_to_obj_paths[obj_class][0])
    #
    #     count = 0
    #     for obj_path in class_to_obj_paths[obj_class]:
    #         load_objects(model, obj_path, device)
    #         count += 1
    #         if count == 1:
    #             break
    #
    #     break

    ############################################################################################

    # train_mlp()

    spd = SemanticPoseDataset(debug=True)
    for d in spd:
        print(d)



    # # seed = 1
    # # np.random.seed(seed)
    # # random.seed(seed)
    # # torch.random.manual_seed(seed)
    #
    # # see the demo object descriptions folder for other object models you can try
    # # obj_model1 = osp.join(path_util.get_ndf_demo_obj_descriptions(), 'mug_centered_obj_normalized/1a1c0a8d4bad82169f0594e65f756cf5/models/model_normalized.obj')
    # # obj_model2 = osp.join(path_util.get_ndf_demo_obj_descriptions(), 'mug_centered_obj_normalized/1a97f3c83016abca21d0de04f408950f/models/model_normalized.obj')
    # # model_path = osp.join(path_util.get_ndf_model_weights(), 'ndf_demo_mug_weights.pth')
    # # obj_model1 = osp.join(path_util.get_ndf_obj_descriptions(), 'mug_centered_obj_normalized/2d10421716b16580e45ef4135c266a12/models/model_normalized.obj')
    # obj_model1 = osp.join(path_util.get_ndf_demo_obj_descriptions(), 'mug_centered_obj_normalized/1a1c0a8d4bad82169f0594e65f756cf5/models/model_normalized.obj')
    # model_path = osp.join(path_util.get_ndf_model_weights(), 'multi_category_weights.pth')
    #
    # scale1 = 0.25
    # mesh1 = trimesh.load(obj_model1, process=False)
    # mesh1.apply_scale(scale1)
    #
    # # apply a random initial rotation to the new shape
    # # quat = np.random.random(4)
    # # quat = quat / np.linalg.norm(quat)
    # # rot = np.eye(4)
    # # rot[:-1, :-1] = Rotation.from_quat(quat).as_matrix()
    # # mesh1.apply_transform(rot)
    #
    # show_mesh1 = mesh1.copy()
    #
    # offset = 0.1
    # show_mesh1.apply_translation([0, 0, 0])
    #
    # scene = trimesh.Scene()
    # scene.add_geometry([show_mesh1])
    # scene.show()
    #
    # pcd1 = mesh1.sample(5000)
    #
    # shape_pcd = trimesh.PointCloud(pcd1)
    # shape_pcd.show()
    #
    # model = vnn_occupancy_network.VNNOccNet(latent_dim=256, model_type='pointnet', return_features=True, sigmoid=True).cuda()
    # model.load_state_dict(torch.load(model_path))
    #
    # # ndf_alignment = NDFAlignmentCheck(model, pcd1, pcd2, sigma=args.sigma, trimesh_viz=args.visualize)
    # ndf_alignment.sample_pts(show_recon=args.show_recon, render_video=args.video)