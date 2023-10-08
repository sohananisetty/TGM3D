import os

import numpy as np
import torch
from utils.motion_processing.quaternion import (
    qbetween_np,
    qfix,
    qinv,
    qinv_np,
    qmul_np,
    qrot,
    qrot_np,
    quaternion_to_cont6d,
    quaternion_to_cont6d_np,
)
from .skeleton import Skeleton, t2m_kinematic_chain, t2m_raw_offsets
from human_body_prior.body_model.body_model import BodyModel
import utils.rotation_conversions as geometry

l_idx1, l_idx2 = 5, 8
# Right/Left foot
fid_r, fid_l = [8, 11], [7, 10]
# Face direction, r_hip, l_hip, sdr_r, sdr_l
face_joint_indx = [2, 1, 17, 16]
# l_hip, r_hip
r_hip, l_hip = 2, 1
joints_num = 22
n_raw_offsets = torch.from_numpy(t2m_raw_offsets)
kinematic_chain = t2m_kinematic_chain


def get_target_offset():
    example_data = np.load(
        "/srv/hays-lab/scratch/sanisetty3/music_motion/HumanML3D/joints/000021.npy"
    )
    example_data = example_data.reshape(len(example_data), -1, 3)
    example_data = torch.from_numpy(example_data)
    tgt_skel = Skeleton(n_raw_offsets, kinematic_chain, "cpu")
    # (joints_num, 3)
    tgt_offsets = tgt_skel.get_offsets_joints(example_data[0])
    return tgt_offsets


def uniform_skeleton(
    positions,
    target_offset,
):
    src_skel = Skeleton(n_raw_offsets, kinematic_chain, "cpu")
    src_offset = src_skel.get_offsets_joints(torch.from_numpy(positions[0]))
    src_offset = src_offset.numpy()
    tgt_offset = target_offset.numpy()
    # print(src_offset)
    # print(tgt_offset)
    """Calculate Scale Ratio as the ratio of legs"""
    src_leg_len = np.abs(src_offset[l_idx1]).max() + np.abs(src_offset[l_idx2]).max()
    tgt_leg_len = np.abs(tgt_offset[l_idx1]).max() + np.abs(tgt_offset[l_idx2]).max()

    scale_rt = tgt_leg_len / src_leg_len
    # print(scale_rt)
    src_root_pos = positions[:, 0]
    tgt_root_pos = src_root_pos * scale_rt

    """Inverse Kinematics"""
    quat_params = src_skel.inverse_kinematics_np(positions, face_joint_indx)
    # print(quat_params.shape)

    """Forward Kinematics"""
    src_skel.set_offset(target_offset)
    new_joints = src_skel.forward_kinematics_np(quat_params, tgt_root_pos)
    return new_joints


def process_file(positions, feet_thre):
    # (seq_len, joints_num, 3)
    #     '''Down Sample'''
    #     positions = positions[::ds_num]

    tgt_offsets = get_target_offset()

    """Uniform Skeleton"""
    positions = uniform_skeleton(positions, tgt_offsets)

    """Put on Floor"""
    floor_height = positions.min(axis=0).min(axis=0)[1]
    positions[:, :, 1] -= floor_height

    """XZ at origin"""
    root_pos_init = positions[0]
    root_pose_init_xz = root_pos_init[0] * np.array([1, 0, 1])
    positions = positions - root_pose_init_xz

    # '''Move the first pose to origin '''
    # root_pos_init = positions[0]
    # positions = positions - root_pos_init[0]

    """All initially face Z+"""
    r_hip, l_hip, sdr_r, sdr_l = face_joint_indx
    across1 = root_pos_init[r_hip] - root_pos_init[l_hip]
    across2 = root_pos_init[sdr_r] - root_pos_init[sdr_l]
    across = across1 + across2
    across = across / np.sqrt((across**2).sum(axis=-1))[..., np.newaxis]

    # forward (3,), rotate around y-axis
    forward_init = np.cross(np.array([[0, 1, 0]]), across, axis=-1)
    # forward (3,)
    forward_init = (
        forward_init / np.sqrt((forward_init**2).sum(axis=-1))[..., np.newaxis]
    )

    target = np.array([[0, 0, 1]])
    root_quat_init = qbetween_np(forward_init, target)
    root_quat_init = np.ones(positions.shape[:-1] + (4,)) * root_quat_init

    positions_b = positions.copy()

    positions = qrot_np(root_quat_init, positions)

    """New ground truth positions"""
    global_positions = positions.copy()

    """ Get Foot Contacts """

    def foot_detect(positions, thres):
        velfactor, heightfactor = np.array([thres, thres]), np.array([3.0, 2.0])

        feet_l_x = (positions[1:, fid_l, 0] - positions[:-1, fid_l, 0]) ** 2
        feet_l_y = (positions[1:, fid_l, 1] - positions[:-1, fid_l, 1]) ** 2
        feet_l_z = (positions[1:, fid_l, 2] - positions[:-1, fid_l, 2]) ** 2
        #     feet_l_h = positions[:-1,fid_l,1]
        #     feet_l = (((feet_l_x + feet_l_y + feet_l_z) < velfactor) & (feet_l_h < heightfactor)).astype(np.float)
        feet_l = ((feet_l_x + feet_l_y + feet_l_z) < velfactor).astype(np.float32)

        feet_r_x = (positions[1:, fid_r, 0] - positions[:-1, fid_r, 0]) ** 2
        feet_r_y = (positions[1:, fid_r, 1] - positions[:-1, fid_r, 1]) ** 2
        feet_r_z = (positions[1:, fid_r, 2] - positions[:-1, fid_r, 2]) ** 2
        #     feet_r_h = positions[:-1,fid_r,1]
        #     feet_r = (((feet_r_x + feet_r_y + feet_r_z) < velfactor) & (feet_r_h < heightfactor)).astype(np.float)
        feet_r = (((feet_r_x + feet_r_y + feet_r_z) < velfactor)).astype(np.float32)
        return feet_l, feet_r

    #
    feet_l, feet_r = foot_detect(positions, feet_thre)
    # feet_l, feet_r = foot_detect(positions, 0.002)

    """Quaternion and Cartesian representation"""
    r_rot = None

    def get_rifke(positions):
        """Local pose"""
        positions[..., 0] -= positions[:, 0:1, 0]
        positions[..., 2] -= positions[:, 0:1, 2]
        """All pose face Z+"""
        positions = qrot_np(
            np.repeat(r_rot[:, None], positions.shape[1], axis=1), positions
        )
        return positions

    def get_quaternion(positions):
        skel = Skeleton(n_raw_offsets, kinematic_chain, "cpu")
        # (seq_len, joints_num, 4)
        quat_params = skel.inverse_kinematics_np(
            positions, face_joint_indx, smooth_forward=False
        )

        """Fix Quaternion Discontinuity"""
        quat_params = qfix(quat_params)
        # (seq_len, 4)
        r_rot = quat_params[:, 0].copy()
        #     print(r_rot[0])
        """Root Linear Velocity"""
        # (seq_len - 1, 3)
        velocity = (positions[1:, 0] - positions[:-1, 0]).copy()
        #     print(r_rot.shape, velocity.shape)
        velocity = qrot_np(r_rot[1:], velocity)
        """Root Angular Velocity"""
        # (seq_len - 1, 4)
        r_velocity = qmul_np(r_rot[1:], qinv_np(r_rot[:-1]))
        quat_params[1:, 0] = r_velocity
        # (seq_len, joints_num, 4)
        return quat_params, r_velocity, velocity, r_rot

    def get_cont6d_params(positions):
        skel = Skeleton(n_raw_offsets, kinematic_chain, "cpu")
        # (seq_len, joints_num, 4)
        quat_params = skel.inverse_kinematics_np(
            positions, face_joint_indx, smooth_forward=True
        )

        """Quaternion to continuous 6D"""
        cont_6d_params = quaternion_to_cont6d_np(quat_params)
        # (seq_len, 4)
        r_rot = quat_params[:, 0].copy()
        #     print(r_rot[0])
        """Root Linear Velocity"""
        # (seq_len - 1, 3)
        velocity = (positions[1:, 0] - positions[:-1, 0]).copy()
        #     print(r_rot.shape, velocity.shape)
        velocity = qrot_np(r_rot[1:], velocity)
        """Root Angular Velocity"""
        # (seq_len - 1, 4)
        r_velocity = qmul_np(r_rot[1:], qinv_np(r_rot[:-1]))
        # (seq_len, joints_num, 4)
        return cont_6d_params, r_velocity, velocity, r_rot

    cont_6d_params, r_velocity, velocity, r_rot = get_cont6d_params(positions)
    positions = get_rifke(positions)

    """Root height"""
    root_y = positions[:, 0, 1:2]

    """Root rotation and linear velocity"""
    # (seq_len-1, 1) rotation velocity along y-axis
    # (seq_len-1, 2) linear velovity on xz plane
    r_velocity = np.arcsin(r_velocity[:, 2:3])
    l_velocity = velocity[:, [0, 2]]
    #     print(r_velocity.shape, l_velocity.shape, root_y.shape)
    root_data = np.concatenate([r_velocity, l_velocity, root_y[:-1]], axis=-1)

    """Get Joint Rotation Representation"""
    # (seq_len, (joints_num-1) *6) quaternion for skeleton joints
    rot_data = cont_6d_params[:, 1:].reshape(len(cont_6d_params), -1)

    """Get Joint Rotation Invariant Position Represention"""
    # (seq_len, (joints_num-1)*3) local joint position
    ric_data = positions[:, 1:].reshape(len(positions), -1)

    """Get Joint Velocity Representation"""
    # (seq_len-1, joints_num*3)
    local_vel = qrot_np(
        np.repeat(r_rot[:-1, None], global_positions.shape[1], axis=1),
        global_positions[1:] - global_positions[:-1],
    )
    local_vel = local_vel.reshape(len(local_vel), -1)

    data = root_data
    data = np.concatenate([data, ric_data[:-1]], axis=-1)
    data = np.concatenate([data, rot_data[:-1]], axis=-1)
    #     print(data.shape, local_vel.shape)
    data = np.concatenate([data, local_vel], axis=-1)
    data = np.concatenate([data, feet_l, feet_r], axis=-1)

    return data, global_positions, positions, l_velocity


def recover_root_rot_pos(data):
    rot_vel = data[..., 0]
    r_rot_ang = torch.zeros_like(rot_vel).to(data.device)
    """Get Y-axis rotation from rotation velocity"""
    r_rot_ang[..., 1:] = rot_vel[..., :-1]
    r_rot_ang = torch.cumsum(r_rot_ang, dim=-1)

    r_rot_quat = torch.zeros(data.shape[:-1] + (4,)).to(data.device)
    r_rot_quat[..., 0] = torch.cos(r_rot_ang)
    r_rot_quat[..., 2] = torch.sin(r_rot_ang)

    r_pos = torch.zeros(data.shape[:-1] + (3,)).to(data.device)
    data = data.to(torch.float)
    r_pos[..., 1:, [0, 2]] = data[..., :-1, 1:3]
    """Add Y-axis rotation to root position"""
    r_pos = qrot(qinv(r_rot_quat), r_pos)

    r_pos = torch.cumsum(r_pos, dim=-2)

    r_pos[..., 1] = data[..., 3]
    return r_rot_quat, r_pos


def recover_from_rot(data, joints_num, skeleton):
    r_rot_quat, r_pos = recover_root_rot_pos(data)

    r_rot_cont6d = quaternion_to_cont6d(r_rot_quat)

    start_indx = 1 + 2 + 1 + (joints_num - 1) * 3
    end_indx = start_indx + (joints_num - 1) * 6
    cont6d_params = data[..., start_indx:end_indx]
    #     print(r_rot_cont6d.shape, cont6d_params.shape, r_pos.shape)
    cont6d_params = torch.cat([r_rot_cont6d, cont6d_params], dim=-1)
    cont6d_params = cont6d_params.view(-1, joints_num, 6)

    positions = skeleton.forward_kinematics_cont6d(cont6d_params, r_pos)

    return positions


def recover_from_ric(data, joints_num):
    r_rot_quat, r_pos = recover_root_rot_pos(data)
    positions = data[..., 4 : (joints_num - 1) * 3 + 4]
    positions = positions.view(positions.shape[:-1] + (-1, 3)).to(torch.float32)

    """Add Y-axis rotation to local joints"""
    positions = qrot(
        qinv(r_rot_quat[..., None, :]).expand(positions.shape[:-1] + (4,)), positions
    )

    """Add root XZ to joints"""
    positions[..., 0] += r_pos[..., 0:1]
    positions[..., 2] += r_pos[..., 2:3]

    """Concate root and joints"""
    positions = torch.cat([r_pos.unsqueeze(-2), positions], dim=-2)

    return positions


class HMLProcess:
    def __init__(self, device="cuda"):
        self.device = device
        male_bm_path = "/srv/hays-lab/scratch/sanisetty3/music_motion/TGM3D/body_models/smplh/male/model.npz"
        male_dmpl_path = "/srv/hays-lab/scratch/sanisetty3/music_motion/TGM3D/body_models/dmpls/male/model.npz"

        female_bm_path = "/srv/hays-lab/scratch/sanisetty3/music_motion/TGM3D/body_models/smplh/female/model.npz"
        female_dmpl_path = "/srv/hays-lab/scratch/sanisetty3/music_motion/TGM3D/body_models/dmpls/female/model.npz"

        num_betas = 10  # number of body parameters
        num_dmpls = 8  # number of DMPL parameters

        self.male_bm = BodyModel(
            bm_fname=male_bm_path,
            num_betas=num_betas,
            num_dmpls=num_dmpls,
            dmpl_fname=male_dmpl_path,
        ).to(device)
        self.faces = (self.male_bm.f).cpu()

        self.female_bm = BodyModel(
            bm_fname=female_bm_path,
            num_betas=num_betas,
            num_dmpls=num_dmpls,
            dmpl_fname=female_dmpl_path,
        ).to(device)

    def swap_left_right(self, data):
        assert len(data.shape) == 3 and data.shape[-1] == 3
        data = data.copy()
        data[..., 0] *= -1
        right_chain = [2, 5, 8, 11, 14, 17, 19, 21]
        left_chain = [1, 4, 7, 10, 13, 16, 18, 20]
        left_hand_chain = [22, 23, 24, 34, 35, 36, 25, 26, 27, 31, 32, 33, 28, 29, 30]
        right_hand_chain = [43, 44, 45, 46, 47, 48, 40, 41, 42, 37, 38, 39, 49, 50, 51]
        tmp = data[:, right_chain]
        data[:, right_chain] = data[:, left_chain]
        data[:, left_chain] = tmp
        if data.shape[1] > 24:
            tmp = data[:, right_hand_chain]
            data[:, right_hand_chain] = data[:, left_hand_chain]
            data[:, left_hand_chain] = tmp
        return data

    def aa2joints(self, data, gender="male", betas=None):
        ## data: axis angle
        with torch.no_grad():
            root_orient = torch.Tensor(data[:, 3:6]).to(
                self.device
            )  # controls the global root orientation
            pose_body = torch.Tensor(data[:, 6:69]).to(self.device)  # controls the body
            trans = torch.Tensor(data[:, :3]).to(self.device)
            if gender == "male":
                body = self.male_bm(
                    pose_body=pose_body, betas=betas, root_orient=root_orient
                )
            else:
                body = self.female_bm(
                    pose_body=pose_body, betas=betas, root_orient=root_orient
                )
            pose_seq = body.Jtr + trans[:, None, :]

        return pose_seq, body

    def to_aist_axis_angle_from6d(self, smpl_motion: torch.Tensor) -> torch.Tensor:
        trans = smpl_motion[:, :3]
        rots = smpl_motion[:, 3:]

        new_aa_rotations = geometry.matrix_to_axis_angle(
            geometry.rotation_6d_to_matrix(rots.reshape(-1, 22, 6))
        ).reshape(rots.shape[0], -1)
        smpl_aa = torch.cat([trans, new_aa_rotations], 1)

        return smpl_aa

    def get_hml_rep(self, positions: torch.Tensor):
        data, ground_positions, positions, l_velocity = process_file(
            np.array(positions), 0.002
        )
        rec_ric_data = recover_from_ric(
            torch.from_numpy(data).unsqueeze(0).float(), joints_num
        )

        return data, rec_ric_data.squeeze().numpy()

    def smpl2hml(self, motion6d):
        motion_aa = self.to_aist_axis_angle_from6d(motion6d)
        pose_seq, body = self.aa2joints(motion_aa)
        data, rec_ric_data = self.get_hml_rep(pose_seq.cpu().numpy())

        return data
