from glob import glob
from typing import List, Tuple

import torch
import utils.rotation_conversions as geometry
from core.models.smpl.smpl import SMPL
from scipy.spatial.transform import Rotation as R
import pandas as pd
import numpy as np
from tqdm import tqdm
from utils.motion_processing.hml_process import process_file


# Lower legs
l_idx1, l_idx2 = 5, 8
# Right/Left foot
fid_r, fid_l = [8, 11], [7, 10]
# Face direction, r_hip, l_hip, sdr_r, sdr_l
face_joint_indx = [2, 1, 17, 16]
# l_hip, r_hip
r_hip, l_hip = 2, 1

r_hip, l_hip, sdr_r, sdr_l = face_joint_indx


class AISTProcess:
    def __init__(self, device) -> None:
        self.device = device
        self.smpl_model = SMPL().eval().to(device)
        self.joints_num = 22

    def aist_6d_to_smpl(self, aist_vec):
        aist_ex = torch.Tensor(aist_vec)
        trans = aist_ex[:, :3].to(self.device)
        if aist_vec.shape[-1] == 135:
            rots = torch.cat((aist_ex[:, 3:], aist_ex[:, -12:]), 1)
        else:
            rots = aist_ex[:, 3:]
        aist_ex_9d = geometry.rotation_6d_to_matrix(rots.reshape(-1, 6)).reshape(
            aist_vec.shape[0], 24, 3, 3
        )
        global_orient = aist_ex_9d[:, 0:1].to(self.device)
        rotations = aist_ex_9d[:, 1:].to(self.device)
        out_aist_og = self.smpl_model(
            body_pose=rotations, global_orient=global_orient, transl=trans
        )

        return out_aist_og

    def aist_6d_to_9d(self, motion):
        if motion.shape[-1] == 225:
            motion = motion[:, 6:]
        smpl_trans = motion[:, :3]
        smpl_poses = motion[:, 3:]
        n_frames = smpl_poses.shape[0]
        smpl_poses = smpl_poses.reshape(n_frames, 24, 3, 3)
        smpl_poses_6d = geometry.matrix_to_rotation_6d(
            torch.Tensor(smpl_poses)
        ).reshape(n_frames, -1)
        smpl_motion = torch.cat([smpl_trans, smpl_poses_6d], axis=-1)

        return smpl_motion

    def aist_axis_angle_to_6d(self, smpl_motion: torch.Tensor) -> torch.Tensor:
        trans = smpl_motion[:, :3]
        rots = smpl_motion[:, 3:]

        new_6d_rotations = geometry.matrix_to_rotation_6d(
            geometry.axis_angle_to_matrix(rots.reshape(-1, 3))
        ).reshape(rots.shape[0], -1)
        smpl_6d = torch.cat([trans, new_6d_rotations], 1)

        return smpl_6d

    def to_aist_axis_angle_from6d(self, smpl_motion: torch.Tensor) -> torch.Tensor:
        trans = smpl_motion[:, :3]
        rots = smpl_motion[:, 3:]

        new_aa_rotations = geometry.matrix_to_axis_angle(
            geometry.rotation_6d_to_matrix(rots.reshape(-1, 22, 6))
        ).reshape(rots.shape[0], -1)
        smpl_aa = torch.cat([trans, new_aa_rotations], 1)

        return smpl_aa

    def process_and_center_aist(self, sml: torch.Tensor) -> torch.Tensor:
        ## up="y"
        seq_len = sml.shape[0]

        transl = sml[..., :, :3]
        rot_6D = sml[..., :, 3:].reshape(seq_len, -1, 6)

        if sml.shape[-1] == 135:
            rot_6D = torch.cat((rot_6D, rot_6D[:, -2:]), -2)

        rotmats = geometry.rotation_6d_to_matrix(rot_6D)

        smpl_output = self.smpl_model.forward(
            global_orient=rotmats[:, :1],
            body_pose=rotmats[:, 1:],
            transl=transl,
        )

        positions_kyp = smpl_output["smpl"]

        """Put on Floor"""
        floor_height = (positions_kyp.min(axis=0).values).min(axis=0).values[1]

        positions_ = sml[:, :3]

        positions_[:, 1] -= floor_height
        """XZ at origin"""
        root_pos_init = positions_[0]
        root_pose_init_xy = root_pos_init * torch.Tensor([1, 0, 1]).to(self.device)
        positions_ = positions_ - root_pose_init_xy

        sml[:, :3] = positions_

        return sml

    def resample(self, dat: torch.Tensor, in_fps=24, out_fps=20):
        df = pd.DataFrame(dat)
        df["timestamp"] = pd.date_range(
            start="2023-08-23 00:00:00",
            periods=dat.shape[0],
            freq=f"{round((1/in_fps)*1000 , 4)}ms",
        )
        df.set_index("timestamp", inplace=True)
        resampled_df = df.resample(
            f"{round((1/out_fps)*1000 , 5)}ms"
        ).mean()  # Resample to 20 fps
        resampled_motion = torch.Tensor(np.array(resampled_df))
        return resampled_motion

    def rotate_smpl_to(
        self, input_smpl: torch.Tensor, target_quat: List[int]
    ) -> torch.Tensor:
        """
        rotates input motion to the desired orientation by rotating the root joint. Converting all rotation into quaternions
        input_smpl: nx69 or nx135
        target_quat: desired orientation, example [w,x,y,z]
        """

        trans = input_smpl[:, :3]
        rots = input_smpl[:, 3:]  ##axis angle or rotation 6d

        if rots.reshape(trans.shape[0], -1).shape[-1] == 66:
            joint_quat_rotations = geometry.axis_angle_to_quaternion(
                rots.reshape(-1, 3)
            ).reshape(trans.shape[0], 22, 4)

        if rots.reshape(trans.shape[0], -1).shape[-1] == 132:
            joint_quat_rotations = geometry.matrix_to_quaternion(
                geometry.rotation_6d_to_matrix(rots.reshape(-1, 22, 6))
            ).reshape(trans.shape[0], 22, 4)

        root_quat_rotations = joint_quat_rotations[:, 0]
        offset_quat_rotation = geometry.quaternion_multiply(
            torch.Tensor(target_quat),
            geometry.quaternion_invert(root_quat_rotations[0]),
        )
        rotated_all_root_quat = geometry.quaternion_multiply(
            offset_quat_rotation, root_quat_rotations
        )

        rots[:, :3] = geometry.quaternion_to_axis_angle(rotated_all_root_quat)
        new_6d_rotations = geometry.matrix_to_rotation_6d(
            geometry.axis_angle_to_matrix(rots.reshape(-1, 3))
        ).reshape(rots.shape[0], -1)

        def rotate_trans(translation):
            r = R.from_euler("XYZ", [0, 180, 0], degrees=True)
            return translation[:, [2, 1, 0]][:, [1, 0, 2]] @ np.round(r.as_matrix()).T

        trans = rotate_trans(trans)
        final_rotated_smpl = torch.cat([trans, new_6d_rotations], 1)

        return final_rotated_smpl

    def rotate_smpl_by(self, input_smpl_6d: torch.Tensor, y_rot: int) -> torch.Tensor:
        input_smpl = torch.Tensor(input_smpl_6d)
        r = R.from_euler("XYZ", [0, y_rot, 0], degrees=True)
        r_matrix = torch.Tensor(r.as_matrix())
        r_quat = torch.Tensor(r.as_quat()[[3, 0, 1, 2]])

        trans = input_smpl[:, :3]

        root_orient = input_smpl[:, 3:9]
        root_orient_quat = geometry.matrix_to_quaternion(
            geometry.rotation_6d_to_matrix(root_orient)
        ).reshape(-1, 4)
        rotated_all_root_quat = geometry.quaternion_multiply(r_quat, root_orient_quat)

        input_smpl[:, 3:9] = geometry.matrix_to_rotation_6d(
            geometry.quaternion_to_matrix(rotated_all_root_quat)
        )

        input_smpl[:, :3] = trans @ (r_matrix.T.to(input_smpl_6d.device))

        return input_smpl

    def clean_smpl(self, smpl: torch.Tensor):
        pass

    def foot_detect(
        self, positions: torch.Tensor, thres: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        velfactor = torch.Tensor([thres, thres]).to(self.device)

        feet_l_x = (positions[1:, fid_l, 0] - positions[:-1, fid_l, 0]) ** 2
        feet_l_y = (positions[1:, fid_l, 1] - positions[:-1, fid_l, 1]) ** 2
        feet_l_z = (positions[1:, fid_l, 2] - positions[:-1, fid_l, 2]) ** 2
        feet_l = ((feet_l_x + feet_l_y + feet_l_z) < velfactor).float()

        feet_r_x = (positions[1:, fid_r, 0] - positions[:-1, fid_r, 0]) ** 2
        feet_r_y = (positions[1:, fid_r, 1] - positions[:-1, fid_r, 1]) ** 2
        feet_r_z = (positions[1:, fid_r, 2] - positions[:-1, fid_r, 2]) ** 2
        feet_r = (((feet_r_x + feet_r_y + feet_r_z) < velfactor)).float()
        return feet_l, feet_r

    def get_271_motion_representation(
        self, motion: torch.Tensor, angle: int = None
    ) -> torch.Tensor:
        """
        motion: nx135
        angle: degrees to rotate by along y axis
        """
        seq_len = motion.shape[0]
        if angle:
            motion = self.rotate_smpl_by(motion, angle)
        smpl_object = self.aist_6d_to_smpl(motion)
        joint_positions = smpl_object["smpl"][:, :22]
        feet_l, feet_r = self.foot_detect(joint_positions, 0.002)
        feet = torch.cat([feet_l, feet_r], 1)
        velocity = joint_positions[1:] - joint_positions[:-1]
        motion_combo = torch.cat(
            [
                motion[:-1],
                joint_positions[:-1].reshape(seq_len - 1, -1),
                velocity.reshape(seq_len - 1, -1),
                feet.reshape(seq_len - 1, -1),
            ],
            1,
        )
        return motion_combo

    def amass_to_smpl(self, file_path, target_fps, angle=0):
        raw_amass_motion = np.load(file_path, allow_pickle=True)
        fps = 0
        try:
            if "GRAB" in file_path or "SOMA" in file_path:
                fps = raw_amass_motion["mocap_frame_rate"]
            else:
                fps = raw_amass_motion["mocap_framerate"]
            frame_number = raw_amass_motion["trans"].shape[0]
        except:
            return None

        down_sample = max(int(fps / target_fps), 1)
        frame_number = raw_amass_motion["trans"].shape[0]
        pose_seq = []
        for fId in range(0, frame_number, down_sample):
            axis_angle_rots = torch.Tensor(
                raw_amass_motion["poses"][fId : fId + 1, :66]
            )
            trans = torch.Tensor(raw_amass_motion["trans"][fId : fId + 1])
            motion_tensor = torch.cat((trans, axis_angle_rots), 1)
            pose_seq.append(motion_tensor)
        amass_smpl = torch.cat(pose_seq, dim=0).detach().cpu()  ## axis angle

        amass_rotated_smpl_6d = self.rotate_smpl_to(amass_smpl.clone(), [1, 0, 0, 0])
        amass_centered_smpl_6d = self.process_and_center_aist(amass_rotated_smpl_6d)
        processed_motion = self.get_271_motion_representation(
            amass_centered_smpl_6d, angle
        )

        return processed_motion, fps

    def aist6d_to_hml(self, motion6d):
        smpl_keypoints = self.aist_6d_to_smpl(motion6d)["smpl"].cpu().numpy()
        data, ground_positions, positions, l_velocity = process_file(
            smpl_keypoints[:, : self.joints_num], 0.002
        )

        return data
