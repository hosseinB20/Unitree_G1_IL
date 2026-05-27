import os
import sys
import numpy as np
import torch
import cv2

from action_provider.action_base import ActionProvider

# ACT repo
sys.path.append(os.path.expanduser("~/g1_act_task_conditioned"))
from act_policy import ACTPolicy


class ActionProviderACT(ActionProvider):

    def __init__(self, env, args_cli):
        super().__init__("ACT")

        self.env = env
        self.args_cli = args_cli
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print("\n[ACT] loading checkpoint...")

        ckpt_path = os.path.expanduser(args_cli.model_path)
        ckpt = torch.load(ckpt_path, map_location=self.device)

        self.policy = ACTPolicy(
            state_dim=16,
            action_dim=16,
            chunk_size=20,
            hidden_dim=512,
        )

        if isinstance(ckpt, dict) and "model" in ckpt:
            state_dict = ckpt["model"]
        else:
            state_dict = ckpt

        self.policy.load_state_dict(state_dict, strict=False)
        self.policy.to(self.device)
        self.policy.eval()

        print("[ACT] policy loaded successfully")

        self.dataset_joint_order = [
            "left_shoulder_pitch_joint",
            "left_shoulder_roll_joint",
            "left_shoulder_yaw_joint",
            "left_elbow_joint",
            "left_wrist_roll_joint",
            "left_wrist_pitch_joint",
            "left_wrist_yaw_joint",
            "right_shoulder_pitch_joint",
            "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint",
            "right_elbow_joint",
            "right_wrist_roll_joint",
            "right_wrist_pitch_joint",
            "right_wrist_yaw_joint",
            "left_hand_Joint1_1",
            "right_hand_Joint1_1",
        ]

        self.robot_joint_names = list(self.env.scene["robot"].data.joint_names)

        print("\n[ACT] robot joint names:")
        for i, n in enumerate(self.robot_joint_names):
            print(f"{i:02d}: {n}")

        self.dataset_to_robot_indices = []

        print("\n[ACT] FINAL JOINT MAPPING:\n")
        for dataset_name in self.dataset_joint_order:
            if dataset_name not in self.robot_joint_names:
                raise RuntimeError(f"Could not map dataset joint {dataset_name}")

            idx = self.robot_joint_names.index(dataset_name)
            self.dataset_to_robot_indices.append(idx)
            print(f"{dataset_name:35s} -> idx {idx}")

        self.prev_action = None

    def start(self):
        print("[ACT] start() called")

    def stop(self):
        print("[ACT] stop() called")

    def cleanup(self):
        print("[ACT] cleanup() called")

    def _get_camera_image(self, cam_name):
        cam = self.env.scene[cam_name].data.output["rgb"][0]
        return cam.detach().cpu().numpy()

    def _prep_image(self, img):
        img = cv2.resize(img, (224, 224))
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
        return torch.from_numpy(img).float()

    def get_action(self, env):
        robot = self.env.scene["robot"]

        # current simulator joint state in simulator order
        full_joint_state = (
            robot.data.joint_pos[0]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )

        # reorder into dataset order
        state16 = np.array(
            [full_joint_state[idx] for idx in self.dataset_to_robot_indices],
            dtype=np.float32,
        )

        print("\n[ACT] dataset ordered state:")
        print(state16)

        # three camera views expected by ACT
        img_high = self._prep_image(self._get_camera_image("front_camera"))
        img_left = self._prep_image(self._get_camera_image("left_wrist_camera"))
        img_right = self._prep_image(self._get_camera_image("right_wrist_camera"))

        cam_high = img_high.unsqueeze(0).to(self.device)
        cam_left = img_left.unsqueeze(0).to(self.device)
        cam_right = img_right.unsqueeze(0).to(self.device)

        state_tensor = torch.from_numpy(state16).float().unsqueeze(0).to(self.device)

        # use the exact text the model was trained on
        task_strings = ["Pick up the red cup on the table."]

        with torch.no_grad():
            output = self.policy(
                state_tensor,
                cam_high,
                cam_left,
                cam_right,
                task_strings,
            )

        if isinstance(output, tuple):
            output = output[0]

        # if ACT returns a chunk, take the first action in the chunk
        if len(output.shape) == 3:
            output = output[:, 0, :]

        action16 = output.squeeze(0).detach().cpu().numpy()

        print("\n[ACT] raw action16:")
        print(action16)

        # expand back to full robot DOF
        full_action = full_joint_state.copy()
        for i, idx in enumerate(self.dataset_to_robot_indices):
            full_action[idx] = action16[i]

        print("\n[ACT] mapped full action:")
        print(full_action)

        return torch.from_numpy(full_action).float().unsqueeze(0)