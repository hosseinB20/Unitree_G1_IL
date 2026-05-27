import numpy as np
import torch
import torch.nn as nn


class BCPolicy(nn.Module):

    def __init__(self):
        super().__init__()

        self.policy = nn.Module()

        self.policy.nets = nn.Module()

        self.policy.nets.mlp = nn.Module()

        self.policy.nets.mlp._model = nn.Sequential(
            nn.Linear(16, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
        )

        self.policy.nets.decoder = nn.Module()

        self.policy.nets.decoder.nets = nn.Module()

        self.policy.nets.decoder.nets.action = nn.Linear(
            1024,
            16
        )

    def forward(self, x):

        x = self.policy.nets.mlp._model(x)

        x = self.policy.nets.decoder.nets.action(x)

        return x


class ActionProviderBC:

    def __init__(self, env, args_cli):

        self.name = "BCPolicy"

        self.env = env

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        print("\n[BC] loading checkpoint...")

        ckpt = torch.load(
            args_cli.model_path,
            map_location=self.device,
        )

        print("[BC] checkpoint keys:", ckpt.keys())

        # -------------------------------------------------
        # EXTRACT MODEL SECTION
        # -------------------------------------------------

        if "model" not in ckpt:
            raise RuntimeError(
                "[BC] ERROR: checkpoint does not contain 'model' key"
            )

        model_blob = ckpt["model"]

        print(
            "[BC] model_blob type:",
            type(model_blob)
        )

        if hasattr(model_blob, "keys"):
            print(
                "[BC] model_blob keys:",
                model_blob.keys()
            )

        # -------------------------------------------------
        # FIND ACTUAL STATE_DICT
        # -------------------------------------------------

        if isinstance(model_blob, dict):

            if "nets" in model_blob:
                state_dict = model_blob["nets"]
                print("[BC] using model_blob['nets']")

            elif "policy" in model_blob:
                state_dict = model_blob["policy"]
                print("[BC] using model_blob['policy']")

            elif "model" in model_blob:
                state_dict = model_blob["model"]
                print("[BC] using model_blob['model']")

            else:
                state_dict = model_blob
                print("[BC] using model_blob directly")

        else:
            state_dict = model_blob
            print("[BC] using raw model_blob")

        # -------------------------------------------------
        # CREATE POLICY
        # -------------------------------------------------

        self.policy = BCPolicy().to(self.device)

        print("[BC] loading state_dict...")

        missing, unexpected = self.policy.load_state_dict(
            state_dict,
            strict=True
        )

        print("[BC] missing keys:", missing)
        print("[BC] unexpected keys:", unexpected)

        self.policy.eval()

        print("[BC] policy loaded successfully")

    # =====================================================
    # REQUIRED CONTROLLER INTERFACE
    # =====================================================

    def start(self):

        print("[BC] start() called")

    def stop(self):

        print("[BC] stop() called")

    def cleanup(self):

        print("[BC] cleanup() called")

    def reset(self):

        print("[BC] reset() called")

    # =====================================================
    # POLICY ACTION
    # =====================================================

    def get_action(self, env):

        print("[BC] get_action() called")

        # -------------------------------------------------
        # GET OBS BUFFER
        # -------------------------------------------------

        obs = env.obs_buf

        print("[BC] obs_buf type:", type(obs))

        if isinstance(obs, dict):

            print("[BC] obs_buf keys:", obs.keys())

            obs = obs["policy"]

            print("[BC] policy type:", type(obs))

            if isinstance(obs, dict):

                print("[BC] policy keys:", obs.keys())

                first_key = list(obs.keys())[0]

                obs_tensor = obs[first_key]

                print("[BC] using policy key:", first_key)

            else:

                obs_tensor = obs

        else:

            obs_tensor = obs

        print("[BC] obs_tensor type:", type(obs_tensor))
        print("[BC] obs_tensor shape:", obs_tensor.shape)

        # -------------------------------------------------
        # EXTRACT FIRST ENV OBS
        # -------------------------------------------------

        state = obs_tensor[0][:16]

        print("[BC] state shape:", state.shape)
        print("[BC] state:", state)

        # -------------------------------------------------
        # POLICY INFERENCE
        # -------------------------------------------------

        state = state.float().unsqueeze(0).to(self.device)

        with torch.no_grad():
            action = self.policy(state)

        policy_action = action.squeeze(0).detach().cpu().numpy()

        print("[BC] raw policy action:", policy_action)

        # -------------------------------------------------
        # BUILD FULL ACTION VECTOR
        # -------------------------------------------------

        full_action = np.zeros(33, dtype=np.float32)
        full_action[:16] = policy_action

        print("[BC] final action shape:", full_action.shape)
        print("[BC] final action:", full_action)

        # IMPORTANT: return torch tensor, not numpy array
        action_tensor = torch.as_tensor(
            full_action,
            dtype=torch.float32,
            device=env.device if hasattr(env, "device") else self.device
        ).unsqueeze(0)   # shape: (1, 33)

        print("[BC] action_tensor shape:", action_tensor.shape)

        return action_tensor