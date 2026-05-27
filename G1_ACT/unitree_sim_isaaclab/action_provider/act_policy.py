import torch
import torch.nn as nn
import torchvision.models as models

from transformers import CLIPTextModel, CLIPTokenizer


class ImageEncoder(nn.Module):
    def __init__(self, feature_dim=512):
        super().__init__()

        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        self.backbone = nn.Sequential(*list(backbone.children())[:-1])

        self.proj = nn.Linear(512, feature_dim)

    def forward(self, x):
        """
        x: [B, 3, 224, 224]
        """

        feat = self.backbone(x)          # [B, 512, 1, 1]
        feat = feat.flatten(1)           # [B, 512]
        feat = self.proj(feat)           # [B, feature_dim]

        return feat


class ACTPolicy(nn.Module):
    def __init__(
        self,
        state_dim=16,
        action_dim=16,
        chunk_size=20,
        hidden_dim=512,
    ):
        super().__init__()

        self.chunk_size = chunk_size
        self.hidden_dim = hidden_dim

        # ---------------------------------------------------
        # image encoder
        # ---------------------------------------------------

        self.image_encoder = ImageEncoder(hidden_dim)

        # freeze image encoder
        for p in self.image_encoder.parameters():
            p.requires_grad = False

        # ---------------------------------------------------
        # text encoder (CLIP)
        # ---------------------------------------------------

        self.tokenizer = CLIPTokenizer.from_pretrained(
            "openai/clip-vit-base-patch32"
        )

        self.text_encoder = CLIPTextModel.from_pretrained(
            "openai/clip-vit-base-patch32"
        )

        # freeze text encoder
        for p in self.text_encoder.parameters():
            p.requires_grad = False

        self.text_proj = nn.Linear(512, hidden_dim)

        # ---------------------------------------------------
        # state encoder
        # ---------------------------------------------------

        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # ---------------------------------------------------
        # transformer
        # ---------------------------------------------------

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=8,
            batch_first=True,
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=4,
        )

        # ---------------------------------------------------
        # action head
        # ---------------------------------------------------

        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, chunk_size * action_dim),
        )

    def encode_text(self, task_strings, device):
        """
        task_strings: list[str]
        """

        tokens = self.tokenizer(
            task_strings,
            padding=True,
            return_tensors="pt",
        )

        tokens = {k: v.to(device) for k, v in tokens.items()}

        with torch.no_grad():
            out = self.text_encoder(**tokens)

        text_feat = out.pooler_output
        text_feat = self.text_proj(text_feat)

        return text_feat

    def forward(
        self,
        state,
        cam_high,
        cam_left,
        cam_right,
        task_strings,
    ):
        """
        state:     [B, state_dim]
        cam_*:     [B, 3, 224, 224]
        task:      list[str]
        """

        device = state.device

        # ---------------------------------------------------
        # image features
        # ---------------------------------------------------

        high_feat = self.image_encoder(cam_high)
        left_feat = self.image_encoder(cam_left)
        right_feat = self.image_encoder(cam_right)

        vision_feat = (
            high_feat +
            left_feat +
            right_feat
        ) / 3.0

        # ---------------------------------------------------
        # text features
        # ---------------------------------------------------

        text_feat = self.encode_text(task_strings, device)

        # ---------------------------------------------------
        # state features
        # ---------------------------------------------------

        state_feat = self.state_encoder(state)

        # ---------------------------------------------------
        # combine
        # ---------------------------------------------------

        fused = (
            vision_feat +
            text_feat +
            state_feat
        )

        # transformer expects sequence
        fused = fused.unsqueeze(1)   # [B, 1, D]

        feat = self.transformer(fused)

        feat = feat[:, 0]

        # ---------------------------------------------------
        # predict action chunk
        # ---------------------------------------------------

        action = self.action_head(feat)

        B = action.shape[0]

        action = action.view(
            B,
            self.chunk_size,
            -1,
        )

        return action


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = ACTPolicy().to(device)

    B = 2

    state = torch.randn(B, 16).to(device)

    cam_high = torch.randn(B, 3, 224, 224).to(device)
    cam_left = torch.randn(B, 3, 224, 224).to(device)
    cam_right = torch.randn(B, 3, 224, 224).to(device)

    tasks = [
        "Pick up the red cup on the table.",
        "Place the cylinder into the basket.",
    ]

    with torch.no_grad():
        out = model(
            state,
            cam_high,
            cam_left,
            cam_right,
            tasks,
        )

    print("output:", out.shape)
