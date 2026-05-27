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
        feat = self.backbone(x)   # [B, 512, 1, 1]
        feat = feat.flatten(1)    # [B, 512]
        feat = self.proj(feat)    # [B, feature_dim]
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

        # image encoder
        self.image_encoder = ImageEncoder(hidden_dim)
        for p in self.image_encoder.parameters():
            p.requires_grad = False

        # text encoder
        self.tokenizer = CLIPTokenizer.from_pretrained(
            "openai/clip-vit-base-patch32"
        )
        self.text_encoder = CLIPTextModel.from_pretrained(
            "openai/clip-vit-base-patch32"
        )
        for p in self.text_encoder.parameters():
            p.requires_grad = False

        self.text_proj = nn.Linear(512, hidden_dim)

        # state encoder
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # modality embeddings:
        # 0 = left high image
        # 1 = left wrist image
        # 2 = right wrist image
        # 3 = text
        # 4 = state
        self.modality_embed = nn.Parameter(torch.zeros(5, hidden_dim))

        self.fusion_norm = nn.LayerNorm(hidden_dim)
        self.fusion_dropout = nn.Dropout(0.1)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=8,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=4,
        )

        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, chunk_size * action_dim),
        )

    def encode_text(self, task_strings, device):
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

        high_feat = self.image_encoder(cam_high)
        left_feat = self.image_encoder(cam_left)
        right_feat = self.image_encoder(cam_right)
        text_feat = self.encode_text(task_strings, device)
        state_feat = self.state_encoder(state)

        # [B, 5, D]
        tokens = torch.stack(
            [high_feat, left_feat, right_feat, text_feat, state_feat],
            dim=1,
        )

        tokens = tokens + self.modality_embed.unsqueeze(0)
        tokens = self.fusion_norm(tokens)

        feat = self.transformer(tokens)
        pooled = feat.mean(dim=1)
        pooled = self.fusion_dropout(pooled)

        action = self.action_head(pooled)

        B = action.shape[0]
        action = action.view(B, self.chunk_size, -1)
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
        out = model(state, cam_high, cam_left, cam_right, tasks)

    print("output:", out.shape)
