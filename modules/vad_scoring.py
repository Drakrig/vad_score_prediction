import torch
from torch import nn
from modules.ref_encoder import RefEncoder, RefEncoderWhisper

class VADScoringModel(nn.Module):
    """Model to predict VAD (varience,arousal,dominance) scores using a reference encoder based on Eres2NetV2 and MelEncoder fron TransferTTS."""
    def __init__(self, sv_path, mel_encoder_path, projection_weights_path, device, is_half=True, gin_channels=1024):
        super(VADScoringModel, self).__init__()
        self.ref_encoder = RefEncoder(sv_path, mel_encoder_path, projection_weights_path, device, is_half, gin_channels)
        # Freeze encoder parameters
        for param in self.ref_encoder.parameters():
            param.requires_grad = False
        self.classification_head = nn.Sequential(
            nn.Linear(gin_channels, 512),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(512, 3),
            nn.Tanh()  # Assuming scores are in the range [-1, 1]
        ).to(device)
        self.classification_head.to(device)
        self.device = device

    def forward(self, spec: torch.Tensor, sv_ref: torch.Tensor, use_sv: bool = True):
        """Forward pass to compute VAD scores."""
        ref_embedding = self.ref_encoder(spec, sv_ref, use_sv)
        vad_scores = self.classification_head(ref_embedding.squeeze(-1))
        return vad_scores

class VADScoringModelV2(nn.Module):
    """Model to predict VAD (varience,arousal,dominance) scores using a reference encoder based on BUD-E-Whisper."""
    def __init__(self, device, is_half=True, model_id='mkrausio/EmoWhisper-AnS-Small-v0.1'):
        super(VADScoringModelV2, self).__init__()
        self.ref_encoder = RefEncoderWhisper(device, is_half, model_id)
        # Freeze the reference encoder
        for param in self.ref_encoder.parameters():
            param.requires_grad = False
        # Last hidden state shape is (batch_size, 1500, 768)
        self.projection = nn.Conv1d(
            in_channels=768,
            out_channels=256,
            kernel_size=1
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1500 * 256, 512),
            nn.PReLU(),
            nn.Dropout(0.25),
            nn.Linear(512, 3),
            nn.Tanh()
        )
        self.to(device)
        if is_half:
            self.projection.half()
            self.head.half()

    def forward(self, input_features:torch.Tensor, decoder_input_ids:torch.Tensor):
        output = self.ref_encoder(input_features, decoder_input_ids)
        # embeddings shape is (batch_size, 1500, 768)
        embeddings = output.encoder_last_hidden_state.transpose(1, 2)
        # Now shape is (batch_size, 768, 1500)
        # Apply the projection
        projected = self.projection(embeddings)
        scores = self.head(projected)
        return scores
    
    def load_weights(self, weights_path: str):
        """Load weights from a file."""
        states_dict = torch.load(weights_path, map_location="cpu")
        self.head.load_state_dict(states_dict["head_state_dict"])
        self.projection.load_state_dict(states_dict["projection_state_dict"])
        del states_dict["head_state_dict"]
        del states_dict["projection_state_dict"]

class VADScoringModelV3(nn.Module):
    """Model to predict statistical characteristics of VAD (varience,arousal,dominance) scores using a reference encoder based on BUD-E-Whisper."""
    def __init__(self, out_channels, device, is_half=True, model_id='mkrausio/EmoWhisper-AnS-Small-v0.1'):
        super(VADScoringModelV3, self).__init__()
        self.ref_encoder = RefEncoderWhisper(device, is_half, model_id)
        # Freeze the reference encoder
        for param in self.ref_encoder.parameters():
            param.requires_grad = False
        # Last hidden state shape is (batch_size, 1500, 768)
        self.projection = nn.Conv1d(
            in_channels=768,
            out_channels=out_channels,
            kernel_size=1
        )
        self.means_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1500 * out_channels, 512),
            nn.PReLU(),
            nn.Dropout(0.25),
            nn.Linear(512, 3),
            nn.Tanh()
        )
        self.stds_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1500 * out_channels, 512),
            nn.PReLU(),
            nn.Dropout(0.25),
            nn.Linear(512, 3),
            nn.Softplus()  # Ensure stds are positive
        )
        self.to(device)
        if is_half:
            self.projection.half()
            self.means_head.half()
            self.stds_head.half()

    def forward(self, input_features:torch.Tensor, decoder_input_ids:torch.Tensor):
        output = self.ref_encoder(input_features, decoder_input_ids)
        # embeddings shape is (batch_size, 1500, 768)
        embeddings = output.encoder_last_hidden_state.transpose(1, 2)
        # Now shape is (batch_size, 768, 1500)
        # Apply the projection
        projected = self.projection(embeddings)
        means = self.means_head(projected)
        stds = self.stds_head(projected)
        return means, stds
    
    def load_weights(self, weights_path: str):
        """Load weights from a file."""
        states_dict = torch.load(weights_path, map_location="cpu")
        self.means_head.load_state_dict(states_dict["means_head_state_dict"])
        self.stds_head.load_state_dict(states_dict["stds_head_state_dict"])
        self.projection.load_state_dict(states_dict["projection_state_dict"])
        del states_dict["means_head_state_dict"]
        del states_dict["stds_head_state_dict"]
        del states_dict["projection_state_dict"]