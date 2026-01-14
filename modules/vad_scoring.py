import numbers
import torch
import torch.nn.init as init
from torch import nn
from torch import Size, Tensor
from typing import Union, List, Optional, Tuple
from modules.ref_encoder import RefEncoder, RefEncoderWhisper, RefEncoderEmo2Vec

# v1s components
class AdaNorm(nn.Module):
    def __init__(self, normalized_shape: Union[int, List[int], Size], k: float = 0.1, eps: float = 1e-5, bias: bool = False) -> None:
        super(AdaNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.k = k
        self.eps = eps
        self.weight = nn.Parameter(torch.empty(self.normalized_shape))
        if bias:
            self.bias = nn.Parameter(torch.empty(self.normalized_shape))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.ones_(self.weight)
        if self.bias is not None:
            init.zeros_(self.bias)

    def forward(self, input: Tensor) -> Tensor:
        mean = torch.mean(input, dim=-1, keepdim=True)
        var = (input - mean).pow(2).mean(dim=-1, keepdim=True) + self.eps
    
        input_norm = (input - mean) * torch.rsqrt(var)
        
        adanorm = self.weight * (1 - self.k * input_norm) * input_norm

        if self.bias is not None:
            adanorm = adanorm + self.bias
    
        return adanorm

class TransformerBlock(nn.Module):
    def __init__(self, input_dim:int, output_dim:int, n_heads:int, d_ff:int, dropout:float=0.1):
        super(TransformerBlock, self).__init__()
        self.adanorm1 = AdaNorm(input_dim)
        self.attn = nn.MultiheadAttention(input_dim, n_heads, dropout=dropout)
        self.ff = nn.Sequential(
            nn.Linear(input_dim, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, output_dim)
        )
        # Upsample the feature map
        self.adanorm2 = AdaNorm(input_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # Apply AdaNorm and Multihead Attention
        x_norm = self.adanorm1(x)
        x_attn = self.attn(x_norm, x_norm, x_norm)[0]
        # Residual connection after attention
        x_attn = x + self.dropout(self.adanorm2(x_attn))  # Residual with AdaNorm2
        # Feedforward layer after attention and upsampling
        x_ff = self.ff(x_attn)
        x = x + self.dropout(x_ff)  # Residual connection after feedforward layer

        return x

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model):
        super(PositionalEmbedding, self).__init__()
        self.embedding = nn.Parameter(torch.randn(1, 1024, d_model))
    def forward(self, x):
        return x + self.embedding

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

class VADScoringModelTwoHeadedV2(nn.Module):
    """Model to predict VAD (varience,arousal,dominance) scores using a reference encoder."""
    def __init__(self, 
                 sv_path,
                 mel_encoder_path, 
                 projection_weights_path, 
                 prelu_weights, 
                 device,
                 n_layers=4,
                 n_heads=8, 
                 is_half=True,
                 out_channels=1, 
                 gin_channels=1024):
        super(VADScoringModelTwoHeadedV2, self).__init__()
        self.ref_encoder = RefEncoder(sv_path, mel_encoder_path, projection_weights_path, prelu_weights, device, is_half, gin_channels)
        # Freeze encoder parameters
        for param in self.ref_encoder.parameters():
            param.requires_grad = False
        self.positional_embedding = PositionalEmbedding(gin_channels).to(device)
        # Define the transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(gin_channels, gin_channels, n_heads=n_heads, d_ff=2048, dropout=0.1) for _ in range(n_layers)
        ]).to(device)
        # Projection
        self.projection = nn.Conv1d(gin_channels, out_channels, kernel_size=1).to(device)
        # Define the means and stds heads
        self.means_head = nn.Sequential(
            nn.Linear(out_channels * gin_channels, gin_channels),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(gin_channels, 3),
            nn.Tanh()  # Assuming scores are in the range [-1, 1]
        ).to(device)
        self.stds_head = nn.Sequential(
            nn.Linear(out_channels * gin_channels, gin_channels),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(gin_channels, 3),
            nn.Softplus()  # Must be positive
        ).to(device)
        if is_half:
            self.positional_embedding.half()
            for block in self.transformer_blocks:
                block.half()
            self.means_head.half()
            self.stds_head.half()
        self.means_head.to(device)
        self.stds_head.to(device)
        self.device = device

    @torch.no_grad()
    def encode(self, spec: torch.Tensor, sv_ref: torch.Tensor, use_sv: bool = True):
        """Encode the spectrogram and speaker vector reference to get the embedding."""
        ref_embedding = self.ref_encoder(spec, sv_ref, use_sv)
        return ref_embedding

    def forward(self, spec: torch.Tensor, sv_ref: torch.Tensor, use_sv: bool = True, normalize=True):
        """Forward pass to compute VAD scores."""
        ref_embedding = self.encode(spec, sv_ref, use_sv)
        if normalize:
            norm_factor = torch.stack([ref_embedding.max(dim=1).values, ref_embedding.min(dim=1).values.abs()]).T.max(dim=1).values.unsqueeze(0).T
            ref_embedding = ref_embedding / norm_factor
        ref_embedding = self.positional_embedding(ref_embedding.unsqueeze(-1))
        for block in self.transformer_blocks:
            ref_embedding = block(ref_embedding)
        ref_embedding = self.projection(ref_embedding).flatten(start_dim=1)
        means = self.means_head(ref_embedding)
        stds = self.stds_head(ref_embedding)
        return means, stds
    
    def load_weights(self, weights_path: str):
        """Load weights from a file."""
        states_dict = torch.load(weights_path, map_location="cpu")
        self.positional_embedding.load_state_dict(states_dict["positional_embedding_state_dict"])
        self.projection.load_state_dict(states_dict["projection_state_dict"])
        self.transformer_blocks.load_state_dict(states_dict["transformer_blocks_state_dict"])
        self.means_head.load_state_dict(states_dict["means_head_state_dict"])
        self.stds_head.load_state_dict(states_dict["stds_head_state_dict"])
        del states_dict["means_head_state_dict"]
        del states_dict["stds_head_state_dict"]
        del states_dict["positional_embedding_state_dict"]
        del states_dict["projection_state_dict"]
        del states_dict["transformer_blocks_state_dict"]

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
    VADScoringModel
    def load_weights(self, weights_path: str):
        """Load weights from a file."""
        states_dict = torch.load(weights_path, map_location="cpu")
        self.head.load_state_dict(states_dict["head_state_dict"])
        self.projection.load_state_dict(states_dict["projection_state_dict"])
        del states_dict["head_state_dict"]
        del states_dict["projection_state_dict"]

class VADScoringModelV3(nn.Module):
    """Model to predict statistical characteristics of VAD (varience,arousal,dominance) scores using a reference encoder based on BUD-E-Whisper."""
    def __init__(self, out_channels, device, is_half=True, version="v3s", model_id='mkrausio/EmoWhisper-AnS-Small-v0.1'):
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
            nn.PReLU() if version == "v3" else nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(512, 3),
            nn.Tanh()
        )
        self.stds_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1500 * out_channels, 512),
            nn.PReLU() if version == "v3" else nn.GELU(),
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

class LearnableQuery(nn.Module):
    def __init__(self, embed_dim):
        super(LearnableQuery, self).__init__()
        self.query = nn.Parameter(torch.randn(1, 1, embed_dim))
    
    def forward(self, batch_size):
        return self.query.expand(batch_size, -1, -1)

class VADScoringModelV4(nn.Module):
    """Based on emotion2vec endoer."""
    def __init__(self, cfg: dict, encoder_weights_path: str, device, n_heads=8, dropout=0.0, is_half=True, version="v4"):
        super(VADScoringModelV4, self).__init__()
        self.ref_encoder = RefEncoderEmo2Vec(cfg, encoder_weights_path, device, is_half)
        # Freeze the reference encoder
        for param in self.ref_encoder.parameters():
            param.requires_grad = False

        self.learnable_query = LearnableQuery(cfg.get("model_conf").get("embed_dim")).to(device)

        self.attn = nn.MultiheadAttention(cfg.get("model_conf").get("embed_dim"), num_heads=n_heads, dropout=dropout, batch_first=True)

        self.norm = nn.LayerNorm(cfg.get("model_conf").get("embed_dim")).to(device)

        self.means_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(cfg.get("model_conf").get("embed_dim"), 512),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(512, 3),
            nn.Tanh()
        )
        self.stds_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(cfg.get("model_conf").get("embed_dim"), 512),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(512, 3),
            nn.Softplus()  # Ensure stds are positive
        )
        self.to(device)
        if is_half:
            self.means_head.half()
            self.stds_head.half()

    @torch.no_grad()
    def encode(self, audio: torch.Tensor)->tuple[torch.Tensor, list[torch.Tensor,...]]:
        """Encode the audio to get the embedding."""
        embed, features = self.ref_encoder(audio)
        return embed, features

    def forward(self, audio: torch.Tensor):
        """Forward pass to compute VAD scores."""
        embed, _ = self.encode(audio)
        if embed.ndim == 2:
            embed = embed.unsqueeze(0)
        # V2 scheme with attention
        batch_size = embed.size(0)
        query = self.learnable_query(batch_size)
        normed_embed = self.norm(embed)
        embed, _ = self.attn(query, normed_embed, normed_embed)
        embed = embed.squeeze(1)
        means = self.means_head(embed)
        stds = self.stds_head(embed)
        return means, stds