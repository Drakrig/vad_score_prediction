import torch
from torch import nn
from modules.melencoder import MelStyleEncoder
from modules.sv import SV
from transformers import AutoModelForSpeechSeq2Seq
from modules.emo2vec.model import Emotion2vec

class RefEncoder(nn.Module):
    """Reference Encoder that combines a speaker verification model and a mel-style encoder."""
    def __init__(
        self,
        sv_path,
        mel_encoder_path,
        projection_weights,
        prelu_weights,
        device,
        is_half=True,
        gin_channels=1024,
    ):
        super(RefEncoder, self).__init__()
        self.sv_model = SV(sv_path, device, is_half)
        self.mel_encoder = MelStyleEncoder(704, style_vector_dim=gin_channels).to(device)
        self.mel_encoder.load_state_dict(torch.load(mel_encoder_path))
        self.mel_encoder.eval()
        self.linear_projection = nn.Linear(20480, gin_channels).to(device)
        self.linear_projection.load_state_dict(torch.load(projection_weights))
        if is_half:
            self.linear_projection.half()
        self.linear_projection.eval()
        self.prelu = nn.PReLU(num_parameters=gin_channels).to(device)
        self.prelu.load_state_dict(
            torch.load(prelu_weights)
        )
        self.device = device
    
    def forward(self, spec:torch.LongTensor, sv_ref:torch.Tensor, use_sv:bool=True):
        with torch.no_grad():
            spec = spec.to(self.device)
            sv_ref = sv_ref.to(self.device)
            refer_lengths = torch.LongTensor([spec.size(2)]).to(self.device)
            refer_mask = torch.unsqueeze(
                self.sequence_mask(refer_lengths, spec.size(2)), 1
            ).to(spec.dtype)
            embed = self.mel_encoder(spec[:, :704] * refer_mask.repeat(spec.size(0),1,1), refer_mask.repeat(spec.size(0),1,1))
            if use_sv:
                sv_embed = self.sv_model.compute_embedding3(sv_ref.to(self.sv_model.device))
                sv_embed = self.linear_projection(sv_embed)
                embed += sv_embed.unsqueeze(-1)
                embed = self.prelu(embed)
        return embed.squeeze(-1)

    @staticmethod
    def sequence_mask(length, max_length=None):
        if max_length is None:
            max_length = length.max()
        x = torch.arange(max_length, dtype=length.dtype, device=length.device)
        return x.unsqueeze(0) < length.unsqueeze(1)

class RefEncoderWhisper(nn.Module):
    """Reference Encoder based on Whisper model for emotion recognition."""
    def __init__(
        self,
        device,
        is_half=True,
        model_id='mkrausio/EmoWhisper-AnS-Small-v0.1',
    ):
        super(RefEncoderWhisper, self).__init__()
        self.device = device
        self.is_half = is_half
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id,
            low_cpu_mem_usage=True,
            use_safetensors=True
        )
        if is_half:
            self.model.half()
        self.model.eval()
    
    def forward(self, input_features:torch.Tensor, decoder_input_ids:torch.Tensor):
        if self.is_half:
            input_features = input_features.half().to(self.device)
            decoder_input_ids = decoder_input_ids.to(self.device)
        
        with torch.no_grad():
            last_hidden_state = self.model(input_features, decoder_input_ids=decoder_input_ids)
        
        return last_hidden_state

class RefEncoderEmo2Vec(nn.Module):
    def __init__(
        self,
        cfg: dict,
        weights_path: str,
        device,
        is_half=True,
    ):
        super(RefEncoderEmo2Vec, self).__init__()
        self.device = device
        self.is_half = is_half
        self.model = Emotion2vec(**cfg)
        self.model.load_weights(weights_path)
        if is_half:
            self.model.half()
        self.model.eval()
    
    def forward(self, input_features:torch.Tensor):
        if self.is_half:
            input_features = input_features.half().to(self.device)
        else:
            input_features = input_features.to(self.device)
        
        with torch.no_grad():
            embed, features = self.model.inference(input_features)
        return embed, features
        