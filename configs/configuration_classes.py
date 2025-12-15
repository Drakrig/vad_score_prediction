from typing import TypedDict

class ModelConfig(TypedDict):
    device: str
    is_half: bool

class RefEncoderModelConfig(ModelConfig):
    sv_model_path: str
    sv_model_sr: int
    use_sv: bool
    projection_weights_path: str
    gin_channels: int
    mel_encoder_path: str
    mel_encoder_sr: int
    filter_length: int
    hop_length: int
    win_length: int
    is_half: bool

class RefEncoderWhisperModelConfig(ModelConfig):
    model_id: str
    device: str
    is_half: bool

class VADScoringModelConfig(ModelConfig):
    model_version: str
    loss_type: str
    projection_weights_path: str
    means_head_weights_path: str
    stds_head_weights_path: str

class TrainConfig(ModelConfig):
    batch_size: int
    num_workers: int
    pin_memory: bool
    shuffle: bool
    epochs: int
    learning_rate: float
    weight_decay: float
    data_dir: str
    annotations_file: str
    test_size: float
    limit_train_samples: int
    min_samples_threshold: int
    random_state: int
    save_dir: str
    save_interval: int
    logging_interval: int
    save_interval_steps: int
    model_version: str
    model_id: str
    loss_type: str
    kl_weight: float