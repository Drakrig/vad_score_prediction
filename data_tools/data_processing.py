import torch    
import torchaudio
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from configs.configuration_classes import TrainConfig
from transformers import AutoProcessor, AutoConfig

def create_dataset(dataframe, config: TrainConfig):
    """Create a dataset from a dataframe.
    :param dataframe: DataFrame containing audio file paths and labels.
    :type dataframe: pd.DataFrame
    :param config: Configuration object containing model parameters.
    :type config: TrainConfig
    :return: An instance of EmotionAudioDataset or its subclass based on the model version.
    :rtype: EmotionAudioDataset or its subclass..
    """
    if config["model_version"] == "v1" or  config["model_version"] == "v1s":
        return EmotionAudioDatasetForVADv1(dataframe, config)
    elif config["model_version"] == "v2" or config["model_version"] == "v3" or config["model_version"] == "v3s":
        return EmotionAudioDatasetForVADv2(dataframe, config)
    elif config["model_version"] == "v4":
        return EmotionAudioDatasetForVADv4(dataframe, config)
    else:
        assert config["model_version"] == "decoder", f"Unsupported model version: {config['model_version']}"
        return EmotionAudioDatasetForDecoder(dataframe, config)

def pad_to_max_length(tensors):
    """Pad a list of tensors to the maximum length of the last dimension.
    :param tensors: List of tensors to pad.
    :type tensors: list of torch.Tensor
    :return: A single tensor with all input tensors padded to the maximum length.
    :rtype: torch.Tensor
    """
    max_length = max(tensor.shape[-1] for tensor in tensors)
    # Padding each tensor to the maximum length of last dimension
    padded = []
    for tensor in tensors:
        if tensor.shape[-1] < max_length:
            padding = (0, max_length - tensor.shape[-1])
            padded_tensor = torch.nn.functional.pad(tensor, padding, mode='constant', value=0)
        else:
            padded_tensor = tensor
        padded.append(padded_tensor)
    return torch.concat(padded, dim=0)

# Create dataloader
def create_dataloader(dataset, config: TrainConfig):
    """Create a DataLoader for the dataset.
    :param dataset: The dataset to create a DataLoader for.
    :type dataset: EmotionAudioDataset
    :param config: Configuration object containing DataLoader parameters.
    :type config: TrainConfig
    :return: DataLoader instance.
    :rtype: DataLoader
    """
    return DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=config["shuffle"],
        num_workers=config["num_workers"],
        pin_memory=config["pin_memory"],
        collate_fn=dataset.collate_fn
    )

class EmotionAudioDataset(Dataset):
    """Dataset class template."""
    def __init__(self, dataframe, config: TrainConfig):
        self.dataframe = dataframe.reset_index(drop=True)
        self.config = config
        self.resampler = {}
    
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        "Must be implemented in subclasses."
        pass
    
    @staticmethod
    def collate_fn(batch):
        "Must be implemented in subclasses."
        pass

class EmotionAudioDatasetForVADv1(EmotionAudioDataset):
    """Dataset class to work with V1 version of the VAD scoring model."""
    def __init__(self, dataframe, config: TrainConfig):
        super().__init__(dataframe, config)
        self.resampler[self.config["mel_encoder_sr"]] = {
            self.config["sv_model_sr"]: torchaudio.transforms.Resample(orig_freq=self.config["mel_encoder_sr"], new_freq=self.config["sv_model_sr"])
            }
        self._hann_window = {}
            
    def __getitem__(self, idx):
        """Get item from the dataset.
        :param idx: Index of the item to get.
        :type idx: int
        :return: A tuple containing the spectrogram, audio tensor, and VAD means and stds.
        :rtype: tuple
        """
        row = self.dataframe.iloc[idx]
        audio_path = row["full_path"]
        pleasure_mean = row["pleasure_mean"]
        pleasure_std = row["pleasure_std"]
        arousal_mean = row["arousal_mean"]
        arousal_std = row["arousal_std"]
        dominance_mean = row["dominance_mean"]
        dominance_std = row["dominance_std"]

        # Create label tensor
        vad_means = torch.tensor([pleasure_mean, arousal_mean, dominance_mean ], dtype=torch.float32)
        vad_stds = torch.tensor([pleasure_std, arousal_std, dominance_std], dtype=torch.float32)
        # Load and resample audio
        spec, audiop_tensor = self.get_features(audio_path)
        return spec, audiop_tensor, vad_means, vad_stds
    
    @staticmethod
    def collate_fn(batch):
        """Custom collate function to handle variable-length sequences.
        :param batch: List of tuples containing spectrogram, audio tensor, and VAD means and stds.
        :type batch: list
        :return: A tuple containing the batched spectrograms, audio tensors, and VAD means and stds.
        :rtype: tuple
        """
        specs, audio_tensors, vad_means, vad_stds = zip(*batch)
        specs = pad_to_max_length(specs)
        audio_tensors = pad_to_max_length(audio_tensors)
        vad_means = torch.stack(vad_means)
        vad_stds = torch.stack(vad_stds)
        return specs, audio_tensors, vad_means, vad_stds
    
    def spectrogram_torch(self, y, n_fft, sampling_rate, hop_size, win_size, center=False):
        """ Compute the spectrogram of an audio signal using PyTorch.

        :param y: Audio signal tensor.
        :type y: torch.Tensor
        :param n_fft: Number of FFT points.
        :type n_fft: int
        :param sampling_rate: Sampling rate of the audio signal.
        :type sampling_rate: int
        :param hop_size: Hop size for the STFT.
        :type hop_size: int
        :param win_size: Window size for the STFT.
        :type win_size: int
        :param center: Whether to pad the signal on both sides so that the output length matches the input length, defaults to False
        :type center: bool, optional
        :return: Spectrogram tensor.
        :rtype: torch.Tensor
        """
        if torch.min(y) < -1.2:
            print("min value is ", torch.min(y))
        if torch.max(y) > 1.2:
            print("max value is ", torch.max(y))

        dtype_device = str(y.dtype) + "_" + str(y.device)
        # wnsize_dtype_device = str(win_size) + '_' + dtype_device
        key = "%s-%s-%s-%s-%s" % (dtype_device, n_fft, sampling_rate, hop_size, win_size)
        # if wnsize_dtype_device not in hann_window:
        if key not in self._hann_window:
            self._hann_window[key] = torch.hann_window(win_size).to(dtype=y.dtype, device=y.device)

        y = torch.nn.functional.pad(
            y.unsqueeze(1), (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)), mode="reflect"
        )
        y = y.squeeze(1)
        spec = torch.stft(
            y,
            n_fft,
            hop_length=hop_size,
            win_length=win_size,
            window=self._hann_window[key],
            center=center,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=False,
        )

        spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-8)
        return spec
    
    def get_features(self, audio_path):
        """Extract spectrogram and audio tensor from an audio file.
        :param audio_path: Path to the audio file.
        :type audio_path: str
        :return: A tuple containing the spectrogram and audio tensor.
        :rtype: tuple
        """
        waveform, original_sr = torchaudio.load(audio_path)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if original_sr not in self.resampler:
            self.resampler[original_sr] = {
                    self.config["mel_encoder_sr"]: torchaudio.transforms.Resample(orig_freq=original_sr, new_freq=self.config["mel_encoder_sr"])
                }
        resampled_waveform = self.resampler[original_sr][self.config["mel_encoder_sr"]](waveform)
        maxx = resampled_waveform.abs().max()
        if maxx>1:
            resampled_waveform /= min(2,maxx)
        spec = self.spectrogram_torch(
            resampled_waveform,
            self.config["filter_length"],
            self.config["mel_encoder_sr"],
            self.config["hop_length"],
            self.config["win_length"],
            center=False,
        )
        audio_tensor = self.resampler[self.config["mel_encoder_sr"]][self.config["sv_model_sr"]](resampled_waveform)
        return spec, audio_tensor
    
    
class EmotionAudioDatasetForVADv2(EmotionAudioDataset):
    """Dataset class to work with V2 an V3 versions of the VAD scoring model."""
    def __init__(self, dataframe, config: TrainConfig):
        """Initialize the dataset with a dataframe and configuration.

        :param dataframe: DataFrame containing audio file paths and VAD statistics.
        :type dataframe: pd.DataFrame
        :param config: Configuration object containing model parameters.
        :type config: TrainConfig
        """
        super().__init__(dataframe, config)
        self.processor = AutoProcessor.from_pretrained(config["model_id"])
        # Load model config
        model_config = AutoConfig.from_pretrained(config["model_id"])
        self.decoder_start_token_id = model_config.decoder_start_token_id
    
    def __getitem__(self, idx):
        """Get item from the dataset.
        :param idx: Index of the item to get.
        :type idx: int
        :return: A tuple containing the spectrogram, audio tensor, and VAD means and stds.
        :rtype: tuple
        """
        row = self.dataframe.iloc[idx]
        audio_path = row["full_path"]
        pleasure_mean = row["pleasure_mean"]
        pleasure_std = row["pleasure_std"]
        arousal_mean = row["arousal_mean"]
        arousal_std = row["arousal_std"]
        dominance_mean = row["dominance_mean"]
        dominance_std = row["dominance_std"]

        # Create label tensor
        vad_means = torch.tensor([pleasure_mean, arousal_mean, dominance_mean ], dtype=torch.float32)
        vad_stds = torch.tensor([pleasure_std, arousal_std, dominance_std], dtype=torch.float32)
        # Extract features
        input_features, decoder_input_ids = self.get_features(audio_path)
        return input_features, decoder_input_ids, vad_means, vad_stds
    
    @staticmethod
    def collate_fn(batch):
        """
        Custom collate function to handle variable-length sequences.
        Args:
            batch (list): List of tuples containing spectrogram, audio tensor, and VAD means and stds.
        Returns:
            tuple: A tuple containing the batched spectrograms, audio tensors, and VAD means and stds.
        """
        
        input_features, decoder_input_ids, vad_means, vad_stds = zip(*batch)
        input_features = pad_to_max_length(input_features)
        decoder_input_ids = pad_to_max_length(decoder_input_ids)
        vad_means = torch.stack(vad_means)
        vad_stds = torch.stack(vad_stds)
        return input_features, decoder_input_ids, vad_means, vad_stds
    
    def get_features(self, audio_path):
        """Extract input features and decoder input IDs from an audio file.
        :param audio_path: Path to the audio file.
        :type audio_path: str
        :return: A tuple containing the input features and decoder input IDs.
        :rtype: tuple
        """
        waveform, original_sr = torchaudio.load(audio_path)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if original_sr not in self.resampler:
            self.resampler[original_sr] = {
                16000: torchaudio.transforms.Resample(orig_freq=original_sr, new_freq=16000)
            }
        resampled_waveform = self.resampler[original_sr][16000](waveform)
        inputs = self.processor(
            resampled_waveform.squeeze(0).numpy(),
            sampling_rate=16000,
            return_tensors="pt"
        )
        decoder_input_ids = torch.tensor([[1, 1]]) * self.decoder_start_token_id
        return inputs.input_features, decoder_input_ids

class EmotionAudioDatasetForVADv4(EmotionAudioDataset):
    def __init__(self, dataframe, config: TrainConfig):
        super().__init__(dataframe, config)
        self.is_half = config["is_half"]
    
    def __getitem__(self, idx):
        """
        Get item from the dataset.
        Args:
            idx (int): Index of the item to get.
        Returns:
            tuple: A tuple containing the spectrogram, audio tensor, and VAD means and stds.
        """
        row = self.dataframe.iloc[idx]
        audio_path = row["full_path"]
        pleasure_mean = row["pleasure_mean"]
        pleasure_std = row["pleasure_std"]
        arousal_mean = row["arousal_mean"]
        arousal_std = row["arousal_std"]
        dominance_mean = row["dominance_mean"]
        dominance_std = row["dominance_std"]

        # Create label tensor
        vad_means = torch.tensor([pleasure_mean, arousal_mean, dominance_mean ], dtype=torch.float32)
        vad_stds = torch.tensor([pleasure_std, arousal_std, dominance_std], dtype=torch.float32)
        # Load and resample audio
        audio_tensor = self.get_features(audio_path)
        if self.is_half:
            audio_tensor = audio_tensor.half()
            vad_means = vad_means.half()
            vad_stds = vad_stds.half()
        return audio_tensor, vad_means, vad_stds
    
    def get_features(self, audio_path):
        waveform, original_sr = torchaudio.load(audio_path)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if original_sr != self.config["sv_model_sr"]: #16000
            resampler = torchaudio.transforms.Resample(orig_freq=original_sr, new_freq=self.config["sv_model_sr"])
            resampled_waveform = resampler(waveform)
        else:
            resampled_waveform = waveform
        resampled_waveform =  F.layer_norm(resampled_waveform, resampled_waveform.shape).view(1, -1)
        return resampled_waveform
    
    @staticmethod
    def collate_fn(batch):
        """
        Custom collate function to handle variable-length sequences.
        Args:
            batch (list): List of tuples containing spectrogram, audio tensor, and VAD means and stds.
        Returns:
            tuple: A tuple containing the batched spectrograms, audio tensors, and VAD means and stds.
        """
        audio_tensors, vad_means, vad_stds = zip(*batch)
        audio_tensors = pad_to_max_length(audio_tensors)
        vad_means = torch.stack(vad_means)
        vad_stds = torch.stack(vad_stds)
        return audio_tensors, vad_means, vad_stds


