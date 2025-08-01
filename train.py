import yaml
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from modules.vad_scoring import VADScoringModel, VADScoringModelV2, VADScoringModelV3
from data_tools.data_processing import create_dataset, create_dataloader
from configs.configuration_classes import TrainConfig
from tqdm import tqdm

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import accelerate

def print_config(config: TrainConfig):
    """Print the training configuration."""
    print("Training Configuration:")
    for key, value in config.items():
        print(f"{key}: {value}")

def load_config(config_path: str) -> TrainConfig:
    """Load configuration from a YAML file."""
    with open(config_path, 'r') as file:
        config_dict = yaml.safe_load(file)
    return TrainConfig(**config_dict)

def save_checkpoint(model, save_path: str, epoch: int, step: int, config: TrainConfig): 
    save_path = Path(config["save_dir"]) / f"vad_model_epoch_{epoch+1}_step_{step}.pth"
    if config["model_version"] == "v1":
        torch.save(model.classification_head.state_dict(), save_path)
    elif config["model_version"] == "v2":
        torch.save({
            "head_state_dict": model.head.state_dict(),
            "projection_state_dict": model.projection.state_dict()
        }, save_path)
    else:
        torch.save({
            "means_head_state_dict": model.means_head.state_dict(),
            "stds_head_state_dict": model.stds_head.state_dict(),
            "projection_state_dict": model.projection.state_dict()
        }, save_path)
    print(f"Model saved to {save_path}")

def kl_divergence_gaussians(mu_pred, std_pred, mu_prior, std_prior):
    """
    mu_pred, std_pred: tensors of shape (batch_size, 3)
    mu_prior, std_prior: tensors of shape (3,) or broadcastable
    """
    # Ensure numerical stability
    eps = 1e-6
    std_pred = std_pred.clamp(min=eps, max=1.0)

    var_pred = std_pred ** 2
    var_prior = std_prior ** 2

    kl = (
        torch.log(std_prior / std_pred)
        + (var_pred + (mu_pred - mu_prior) ** 2) / (2 * var_prior)
        - 0.5
    )

    return kl.mean()  # average over batch and targets


def train_model(model, dataloader: DataLoader, val_dataloader: DataLoader, device: str, config: TrainConfig, writer: SummaryWriter):
    """Train the VAD scoring model."""
    criterion = nn.MSELoss()
    # For V1 model we train the classification head only
    if config["model_version"] == "v1":
        optimizer = optim.AdamW(model.classification_head.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    else:
        # For V2 and V3 model we train the classification head and projection layer
        decay = []
        no_decay = []
        # Projection is fully trainable
        for name, param in model.projection.named_parameters():
                decay.append(param)
        if config["model_version"] == "v2":
            # PReLU is recommended not to be train with weight decay
            for name, param in model.head.named_parameters():
                if 'prelu' in name:  # exclude PReLU alpha from weight decay
                    no_decay.append(param)
                else:
                    decay.append(param)
        else:
            # In V3 we have means and stds heads
            for name, param in model.means_head.named_parameters():
                if 'prelu' in name:  # exclude PReLU alpha from weight decay
                    no_decay.append(param)
                else:
                    decay.append(param)
            for name, param in model.stds_head.named_parameters():
                if 'prelu' in name:
                    no_decay.append(param)
                else:
                    decay.append(param)
        
        optimizer = optim.AdamW(
            [
                {'params': decay, 'weight_decay': config["weight_decay"]},
                {'params': no_decay, 'weight_decay': 0.0}
            ],
            lr=config["learning_rate"]
        )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"] * len(dataloader))
    stats = {
        "train_loss": [],
        "val_loss": []
    }
    # Accelerate setup for distributed training
    for param in model.ref_encoder.parameters():
        param.requires_grad = False
    
    accelerator = accelerate.Accelerator()
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
    model = model.to(device)
    # Freeze the reference encoder parameters
    
    steps_per_epoch = len(dataloader)
    for epoch in range(config["epochs"]):
        total_loss = 0.0
        model.train()
        step = 0
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{config['epochs']}"):
            if config["model_version"] == "v1":
                specs, audio_tensors, vad_means, vad_stds = batch
                specs, audio_tensors = specs.to(device), audio_tensors.to(device)
            else:
                input_features, decoder_input_ids, vad_means, vad_stds = batch
                #audio_tensors = audio_tensors.to(device)
            vad_means = vad_means.to(device)
            optimizer.zero_grad()
            if config["model_version"] == "v1":
                predicted_means = model(specs, audio_tensors)
            elif config["model_version"] == "v2":
                predicted_means = model(input_features, decoder_input_ids)
            else:
                predicted_means, predicted_stds = model(input_features, decoder_input_ids)
            if config["loss_type"] == "mse":
                loss = criterion(predicted_means, vad_means)
            else:
                # KL Divergence loss
                loss = criterion(predicted_means, vad_means) + kl_divergence_gaussians(predicted_means, predicted_stds, vad_means, vad_stds) * config["kl_weight"]
                
            accelerator.backward(loss)
            #loss.backward()
            optimizer.step()

            total_loss += loss.item()
            step += 1
            # print loss every n steps
            if accelerator.is_main_process and (step % config["logging_interval"] == 0):
                print(f"Step [{len(stats['train_loss'])}], Loss: {loss.item()}")
                writer.add_scalar("Loss/train", loss.item(), epoch * step + step)
            if accelerator.is_main_process and (step % config["save_interval_steps"] == 0):
                # Save model checkpoint
                save_path = Path(config["save_dir"]) / f"vad_model_step_{epoch+1}_{step}.pth"
                save_checkpoint(accelerator.unwrap_model(model), save_path, epoch, step, config)

        scheduler.step()
        print(f"Epoch [{epoch+1}/{config['epochs']}], Loss: {total_loss/len(dataloader)}")
        # Log training loss
        writer.add_scalar("Loss/train_epoch", total_loss / len(dataloader), epoch)
        stats["train_loss"].append(total_loss / len(dataloader))
        # Calculate validation loss
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for val_batch in val_dataloader:
                if config["model_version"] == "v1":
                    val_specs, val_audio_tensors, val_vad_means, _ = val_batch
                    val_specs, val_audio_tensors = val_specs.to(device), val_audio_tensors.to(device)
                else:
                    val_input_features, val_decoder_input_ids, val_vad_means, val_vad_stds = val_batch
                    #val_specs = val_input_features.to(device)
                    #val_audio_tensors = val_decoder_input_ids.to(device)
                val_vad_means = val_vad_means.to(device)
                val_vad_stds = val_vad_stds.to(device) if config["loss_type"] == "kl" else None
                if config["model_version"] == "v1":
                    val_means_predicted = model(val_specs, val_audio_tensors)
                elif config["model_version"] == "v2":
                    val_means_predicted = model(val_input_features, val_decoder_input_ids)
                else:
                    val_means_predicted, val_stds_predicted = model(val_input_features, val_decoder_input_ids)
                if config["loss_type"] == "mse":
                    loss = criterion(val_means_predicted, val_vad_means)
                else:
                    # KL Divergence loss
                    loss = criterion(val_means_predicted, val_vad_means) + kl_divergence_gaussians(val_means_predicted, val_stds_predicted, val_vad_means, val_vad_stds) * config["kl_weight"]
                val_loss += loss.item()
        val_loss /= len(val_dataloader)
        writer.add_scalar("Loss/val_epoch", val_loss, epoch)
        stats["val_loss"].append(val_loss)
        print(f"Validation Loss after epoch {epoch+1}: {val_loss}")
        # Save head checkpoint 
        if accelerator.is_main_process and (epoch + 1) % config["save_interval"] == 0:
            # Save model checkpoint
            save_path = Path(config["save_dir"]) / f"vad_model_epoch_{epoch+1}_{step}.pth"
            save_checkpoint(accelerator.unwrap_model(model), save_path, epoch, step, config)
    
    #Save results
    if accelerator.is_main_process:
        # Save model checkpoint
        save_path = Path(config["save_dir"]) / "vad_model_final.pth"
        save_checkpoint(accelerator.unwrap_model(model), save_path, config["epochs"], step, config)

def main():
    # Load configuration
    config_path = "configs/vad_train_config.yaml"
    config = load_config(config_path)
    print_config(config)

    # TensorBoard setup
    log_dir = Path(config["save_dir"]) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir, comment=f"{config['model_version']}_lr_{config['learning_rate']}_bs_{config['batch_size']}_loss_{config['loss_type']}]")

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset
    data_dir = Path(config["data_dir"])
    csv_file_path = data_dir / config["annotations_file"]
    df = pd.read_csv(csv_file_path, sep=";")
    # Drop classes with less than min_samples_threshold
    df = df[df['verified_emotion'].map(df['verified_emotion'].value_counts()) >= config["min_samples_threshold"]]
    # Find the smallest emotion class
    max_samples_per_emotion = df['verified_emotion'].value_counts().sort_values(ascending=True)[0]
    # Limit each emotion class to max_samples_per_emotion
    df = df.groupby('verified_emotion').apply(lambda x: x.sample(n=max_samples_per_emotion, random_state=config["random_state"])).reset_index(drop=True)

    if config["limit_train_samples"] > 0:
        # Limit the number of samples in the training set
        df = df.groupby('verified_emotion').apply(lambda x: x.sample(n=config["limit_train_samples"], random_state=config["random_state"])).reset_index(drop=True)
        print(f"Limited dataset to {config['limit_train_samples']} samples per emotion class.")

    # Split dataset into training and validation sets
    # Ensure even distribution of emotions in both sets
    train_df, val_df = train_test_split(df, test_size=config["test_size"], random_state=config["random_state"], stratify=df['verified_emotion'])
    print(f"Training samples: {len(train_df)}, Validation samples: {len(val_df)}")
    print(f"Training emotions distribution:\n{train_df['verified_emotion'].value_counts()}")
    print(f"Validation emotions distribution:\n{val_df['verified_emotion'].value_counts()}")
    
    train_dataset = create_dataset(train_df, config)
    val_dataset = create_dataset(val_df, config)

    train_dataloader = create_dataloader(train_dataset, config)
    val_dataloader = create_dataloader(val_dataset, config)

    # Initialize model
    if config["model_version"] == "v3":
        print("Using VADScoringModelV3")
        model = VADScoringModelV3(device, is_half=config["is_half"], model_id=config["model_id"])
    elif config["model_version"] == "v2":
        print("Using VADScoringModelV2")
        model = VADScoringModelV2(device, is_half=config["is_half"], model_id=config["model_id"])
    else:
        print("Using VADScoringModel")
        model = VADScoringModel(
            sv_path=config["sv_model_path"],
            mel_encoder_path=config["mel_encoder_path"],
            projection_weights_path=config["projection_weights_path"],
            device=device,
            is_half=config["is_half"],
            gin_channels=config["gin_channels"]
        ).to(device)

    # Train the model
    train_model(model, train_dataloader, val_dataloader, device, config, writer)

if __name__ == "__main__":
    main()