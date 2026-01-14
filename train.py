import yaml
from pathlib import Path
import pandas as pd
from datetime import datetime
from shutil import copy
from tqdm import tqdm
import re

from sklearn.model_selection import train_test_split
from modules.vad_scoring import (
    VADScoringModel, 
    VADScoringModelV2, 
    VADScoringModelV3, 
    VADScoringModelTwoHeadedV2,
    VADScoringModelV4)
from data_tools.data_processing import create_dataset, create_dataloader

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import accelerate

def print_config(config: dict):
    """Print the training configuration."""
    print("Training Configuration:")
    for key, value in config.items():
        print(f"{key}: {value}")

def load_config(config_path: str) -> dict:
    """Load configuration from a YAML file."""
    with open(config_path, 'r') as file:
        config_dict = yaml.safe_load(file)
    return config_dict

def save_checkpoint(model, save_path: str, epoch: int, step: int, config: dict): 
    save_path = Path(config["save_dir"]) / f"vad_model_epoch_{epoch+1}_step_{step}.pth"
    if config["model_version"] == "v1":
        torch.save(model.classification_head.state_dict(), save_path)
    elif config["model_version"] == "v1s":
        torch.save({
            "positional_embedding_state_dict": model.positional_embedding.state_dict(),
            "means_head_state_dict": model.means_head.state_dict(),
            "stds_head_state_dict": model.stds_head.state_dict(),
            "projection_state_dict": model.projection.state_dict(),
            "transformer_blocks_state_dict": model.transformer_blocks.state_dict()
        }, save_path)
    elif config["model_version"] == "v2":
        torch.save({
            "head_state_dict": model.head.state_dict(),
            "projection_state_dict": model.projection.state_dict()
        }, save_path)
    elif config["model_version"] == "v3" or config["model_version"] == "v3s":
        torch.save({
            "means_head_state_dict": model.means_head.state_dict(),
            "stds_head_state_dict": model.stds_head.state_dict(),
            "projection_state_dict": model.projection.state_dict()
        }, save_path)
    elif config["model_version"] == "v4":
        torch.save({
            # Common
            "means_head_state_dict": model.means_head.state_dict(),
            "stds_head_state_dict": model.stds_head.state_dict(),
            # V2
            "query_state_dict": model.learnable_query.state_dict(),
            "attention_state_dict": model.attn.state_dict(),
            "norm_state_dict": model.norm.state_dict(),
        }, save_path)
    else:
        torch.save({
            "classification_head_state_dict": model.classification_head.state_dict(),
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

def kl_impact(step: int, config: dict) -> float:
    """
    Calculate the KL divergence weight based on the current step.
    The weight increases linearly from 0 to config["kl_weight"] over the first config["kl_warmup_steps"] steps.
    After that, it remains constant.
    """
    cycle_pos = step % (config["kl_cycle_length"]+1)
    if cycle_pos == config["kl_cycle_length"]:
        return config["kl_max_weight"]
    else:
        return config["kl_min_weight"] + (config["kl_max_weight"] - config["kl_min_weight"]) * (cycle_pos / config["kl_cycle_length"])


def train_model(model, dataloader: DataLoader, val_dataloader: DataLoader, device: str, config: dict, writer: SummaryWriter):
    """Train the VAD scoring model."""
    criterion = nn.MSELoss()
    metric = nn.PairwiseDistance(p=2)  # L2 distance for evaluation
    # For V1 model we train the classification head only
    if config["model_version"] == "v1":
        optimizer = optim.AdamW(model.classification_head.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    else:
        decay = []
        no_decay = []
        # Projection is common for v1s, v2 and v3 models
        if not re.search(r'v4', config["model_version"]) is not None:
            try:
                for name, param in model.projection.named_parameters():
                    decay.append(param)
            except:
                pass
        # For V1s model we train the postional embedding, transformer blocks, projection layer and means and stds heads
        if re.search(r'v1', config["model_version"]) is not None:
            for name, param in model.positional_embedding.named_parameters():
                decay.append(param)
            for name, param in model.means_head.named_parameters():
                decay.append(param)
            for name, param in model.stds_head.named_parameters():
                decay.append(param)
            # V1s with attention
            for name, param in model.transformer_blocks.named_parameters():
                decay.append(param)
        elif re.search(r'v2', config["model_version"]) is not None:
            # For V2 model we train the projection layer and means head except PReLU alpha
            for name, param in model.head.named_parameters():
                if 'prelu' in name:  # exclude PReLU alpha from weight decay
                    no_decay.append(param)
                else:
                    decay.append(param)
        elif re.search(r'v3', config["model_version"]) is not None:
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
        elif re.search(r'v4', config["model_version"]) is not None:
            # For v4 model we train the projection layer, stds and means heads
            for name, param in model.means_head.named_parameters():
                decay.append(param)
            for name, param in model.stds_head.named_parameters():
                decay.append(param)
            # V4 with attention
            for name, param in model.learnable_query.named_parameters():
                decay.append(param)
            for name, param in model.attn.named_parameters():
                decay.append(param)
            # v4 with norm layer
            for name, param in model.norm.named_parameters():
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
    
    accelerator = accelerate.Accelerator(gradient_accumulation_steps=config["accamulation_steps"])
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
    model = model.to(device)

    step = 0
    steps_per_epoch = len(dataloader)
    for epoch in range(config["epochs"]):
        total_loss = 0.0
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{config['epochs']}"):
            with accelerator.accumulate(model):
                if re.search(r'v1', config["model_version"]) is not None:
                    specs, audio_tensors, vad_means, vad_stds = batch
                    specs, audio_tensors = specs.to(device), audio_tensors.to(device)
                if re.search(r'v4', config["model_version"]) is not None:
                    audio_tensors, vad_means, vad_stds = batch
                    audio_tensors = audio_tensors.to(device)
                else:
                    input_features, decoder_input_ids, vad_means, vad_stds = batch
                vad_means = vad_means.to(device)
                vad_stds = vad_stds.to(device) if config["loss_type"] == "kl" else None
                optimizer.zero_grad()

                if re.search(r'v1s', config["model_version"]) is not None:
                    predicted_means, predicted_stds = model(specs, audio_tensors, normalize=config["normalize_embeddings"])
                elif re.search(r'v2', config["model_version"]) is not None:
                    predicted_means = model(input_features, decoder_input_ids)
                elif re.search(r'v3', config["model_version"]) is not None:
                    predicted_means, predicted_stds = model(input_features, decoder_input_ids)
                elif re.search(r'v4', config["model_version"]) is not None:
                    predicted_means, predicted_stds = model(audio_tensors)
                else: # v1
                    predicted_means = model(specs, audio_tensors)

                if config["loss_type"] == "mse":
                    loss = criterion(predicted_means, vad_means)
                else:
                    # KL Divergence loss
                    loss = config["mean_loss_weight"] * criterion(predicted_means, vad_means) + kl_impact(step, config) * kl_divergence_gaussians(predicted_means, predicted_stds, vad_means, vad_stds)
                    
                accelerator.backward(loss)
                optimizer.step()
                scheduler.step()

                total_loss += loss.item()
                step += 1
                # print loss every n steps
                if accelerator.is_main_process and (step % config["logging_interval"] == 0):
                    current_epoch_step = (step - 1) % steps_per_epoch + 1
                    print(f"Step [{current_epoch_step}], Loss: {(total_loss / current_epoch_step):.4f}")
                    writer.add_scalar("Loss/train", total_loss / current_epoch_step, step)
        if accelerator.is_main_process:
            total_loss /= len(dataloader)
            print(f"Epoch [{epoch+1}/{config['epochs']}], Loss: {total_loss}")
            # Log training loss
            writer.add_scalar("Loss/epoch/train", total_loss, epoch)
        # Calculate validation loss
        val_loss = 0.0
        mean_mse_impact = 0.0
        mean_kl_impact = 0.0
        pairwise_value = 0.0
        with torch.no_grad():
            for val_batch in val_dataloader:
                if config["model_version"] == "v1" or config["model_version"] == "v1s":
                    val_specs, val_audio_tensors, val_vad_means, val_vad_stds = val_batch
                    val_specs, val_audio_tensors = val_specs.to(device), val_audio_tensors.to(device)
                elif config["model_version"] == "v4":
                    val_audio_tensors, val_vad_means, val_vad_stds = val_batch
                    val_audio_tensors = val_audio_tensors.to(device)
                else:
                    val_input_features, val_decoder_input_ids, val_vad_means, val_vad_stds = val_batch
                    val_input_features = val_input_features.to(device)
                    val_decoder_input_ids = val_decoder_input_ids.to(device)
                val_vad_means = val_vad_means.to(device)
                val_vad_stds = val_vad_stds.to(device) if config["loss_type"] == "kl" else None

                if re.search(r'v1s', config["model_version"]) is not None:
                    val_means_predicted, val_stds_predicted = model(val_specs, val_audio_tensors, normalize=config["normalize_embeddings"])
                elif re.search(r'v2', config["model_version"]) is not None:
                    val_means_predicted = model(val_input_features, val_decoder_input_ids)
                elif re.search(r'v3', config["model_version"]) is not None:
                    val_means_predicted, val_stds_predicted = model(val_input_features, val_decoder_input_ids)
                elif re.search(r'v4', config["model_version"]) is not None:
                    val_means_predicted, val_stds_predicted = model(val_audio_tensors)
                else: # v1
                    val_means_predicted = model(val_specs, val_audio_tensors)

                if config["loss_type"] == "mse":
                    loss = criterion(val_means_predicted, val_vad_means)
                else:
                    # KL Divergence loss
                    mse_loss = criterion(val_means_predicted, val_vad_means)
                    pairwise_value += metric(val_means_predicted, val_vad_means).mean().item()
                    mean_mse_impact += mse_loss.item()
                    kl_loss = kl_divergence_gaussians(val_means_predicted, val_stds_predicted, val_vad_means, val_vad_stds)
                    mean_kl_impact += kl_loss.item()
                    loss = config["mean_loss_weight"] * mse_loss + kl_loss * config["kl_max_weight"]
                val_loss += loss.item()
        if accelerator.is_main_process:
            val_loss /= len(val_dataloader)
            mean_mse_impact /= len(val_dataloader)
            mean_kl_impact /= len(val_dataloader)
            pairwise_value /= len(val_dataloader)
            writer.add_scalar("Loss/epoch/val", val_loss, epoch)
            writer.add_scalar("Metrics/mean_mse_impact", mean_mse_impact, epoch)
            writer.add_scalar("Metrics/mean_kl_impact", mean_kl_impact, epoch)
            writer.add_scalar("Metrics/pairwise_distance", pairwise_value, epoch)
            print(f"Validation Loss after epoch {epoch+1}: {val_loss}")
        # Save head checkpoint 
        if accelerator.is_main_process and (epoch + 1) % config["save_interval"] == 0:
            # Save model checkpoint
            save_path = Path(config["save_dir"]) / f"vad_model_epoch_{epoch+1}_{step}.pth"
            save_checkpoint(accelerator.unwrap_model(model), save_path, epoch, step, config)
    
def main():
    # Load configuration
    config_path = "configs/vad_train_config.yaml"
    config = load_config(config_path)
    print_config(config)

    task_name = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_scorer_{config['model_version']}_training"
    config["save_dir"] = Path(config["save_dir"]) / task_name
    config["save_dir"].mkdir(parents=True, exist_ok=True)
    (config["save_dir"] / "configs").mkdir(parents=True, exist_ok=True)
    copy(config_path, config["save_dir"] / config_path)

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
    if config["min_samples_threshold"] > 0:
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
    if re.search(r'v4', config["model_version"]) is not None:
        print("Using VADScoringModelV4")
        cfg = yaml.safe_load(open(config["emo2vec_config"], "r"))
        model = VADScoringModelV4(
            cfg, 
            config["emo2vec_model_weights"], 
            device, 
            is_half=config["is_half"]
            )
    elif re.search(r'v3', config["model_version"]) is not None:
        print("Using VADScoringModelV3")
        model = VADScoringModelV3(device=device, out_channels=config["out_channels"], is_half=config["is_half"], model_id=config["model_id"], version=config["model_version"])
    elif re.search(r'v2', config["model_version"]) is not None:
        print("Using VADScoringModelV2")
        model = VADScoringModelV2(device, is_half=config["is_half"], model_id=config["model_id"])
    elif re.search(r'v1s', config["model_version"]) is not None:
        print("Using Two headed VADScoringModel")
        model = VADScoringModelTwoHeadedV2(
            sv_path=config["sv_model_path"],
            mel_encoder_path=config["mel_encoder_path"],
            projection_weights_path=config["projection_weights_path"],
            prelu_weights=config["prelu_weights_path"],
            device=device,
            is_half=config["is_half"],
            gin_channels=config["gin_channels"],
            n_layers=config["n_layers"],
            n_heads=config["n_heads"],
            out_channels=config["out_channels"]
        ).to(device)
    else:
        print("Using VADScoringModel")
        model = VADScoringModel(
            sv_path=config["sv_model_path"],
            mel_encoder_path=config["mel_encoder_path"],
            projection_weights_path=config["projection_weights_path"],
            device=device,
            is_half=config["is_half"],
            gin_channels=config["gin_channels"],
            
        ).to(device)

    if config["is_half"]:
        model = model.half()
    # Train the model
    train_model(model, train_dataloader, val_dataloader, device, config, writer)
    writer.close()

if __name__ == "__main__":
    main()
