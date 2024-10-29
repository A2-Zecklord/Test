import torch
import torchaudio
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from models import DiffusionModel
import argparse
import os
from pathlib import Path
from torch.nn import functional as F
from tqdm import tqdm
import numpy as np
import wandb  # Optional: for logging

class MusicDataset(Dataset):
    def __init__(self, wav_folder_path, csv_path, max_audio_length=220500):
        self.wav_folder_path = Path(wav_folder_path)
        self.max_audio_length = max_audio_length
        
        self.lyrics_df = pd.read_csv(csv_path)
        self.lyrics_df['lyrics'] = self.lyrics_df['lyrics'].astype(str)
        self.lyrics_df['prompt'] = self.lyrics_df['prompt'].astype(str)
        
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=22050,
            n_fft=1024,
            hop_length=256,
            n_mels=80
        )
        
        self.expected_mel_length = (self.max_audio_length // 256) + 1
        
        self.valid_files = []
        for idx, row in self.lyrics_df.iterrows():
            audio_path = self.wav_folder_path / row['audio_file']
            if audio_path.exists():
                self.valid_files.append(idx)
            else:
                print(f"Warning: File not found: {audio_path}")
        
        print(f"Found {len(self.valid_files)} valid audio files out of {len(self.lyrics_df)} entries")
        print(f"Expected mel spectrogram length: {self.expected_mel_length}")

    def pad_or_truncate(self, waveform):
        if waveform.size(1) > self.max_audio_length:
            waveform = waveform[:, :self.max_audio_length]
        elif waveform.size(1) < self.max_audio_length:
            pad_length = self.max_audio_length - waveform.size(1)
            waveform = torch.nn.functional.pad(waveform, (0, pad_length))
        return waveform
    
    def __len__(self):
        return len(self.valid_files)
    
    def __getitem__(self, idx):
        real_idx = self.valid_files[idx]
        row = self.lyrics_df.iloc[real_idx]
        
        audio_path = str(self.wav_folder_path / row['audio_file'])
        waveform, sample_rate = torchaudio.load(audio_path)
        
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        if sample_rate != 22050:
            resampler = torchaudio.transforms.Resample(sample_rate, 22050)
            waveform = resampler(waveform)
        
        waveform = self.pad_or_truncate(waveform)
        
        # Generate mel spectrogram and ensure correct dimensions
        mel_spec = self.mel_transform(waveform)  # [1, n_mels, time]
        mel_spec = mel_spec.squeeze(0)  # [n_mels, time]
        mel_spec = mel_spec.unsqueeze(0)  # Add channel dim [1, n_mels, time]
        
        # Rest of the code remains the same
        lyrics = str(row['lyrics'])
        prompt = str(row['prompt'])
        
        lyrics_tokens = self.tokenizer(
            lyrics,
            padding='max_length',
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )
        
        prompt_tokens = self.tokenizer(
            prompt,
            padding='max_length',
            truncation=True,
            max_length=128,
            return_tensors='pt'
        )
        
        lyrics_tokens = {k: v.squeeze(0) for k, v in lyrics_tokens.items()}
        prompt_tokens = {k: v.squeeze(0) for k, v in prompt_tokens.items()}
        
        return {
            'audio': waveform,
            'mel_spec': mel_spec,  # Shape: [1, n_mels, time]
            'lyrics_tokens': lyrics_tokens,
            'prompt_tokens': prompt_tokens
        }

def custom_collate(batch):
    audio = torch.stack([item['audio'] for item in batch])
    mel_spec = torch.stack([item['mel_spec'] for item in batch])  # Will be [batch, 1, n_mels, time]
    
    lyrics_tokens = {
        key: torch.stack([item['lyrics_tokens'][key] for item in batch])
        for key in batch[0]['lyrics_tokens']
    }
    
    prompt_tokens = {
        key: torch.stack([item['prompt_tokens'][key] for item in batch])
        for key in batch[0]['prompt_tokens']
    }
    
    return {
        'audio': audio,
        'mel_spec': mel_spec,
        'lyrics_tokens': lyrics_tokens,
        'prompt_tokens': prompt_tokens
    }

class DiffusionTrainer:
    def __init__(self, time_steps=1000, beta_start=1e-4, beta_end=0.02):
        self.time_steps = time_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        
        # Create noise schedule
        self.betas = torch.linspace(beta_start, beta_end, time_steps)
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        
    def get_noise_schedule(self, t):
        return self.alphas_cumprod[t]
    
    def add_noise(self, x, t):
        noise = torch.randn_like(x)
        return (
            self.sqrt_alphas_cumprod[t][:, None, None] * x + 
            self.sqrt_one_minus_alphas_cumprod[t][:, None, None] * noise
        ), noise

def train(args):
    # Setup logging
    if args.use_wandb:
        wandb.init(project="music-diffusion", config=args)
    
    # Force CUDA device
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This model requires a GPU to train.")
    
    device = torch.device('cuda')
    torch.backends.cudnn.benchmark = True
    
    # Print GPU info
    gpu_name = torch.cuda.get_device_name(0)
    print(f"\nUsing GPU: {gpu_name}")
    print(f"GPU Memory allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
    print(f"GPU Memory cached: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB\n")
    
    # Initialize dataset and dataloader
    dataset = MusicDataset(args.wav_folder, args.csv_file)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=custom_collate,
        pin_memory=True
    )
    
    # Initialize model and diffusion
    model = DiffusionModel(
        hidden_dim=args.hidden_dim,
        mel_channels=80,
        time_steps=args.time_steps
    ).to(device)
    
    diffusion = DiffusionTrainer(
        time_steps=args.time_steps,
        beta_start=args.beta_start,
        beta_end=args.beta_end
    )
    
    # Move diffusion schedules to GPU
    diffusion.betas = diffusion.betas.to(device)
    diffusion.alphas = diffusion.alphas.to(device)
    diffusion.alphas_cumprod = diffusion.alphas_cumprod.to(device)
    diffusion.sqrt_alphas_cumprod = diffusion.sqrt_alphas_cumprod.to(device)
    diffusion.sqrt_one_minus_alphas_cumprod = diffusion.sqrt_one_minus_alphas_cumprod.to(device)
    
    # Initialize optimizer and scaler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    scaler = torch.cuda.amp.GradScaler()
    
    # Create checkpoint directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Training info
    total_steps = len(dataloader) * args.epochs
    print(f"\nStarting training for {args.epochs} epochs")
    print(f"Total steps: {total_steps}")
    print(f"Steps per epoch: {len(dataloader)}")
    print(f"Batch size: {args.batch_size}")
    print(f"Training on {len(dataset)} samples\n")
    
    # Training loop
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        print(f"Epoch {epoch + 1}/{args.epochs}")
        print("-" * 50)
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}")
        for batch_idx, batch in enumerate(progress_bar):
            current_step = epoch * len(dataloader) + batch_idx + 1
            
            optimizer.zero_grad()
            
            # Prepare input
            mel_spec = batch['mel_spec'].to(device)
            
            # Debug print
            if batch_idx == 0:
                print(f"Mel spec shape before model: {mel_spec.shape}")
                print(f"Mel spec min: {mel_spec.min()}, max: {mel_spec.max()}")

            lyrics_tokens = {k: v.to(device) for k, v in batch['lyrics_tokens'].items()}
            prompt_tokens = {k: v.to(device) for k, v in batch['prompt_tokens'].items()}
            
            # Sample random timesteps
            t = torch.randint(0, diffusion.time_steps, (mel_spec.shape[0],), device=device)
            
            try:
                # Use automatic mixed precision
                with torch.cuda.amp.autocast():
                    # Add noise to mel spectrograms
                    noisy_mel, noise = diffusion.add_noise(mel_spec, t)
                    print(f"Noisy mel shape: {noisy_mel.shape}")  # Debug print
                    
                    # Predict noise
                    predicted_noise = model(noisy_mel, t, lyrics_tokens, prompt_tokens)
                    
                    # Calculate loss
                    loss = F.mse_loss(predicted_noise, noise)
                    
                # Scale loss and backpropagate
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
                total_loss += loss.item()
                
                # Update progress bar
                progress_bar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'avg_loss': f"{total_loss / (batch_idx + 1):.4f}",
                    'gpu_mem': f"{torch.cuda.memory_allocated() / 1024**2:.1f}MB"
                })
                
                # Log to wandb
                if args.use_wandb:
                    wandb.log({
                        'loss': loss.item(),
                        'step': current_step,
                        'epoch': epoch
                    })
                
            except RuntimeError as e:
                print(f"Error in batch {batch_idx}: {str(e)}")
                torch.cuda.empty_cache()
                continue
        
        # End of epoch
        avg_loss = total_loss / len(dataloader)
        print(f'\nEpoch {epoch + 1}/{args.epochs} completed. Average loss: {avg_loss:.4f}')
        
        # Print GPU memory
        print(f"GPU Memory allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
        print(f"GPU Memory cached: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")
        
        # Save checkpoint
        if (epoch + 1) % args.checkpoint_interval == 0:
            checkpoint_path = os.path.join(args.checkpoint_dir, f'checkpoint_epoch_{epoch + 1}.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'diffusion_params': {
                    'time_steps': diffusion.time_steps,
                    'beta_start': diffusion.beta_start,
                    'beta_end': diffusion.beta_end
                }
            }, checkpoint_path)
            print(f'Checkpoint saved: {checkpoint_path}\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # Data parameters
    parser.add_argument('--wav_folder', type=str, required=True,
                        help='Path to folder containing WAV files')
    parser.add_argument('--csv_file', type=str, required=True,
                        help='Path to CSV file containing lyrics and prompts')
    
    # Model parameters
    parser.add_argument('--hidden_dim', type=int, default=512,
                        help='Hidden dimension size')
    parser.add_argument('--time_steps', type=int, default=1000,
                        help='Number of diffusion steps')
    parser.add_argument('--beta_start', type=float, default=1e-4,
                        help='Starting value for noise schedule')
    parser.add_argument('--beta_end', type=float, default=0.02,
                        help='Ending value for noise schedule')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate for optimizer')
    parser.add_argument('--weight_decay', type=float, default=1e-6,
                        help='Weight decay for optimizer')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of dataloader workers')
    
    # Logging and checkpointing
    parser.add_argument('--log_interval', type=int, default=10,
                        help='How often to log training progress')
    parser.add_argument('--checkpoint_interval', type=int, default=5,
                        help='How often to save checkpoints')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--use_wandb', action='store_true',
                        help='Whether to use Weights & Biases logging')
    
    args = parser.parse_args()
    
    # Convert paths to absolute paths
    args.wav_folder = os.path.abspath(args.wav_folder)
    args.csv_file = os.path.abspath(args.csv_file)
    args.checkpoint_dir = os.path.abspath(args.checkpoint_dir)
    
    # Print paths
    print(f"WAV folder path: {args.wav_folder}")
    print(f"CSV file path: {args.csv_file}")
    print(f"Checkpoint directory: {args.checkpoint_dir}")
    
    # Validate paths
    if not os.path.exists(args.wav_folder):
        raise ValueError(f"WAV folder not found: {args.wav_folder}")
    if not os.path.exists(args.csv_file):
        raise ValueError(f"CSV file not found: {args.csv_file}")
    
    # Print training configuration
    print("\nTraining Configuration:")
    print("-" * 50)
    print(f"Model Hidden Dimension: {args.hidden_dim}")
    print(f"Diffusion Steps: {args.time_steps}")
    print(f"Beta Schedule: {args.beta_start} to {args.beta_end}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning Rate: {args.learning_rate}")
    print(f"Weight Decay: {args.weight_decay}")
    print(f"Number of Workers: {args.num_workers}")
    print("-" * 50)
    print()
    
    try:
        train(args)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"\nError during training: {str(e)}")
        raise
    finally:
        if args.use_wandb:
            wandb.finish()