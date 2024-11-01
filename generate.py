import torch
import torchaudio 
from transformers import AutoTokenizer
from models import DiffusionModel, HiFiGANGenerator
import argparse
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import os

class DiffusionSampler:
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
        
    def sample(self, model, lyrics_tokens, prompt_tokens, device, batch_size=1, mel_channels=80, mel_length=860):
        # Start from pure noise
        x = torch.randn(batch_size, 1, mel_channels, mel_length).to(device)
        
        # Gradually denoise
        for t in tqdm(range(self.time_steps - 1, -1, -1), desc="Sampling"):
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
            
            # Get model prediction
            with torch.no_grad():
                predicted_noise = model(x, t_batch, lyrics_tokens, prompt_tokens)
            
            # Update sample
            alpha = self.alphas[t]
            alpha_cumprod = self.alphas_cumprod[t]
            beta = self.betas[t]
            
            if t > 0:
                noise = torch.randn_like(x)
            else:
                noise = 0
                
            x = (1 / torch.sqrt(alpha)) * (
                x - ((1 - alpha) / (torch.sqrt(1 - alpha_cumprod))) * predicted_noise
            ) + torch.sqrt(beta) * noise
            
        return x

def mel_to_audio(mel_spec, device):
    """Convert mel spectrogram to audio using inverse mel spectrogram"""
    # Initialize inverse mel spectrogram transform
    inverse_mel = torchaudio.transforms.InverseMelScale(
        n_stft=(1024 // 2 + 1),  # n_fft // 2 + 1
        n_mels=80,
        sample_rate=22050,
        f_min=0,
        f_max=8000,
    ).to(device)
    
    # Initialize Griffin-Lim algorithm
    griffin_lim = torchaudio.transforms.GriffinLim(
        n_fft=1024,
        n_iter=32,  # Number of iterations for Griffin-Lim
        win_length=1024,
        hop_length=256,
    ).to(device)
    waveform = torch.clamp(waveform, -1, 1)  # Ensure values are in [-1, 1]
    
    # Ensure mel_spec is in the correct format [batch, mel_channels, time]
    if len(mel_spec.shape) == 4:  # [batch, 1, mel_channels, time]
        mel_spec = mel_spec.squeeze(1)
    
    # Convert to linear spectrogram
    linear_spec = inverse_mel(mel_spec)
    
    # Apply Griffin-Lim algorithm to recover phase information
    waveform = griffin_lim(linear_spec)
    
    return waveform

def save_mel_spectrogram(mel_spec, filename):
    """Save mel spectrogram visualization"""
    plt.figure(figsize=(10, 5))
    plt.imshow(mel_spec[0, 0].cpu().numpy(), aspect='auto', origin='lower')
    plt.colorbar()
    plt.title("Generated Mel Spectrogram")
    plt.xlabel("Time")
    plt.ylabel("Mel Frequency")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def generate(args):
    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
    
    # Load model
    print("\nLoading model...")
    model = DiffusionModel(
        hidden_dim=args.hidden_dim,
        mel_channels=80,
        time_steps=args.time_steps
    ).to(device)
    
    # Load checkpoint
    try:
        checkpoint = torch.load(args.checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Successfully loaded checkpoint from {args.checkpoint_path}")
    except Exception as e:
        print(f"Error loading checkpoint: {str(e)}")
        return
    
    model.eval()
    
    # Initialize sampler
    sampler = DiffusionSampler(
        time_steps=args.time_steps,
        beta_start=args.beta_start,
        beta_end=args.beta_end
    )
    
    # Move schedules to device
    sampler.betas = sampler.betas.to(device)
    sampler.alphas = sampler.alphas.to(device)
    sampler.alphas_cumprod = sampler.alphas_cumprod.to(device)
    sampler.sqrt_alphas_cumprod = sampler.sqrt_alphas_cumprod.to(device)
    sampler.sqrt_one_minus_alphas_cumprod = sampler.sqrt_one_minus_alphas_cumprod.to(device)
    
    # Prepare input
    print("\nPreparing inputs...")
    print(f"Lyrics: {args.lyrics}")
    print(f"Prompt: {args.prompt}")
    
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    
    lyrics_tokens = tokenizer(
        args.lyrics, 
        return_tensors='pt', 
        padding=True, 
        truncation=True, 
        max_length=512
    )
    
    prompt_tokens = tokenizer(
        args.prompt, 
        return_tensors='pt', 
        padding=True, 
        truncation=True, 
        max_length=128
    )
    
    # Move to device
    lyrics_tokens = {k: v.to(device) for k, v in lyrics_tokens.items()}
    prompt_tokens = {k: v.to(device) for k, v in prompt_tokens.items()}
    
    # Generate mel spectrogram
    print("\nGenerating mel spectrogram...")
    try:
        mel_spec = sampler.sample(
            model,
            lyrics_tokens,
            prompt_tokens,
            device,
            batch_size=1,
            mel_channels=80,
            mel_length=860
        )
    except Exception as e:
        print(f"Error during mel spectrogram generation: {str(e)}")
        return
    
    # Save mel spectrogram visualization if requested
    if args.save_intermediate:
        mel_plot_path = args.output_file.replace('.wav', '_mel.png')
        print(f"\nSaving mel spectrogram visualization to {mel_plot_path}")
        save_mel_spectrogram(mel_spec, mel_plot_path)
    
    # Convert mel spectrogram to audio
    print("\nConverting mel spectrogram to audio...")
    try:
        audio = mel_to_audio(mel_spec, device)
    except Exception as e:
        print(f"Error during mel to audio conversion: {str(e)}")
        # Save mel spectrogram even if audio conversion fails
        if not args.save_intermediate:
            mel_plot_path = args.output_file.replace('.wav', '_mel.png')
            save_mel_spectrogram(mel_spec, mel_plot_path)
        return
    
    # Ensure audio is in the correct format for saving
    if len(audio.shape) == 1:
        audio = audio.unsqueeze(0)
    elif len(audio.shape) == 3:
        audio = audio.squeeze(0)
    
    # Move to CPU and save
    audio = audio.cpu()
    
    print(f"\nSaving audio to {args.output_file}")
    try:
        torchaudio.save(args.output_file, audio, 22050)
        print("Successfully saved audio file!")
    except Exception as e:
        print(f"Error saving audio file: {str(e)}")
    
    print("\nGeneration complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate music using the trained diffusion model")
    
    # Model parameters
    parser.add_argument('--checkpoint_path', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--hidden_dim', type=int, default=512,
                        help='Hidden dimension of the model')
    parser.add_argument('--time_steps', type=int, default=1000,
                        help='Number of diffusion steps')
    parser.add_argument('--beta_start', type=float, default=1e-4,
                        help='Starting value for noise schedule')
    parser.add_argument('--beta_end', type=float, default=0.02,
                        help='Ending value for noise schedule')
    
    # Generation parameters
    parser.add_argument('--lyrics', type=str, required=True,
                        help='Lyrics for generation')
    parser.add_argument('--prompt', type=str, required=True,
                        help='Additional prompt for generation')
    parser.add_argument('--output_file', type=str, default='generated_music.wav',
                        help='Output file path')
    
    # Additional options
    parser.add_argument('--save_intermediate', action='store_true',
                        help='Save mel spectrogram visualization')
    
    args = parser.parse_args()
    
    try:
        generate(args)
    except KeyboardInterrupt:
        print("\nGeneration interrupted by user")
    except Exception as e:
        print(f"\nUnexpected error during generation: {str(e)}")
        raise