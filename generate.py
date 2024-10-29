# generate.py
import torch
import torchaudio 
from transformers import AutoTokenizer
from models import MusicGenerator
import argparse
from tqdm import tqdm

def interpolate_audio(audio1, audio2, alpha=0.5):
    """Interpolate between two audio tensors"""
    return alpha * audio1 + (1 - alpha) * audio2

def generate(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model = MusicGenerator().to(device)
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Prepare input
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    
    lyrics_tokens = tokenizer(args.lyrics, 
                            return_tensors='pt', 
                            padding=True, 
                            truncation=True, 
                            max_length=512)
    prompt_tokens = tokenizer(args.prompt, 
                            return_tensors='pt', 
                            padding=True, 
                            truncation=True, 
                            max_length=512)
    
    # Move to device
    lyrics_tokens = {k: v.to(device) for k, v in lyrics_tokens.items()}
    prompt_tokens = {k: v.to(device) for k, v in prompt_tokens.items()}
    
    # Generate with multiple steps
    with torch.no_grad():
        # Initial generation
        audio, mel_spec = model(lyrics_tokens, prompt_tokens)
        current_audio = audio
        
        # Refinement steps
        if args.num_steps > 1:
            print(f"Refining audio over {args.num_steps} steps...")
            for step in tqdm(range(args.num_steps - 1)):
                # Generate new version
                new_audio, new_mel = model(lyrics_tokens, prompt_tokens)
                
                # Progressive interpolation
                # As steps increase, we give more weight to the refined versions
                alpha = (step + 1) / args.num_steps
                current_audio = interpolate_audio(new_audio, current_audio, alpha)

    # Properly reshape audio tensor for torchaudio.save()
    audio = current_audio.squeeze()  # Remove any extra dimensions
    if audio.dim() == 1:
        audio = audio.unsqueeze(0)
    elif audio.dim() == 3:
        audio = audio[0]
    
    # Optional denoising (can help with multiple steps)
    if args.denoise_strength > 0:
        # Simple moving average filter
        kernel_size = 3
        kernel = torch.ones(1, 1, kernel_size, device=audio.device) / kernel_size
        audio = torch.nn.functional.conv1d(
            audio.unsqueeze(0), 
            kernel, 
            padding=kernel_size//2
        ).squeeze(0) * args.denoise_strength + audio * (1 - args.denoise_strength)

    # Move to CPU
    audio = audio.cpu()
    
    # Save audio
    torchaudio.save(args.output_file, audio, 22050)
    
    if args.save_intermediate:
        # Save mel spectrogram visualization if requested
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 5))
        plt.imshow(mel_spec[0].cpu().numpy(), aspect='auto', origin='lower')
        plt.colorbar()
        plt.savefig(args.output_file.replace('.wav', '_mel.png'))
        plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', type=str, required=True)
    parser.add_argument('--lyrics', type=str, required=True)
    parser.add_argument('--prompt', type=str, required=True)
    parser.add_argument('--output_file', type=str, default='generated_music.wav')
    parser.add_argument('--num_steps', type=int, default=10, 
                        help='Number of generation steps (more steps = more refinement)')
    parser.add_argument('--denoise_strength', type=float, default=0.1,
                        help='Strength of denoising (0-1, 0 = no denoising)')
    parser.add_argument('--save_intermediate', action='store_true',
                        help='Save mel spectrogram visualization')
    
    args = parser.parse_args()
    
    generate(args)