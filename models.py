import torch
import torch.nn as nn
from transformers import AutoModel
import math
from torch.nn.utils import weight_norm

class DiffusionModel(nn.Module):
    def __init__(self, hidden_dim=512, mel_channels=80, time_steps=1000):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.time_steps = time_steps
        
        # Text encoders
        self.lyrics_encoder = AutoModel.from_pretrained('bert-base-uncased')
        self.prompt_encoder = AutoModel.from_pretrained('bert-base-uncased')
        
        bert_output_dim = 768
        # Adjust projection dimensions
        self.lyrics_projection = nn.Linear(bert_output_dim, hidden_dim // 2)
        self.prompt_projection = nn.Linear(bert_output_dim, hidden_dim // 2)
        
        # Time embedding
        time_dim = hidden_dim * 4
        self.time_embed = nn.Sequential(
            nn.Linear(hidden_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, hidden_dim)
        )
        
        # Initial number of channels
        initial_channels = 1
        
        # U-Net style architecture with adjusted channel numbers
        self.down1 = UNetBlock(initial_channels, hidden_dim // 4, hidden_dim // 2, hidden_dim)
        self.down2 = UNetBlock(hidden_dim // 2, hidden_dim // 2, hidden_dim, hidden_dim)
        self.down3 = UNetBlock(hidden_dim, hidden_dim, hidden_dim * 2, hidden_dim)
        
        self.mid = nn.Sequential(
            nn.Conv2d(hidden_dim * 2, hidden_dim * 2, 3, padding=1),
            nn.InstanceNorm2d(hidden_dim * 2),
            nn.SiLU(),
            nn.Conv2d(hidden_dim * 2, hidden_dim * 2, 3, padding=1)
        )
        
        self.up3 = UNetBlock(hidden_dim * 4, hidden_dim * 2, hidden_dim, hidden_dim)
        self.up2 = UNetBlock(hidden_dim * 2, hidden_dim, hidden_dim // 2, hidden_dim)
        self.up1 = UNetBlock(hidden_dim, hidden_dim // 2, hidden_dim // 4, hidden_dim)
        
        self.final = nn.Conv2d(hidden_dim // 4, 1, 1)
    
    def get_time_embedding(self, timesteps, device):
        half = self.hidden_dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(device)
        args = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return self.time_embed(embedding)
    
    def forward(self, x, timesteps, lyrics_tokens, prompt_tokens):
        # Encode text inputs
        lyrics_encoding = self.lyrics_encoder(**lyrics_tokens).last_hidden_state
        prompt_encoding = self.prompt_encoder(**prompt_tokens).last_hidden_state
        
        # Get mean of text encodings
        lyrics_feat = self.lyrics_projection(lyrics_encoding.mean(1))  # [batch, hidden_dim//2]
        prompt_feat = self.prompt_projection(prompt_encoding.mean(1))  # [batch, hidden_dim//2]
        
        # Combine condition embeddings
        cond = torch.cat([lyrics_feat, prompt_feat], dim=-1)  # [batch, hidden_dim]
        
        # Get time embedding
        t_emb = self.get_time_embedding(timesteps, x.device)  # [batch, hidden_dim]
        
        # U-Net forward pass
        x1 = self.down1(x, t_emb, cond)
        x2 = self.down2(x1, t_emb, cond)
        x3 = self.down3(x2, t_emb, cond)
        
        x = self.mid(x3)
        
        x = self.up3(torch.cat([x, x3], dim=1), t_emb, cond)
        x = self.up2(torch.cat([x, x2], dim=1), t_emb, cond)
        x = self.up1(torch.cat([x, x1], dim=1), t_emb, cond)
        
        return self.final(x)

class UNetBlock(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, emb_dim):
        super().__init__()
        
        # For channels less than 4, use Instance Normalization
        if in_channels < 4:
            self.in_layers = nn.Sequential(
                nn.InstanceNorm2d(in_channels),
                nn.SiLU(),
                nn.Conv2d(in_channels, hidden_channels, 3, padding=1)
            )
        else:
            self.in_layers = nn.Sequential(
                nn.GroupNorm(min(8, in_channels), in_channels),
                nn.SiLU(),
                nn.Conv2d(in_channels, hidden_channels, 3, padding=1)
            )
        
        # Embedding projection
        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, hidden_channels)
        )
        
        # Condition projection
        self.cond_projection = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, hidden_channels)
        )
        
        if hidden_channels < 4:
            self.out_layers = nn.Sequential(
                nn.InstanceNorm2d(hidden_channels),
                nn.SiLU(),
                nn.Conv2d(hidden_channels, out_channels, 3, padding=1)
            )
        else:
            self.out_layers = nn.Sequential(
                nn.GroupNorm(min(8, hidden_channels), hidden_channels),
                nn.SiLU(),
                nn.Conv2d(hidden_channels, out_channels, 3, padding=1)
            )
        
        if in_channels != out_channels:
            self.skip_connection = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.skip_connection = nn.Identity()
    
    def forward(self, x, t_emb, cond):
        h = self.in_layers(x)
        
        # Add time embedding
        emb_out = self.emb_layers(t_emb)[:, :, None, None]
        
        # Add condition embedding
        cond_out = self.cond_projection(cond)[:, :, None, None]
        
        h = h + emb_out + cond_out
        h = self.out_layers(h)
        
        return self.skip_connection(x) + h

def get_padding(kernel_size, dilation=1):
    return int((kernel_size*dilation - dilation)/2)