import torch.nn.functional as F
from torch import nn
import torch

class VAE(nn.Module):
    def __init__(self, input_size, num_channels, latent_dim, num_classes):
        super(VAE, self).__init__()
        self.input_size = input_size
        self.num_channels = num_channels
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        
        self.encoder = nn.Sequential(
            nn.Conv1d(num_channels, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU()
        )
        with torch.no_grad():
            dummy_input = torch.zeros(1, num_channels, input_size)
            encoder_output = self.encoder(dummy_input)
            self.flattened_size = encoder_output.view(1, -1).size(1)
        
        self.fc_mu = nn.Linear(self.flattened_size, latent_dim)
        self.fc_logvar = nn.Linear(self.flattened_size, latent_dim)
        self.class_embedding = nn.Embedding(num_classes, latent_dim)
        
        self.fc_decoder = nn.Linear(latent_dim * 2 * num_channels, self.flattened_size)
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(64, 32, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(32, num_channels, kernel_size=5, stride=2, padding=3, output_padding=1),
            nn.Tanh()
        )
    
    def encode(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z, class_labels):
        z = z.unsqueeze(1).expand(-1, self.num_channels, -1)
        class_embed = self.class_embedding(class_labels)
        class_embed = class_embed.unsqueeze(1).expand(-1, self.num_channels, -1)
        z = torch.cat([z, class_embed], dim=-1)
        z = z.view(z.size(0), -1)
        x = self.fc_decoder(z)
        x = x.view(x.size(0), 64, -1)  
        
        x = self.decoder(x)
        x = F.interpolate(x, size=self.input_size, mode='linear', align_corners=False)
        return x

    def forward(self, x, class_labels):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z, class_labels)
        return x_recon, mu, logvar
