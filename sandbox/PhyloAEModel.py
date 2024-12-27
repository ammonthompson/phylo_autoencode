
import torch
import torch.nn as nn
import torch.nn.functional as fun
from torch.utils.data import Dataset, DataLoader, TensorDataset


class PhyloAEModelCNN(nn.Module):
    def __init__(self, 
                 num_structured_input_width,  # Input width for structured data
                 num_structured_input_channel, 
                 unstructured_input_width,
                 num_structured_latent_channels, 
                 unstructured_latent_width # one latent space is the integer multiple of the other
                 ):
        
        self.unstructured_latent_width = unstructured_latent_width
        self.num_structured_latent_channels = num_structured_latent_channels
        
        # Validate divisibility of the two latent sizes
        if max(self.num_structured_latent_channels, self.unstructured_latent_width) % \
           min(self.num_structured_latent_channels, self.unstructured_latent_width) != 0:
            raise ValueError("One latent size must be an integer multiple of the other.")
        
        super().__init__()

        # Unstructured Encoder
        self.unstructured_encoder = nn.Sequential(
            nn.Linear(unstructured_input_width, 128),
            nn.ReLU(),
            nn.Linear(128, unstructured_latent_width),
            nn.ReLU()
        )

        # Structured Encoder
        self.structured_encoder = nn.Sequential(
            nn.Conv1d(in_channels=num_structured_input_channel, out_channels=16, 
                      kernel_size=3, stride=2, padding=1),  
            nn.ReLU(),
            nn.Conv1d(in_channels=16, out_channels=32, 
                      kernel_size=3, stride=2, padding=1),  
            nn.ReLU(),
            nn.Conv1d(in_channels=32, out_channels=num_structured_latent_channels, 
                      kernel_size=3, stride=2, padding=1),  
            nn.ReLU()
        )

        # Calculate final structured output width after encoder
        structured_output_width = num_structured_input_width // (2**3)  # 3 Conv layers with stride=2
        flat_structured_width = structured_output_width * num_structured_latent_channels
        self.combined_latent_width = flat_structured_width + unstructured_latent_width

        # Shared Latent Layer
        self.shared_layer = nn.Sequential(
            nn.Linear(self.combined_latent_width, self.combined_latent_width),
            nn.ReLU()
        )

        # Unstructured Decoder
        self.unstructured_decoder = nn.Sequential(
            nn.Linear(self.combined_latent_width, 128),
            nn.ReLU(),
            nn.Linear(128, unstructured_input_width)
        )
        
        # Structured Decoder
        self.structured_decoder = nn.Sequential(
            nn.ConvTranspose1d(in_channels=num_structured_latent_channels, out_channels=32, 
                               kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(in_channels=32, out_channels=16, 
                               kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(in_channels=16, out_channels=num_structured_input_channel, 
                               kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )


    def forward(self, unstructured_x, structured_x):
        # Encode
        unstructured_encoded_x = self.unstructured_encoder(unstructured_x) # (nbatch, out_width)
        structured_encoded_x   = self.structured_encoder(structured_x) # (nbatch, nchannels, out_width)

        # Combine Latents
        flat_structured_encoded_x = structured_encoded_x.flatten(start_dim=1)
        combined_latent           = torch.cat((flat_structured_encoded_x, unstructured_encoded_x), dim=1)
        shared_latent             = self.shared_layer(combined_latent)

        # Reshape for structured decoder (must have self.num_structured_latent_channels channels)
        reshaped_latent_width = self.combined_latent_width // self.num_structured_latent_channels
        reshaped_latent       = shared_latent.view(-1, self.num_structured_latent_channels, reshaped_latent_width)

        # Decode
        unstructured_decoded_x = self.unstructured_decoder(shared_latent)
        structured_decoded_x   = self.structured_decoder(reshaped_latent)

        return unstructured_decoded_x, structured_decoded_x

    


