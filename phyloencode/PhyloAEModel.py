
import torch
import torch
import torch.nn as nn
import torch.nn.functional as fun
from torch.utils.data import Dataset, DataLoader, TensorDataset
from typing import List, Dict, Tuple, Optional, Union

from phyloencode import utils

class PhyloAEModelCNN(nn.Module):
    '''
        This class uses a CNN and a dense layer to encode which are concatenated
        in a latent layer which then gets decoded by a transpose CNN and dense layer 
        in parallel.
    '''
    def __init__(self, 
                 num_structured_input_channel, 
                 structured_input_width,  # Input width for structured data
                 unstructured_input_width,
                 unstructured_latent_width = None, # must be integer multiple of num_structured_latent_channels
                 stride = [2,2,4,4],
                 kernel = [3,3,5,5],
                 out_channels = [8, 16, 32, 64]
                 ):
        
        # inputs are assumed to be standardized

        nl = len(out_channels)
        assert(len(stride) == nl)
        assert(len(kernel) == nl)
        
        self.unstructured_latent_width = unstructured_latent_width
        self.num_structured_latent_channels = out_channels[-1]
        
        # if not set, then set to structured latent channels
        if unstructured_latent_width is None:
            self.unstructured_latent_width = self.num_structured_latent_channels

        # Validate divisibility of the two latent sizes
        if self.unstructured_latent_width % self.num_structured_latent_channels != 0:
            raise ValueError("""unstructured_latent_width must be an integer 
                             multiple of num_structured_latent_channels""")
        
        super().__init__()


        # Unstructured Encoder
        self.unstructured_encoder = nn.Sequential(
            nn.Linear(unstructured_input_width, 64),
            nn.ReLU(),
            nn.Linear(64, self.unstructured_latent_width),
            nn.ReLU()
        )

        # Structured Encoder
        self.structured_encoder = nn.Sequential()
        self.structured_encoder.add_module("conv1d_0", nn.Conv1d(in_channels  = num_structured_input_channel, 
                                                                out_channels = out_channels[0], 
                                                                kernel_size  = kernel[0], 
                                                                stride       = stride[0], 
                                                                padding      = 1))
        self.structured_encoder.add_module("conv_ReLU_0", nn.ReLU())
        for i in range(1, nl):
            self.structured_encoder.add_module("conv1d_" + str(i), 
                                               nn.Conv1d(in_channels  = out_channels[i-1], 
                                                        out_channels = out_channels[i], 
                                                        kernel_size  = kernel[i], 
                                                        stride       = stride[i], 
                                                        padding      = 1))
            if i < (nl-1):
                self.structured_encoder.add_module("conv_ReLU_" + str(i), nn.ReLU())

        # Calculate final structured output width after encoder
        struct_outshape = utils.conv1d_sequential_outshape(self.structured_encoder, 
                                                           num_structured_input_channel, 
                                                           structured_input_width)
        
        structured_output_width = struct_outshape[2]
        flat_structured_width = structured_output_width * self.num_structured_latent_channels
        self.combined_latent_width = flat_structured_width + self.unstructured_latent_width
        self.reshaped_shared_latent_width = self.combined_latent_width // self.num_structured_latent_channels


        # Shared Latent Layer
        self.shared_layer = nn.Sequential(
            nn.Linear(self.combined_latent_width, self.combined_latent_width),
            nn.ReLU()
        )
        
        # Unstructured Decoder
        self.unstructured_decoder = nn.Sequential(
            nn.Linear(self.combined_latent_width, 64),
            nn.ReLU(),
            nn.Linear(64, unstructured_input_width)
        )

        self.structured_decoder = nn.Sequential()
        self.structured_decoder.add_module("trans_conv1d_0", 
                                                nn.ConvTranspose1d(in_channels  = out_channels[nl-1], 
                                                          out_channels = out_channels[nl-2], 
                                                          kernel_size  = kernel[nl-1], 
                                                          stride       = stride[nl-1], 
                                                          padding      = 1))
        self.structured_decoder.add_module("tconv_ReLU_0", nn.ReLU())
        for i in range(nl-2, 0, -1):
            self.structured_decoder.add_module("conv1d_" + str(i), 
                                               nn.ConvTranspose1d(in_channels  = out_channels[i], 
                                                        out_channels = out_channels[i-1], 
                                                        kernel_size  = kernel[i], 
                                                        stride       = stride[i], 
                                                        padding      = 1))
            self.structured_decoder.add_module("tconv_ReLU_" + str(i), nn.ReLU())

        # get correct padding and output_padding for final decoder layer
        struct_decode_out_shape = utils.tconv1d_sequential_outshape(self.structured_decoder, 
                                                     self.num_structured_latent_channels, 
                                                     self.reshaped_shared_latent_width)
        
        width = struct_decode_out_shape[2]
        pad = int(((width - 1)*stride[0] + 1*(kernel[0]-1) +1 -structured_input_width)/2) + 1
        outpad = structured_input_width - (width - 1)*stride[0] - 1*(kernel[0]-1) -1 + 2*pad
        
        print("phy latent shape: " + str(struct_outshape))
        print("pad: " + str(pad))
        print("outpad: " + str(outpad))
        print(struct_decode_out_shape)

        self.structured_decoder.add_module("struct_decoder_out", 
                                           nn.ConvTranspose1d(in_channels=out_channels[0], 
                                                              out_channels=num_structured_input_channel, 
                                                              kernel_size=kernel[0], stride=stride[0], padding=pad, 
                                                              output_padding=outpad))
        # print(list(self.structured_encoder))
        # print(list(self.structured_decoder))




    def forward(self, data: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        # Encode
        # data is a tuple (structured, unstructured)
        structured_encoded_x   = self.structured_encoder(data[0]) # (nbatch, nchannels, out_width)
        unstructured_encoded_x = self.unstructured_encoder(data[1]) # (nbatch, out_width)

        # Combine Latents
        flat_structured_encoded_x = structured_encoded_x.flatten(start_dim=1)
        combined_latent           = torch.cat((flat_structured_encoded_x, unstructured_encoded_x), dim=1)

        # print("self.combined_latent_width")
        # print(self.combined_latent_width)
        # print("self.num_structured_latent_channels")
        # print(self.num_structured_latent_channels)
        # print("structured_encoded_x.shape")
        # print(structured_encoded_x.shape)
        # print("unstructured_encoded_x.shape")
        # print(unstructured_encoded_x.shape)
        # print("flat_structured_encoded_x.shape")
        # print(flat_structured_encoded_x.shape)
        # print("combined_latent.shape")
        # print(combined_latent.shape)


        shared_latent = self.shared_layer(combined_latent)

        # Reshape for structured decoder (must have self.num_structured_latent_channels channels)
        reshaped_shared_latent = shared_latent.view(-1, self.num_structured_latent_channels, 
                                                          self.reshaped_shared_latent_width)

        # Decode
        unstructured_decoded_x = self.unstructured_decoder(shared_latent)
        structured_decoded_x   = self.structured_decoder(reshaped_shared_latent)

        # print("shared_latent.shape")
        # print(shared_latent.shape)
        # print("reshaped_shared_latent.shape")
        # print(reshaped_shared_latent.shape)
        # print("unstructured_decoded_x.shape")
        # print(unstructured_decoded_x.shape)
        # print("structured_decoded_x.shape")
        # print(structured_decoded_x.shape)

        return structured_decoded_x, unstructured_decoded_x
    

    def tree_encode(self, phy, aux):
        return(self.structured_encoder(phy))
