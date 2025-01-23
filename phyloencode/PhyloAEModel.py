import numpy as np
import torch
import torch
import torch.nn as nn
import torch.nn.functional as fun
from torch.utils.data import Dataset, DataLoader, TensorDataset
from typing import List, Dict, Tuple, Optional, Union

from phyloencode import utils

class AECNN(nn.Module):
    '''
        This class uses a CNN and a dense layer to encode which are concatenated
        in a latent layer which then gets decoded by a transpose CNN and dense layer 
        in parallel.
    '''
    def __init__(self, #ae_data_container,
                 num_structured_input_channel, 
                 structured_input_width,  # Input width for structured data
                 unstructured_input_width,
                 unstructured_latent_width = None, # must be integer multiple of num_structured_latent_channels
                 stride = [2,2],
                 kernel = [3,3],
                 out_channels = [16, 32]
                 ):
        
        # assumptions:
        # inputs are standardized
        # all strides > 1
        # kernal[i] > stride[i]

        nl = len(out_channels)
        assert(len(stride) == nl)
        assert(len(kernel) == nl)
        assert(np.min(stride) > 1)
        assert(np.min(np.array(kernel) - np.array(stride)) > 0)
                
        # input dimensions
        self.num_structured_input_channel = num_structured_input_channel
        self.structured_input_width       = structured_input_width
        self.unstructured_input_width     = unstructured_input_width

        # some latent layer dimensions
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
        print("cumulative conv layers output shapes: ")
        print((1, self.num_structured_input_channel, self.structured_input_width))

        self.structured_encoder = nn.Sequential()
        self.structured_encoder.add_module("conv1d_0", nn.Conv1d(in_channels  = self.num_structured_input_channel, 
                                                                out_channels = out_channels[0], 
                                                                kernel_size  = kernel[0], 
                                                                stride       = stride[0], 
                                                                padding      = 1))
        
        conv_out_shape = utils.conv1d_sequential_outshape(self.structured_encoder, 
                                                self.num_structured_input_channel, 
                                                self.structured_input_width)
        conv_out_width = [conv_out_shape[2]]
        print(conv_out_shape)

        # add more layers if depth greater than 1.
        if nl > 1:
            self.structured_encoder.add_module("conv_ReLU_0", nn.ReLU())
            for i in range(1, nl):
                self.structured_encoder.add_module("conv1d_" + str(i), 
                                                nn.Conv1d(in_channels  = out_channels[i-1], 
                                                            out_channels = out_channels[i], 
                                                            kernel_size  = kernel[i], 
                                                            stride       = stride[i], 
                                                            padding      = 1))
                
                conv_out_shape = utils.conv1d_sequential_outshape(self.structured_encoder, 
                                                        self.num_structured_input_channel, 
                                                        self.structured_input_width)
                conv_out_width.append(conv_out_shape[2])                
                
                print(conv_out_shape)

                if i < (nl-1):
                    self.structured_encoder.add_module("conv_ReLU_" + str(i), nn.ReLU())

        # Calculate final structured output width after encoder
        struct_outshape = utils.conv1d_sequential_outshape(self.structured_encoder, 
                                                           self.num_structured_input_channel, 
                                                           self.structured_input_width)
        
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
            nn.Linear(64, self.unstructured_input_width)
        )

        self.structured_decoder = nn.Sequential()

        # TODO: right now there has to be at least 2 conv layers. 
        # Implement this so one conv layer is possible
        if nl == 1:
            pass
        elif nl == 2:
            pass
        else:
            pass

        self.structured_decoder.add_module("trans_conv1d_0", 
                                                nn.ConvTranspose1d(in_channels  = out_channels[nl-1], 
                                                          out_channels = out_channels[nl-2], 
                                                          kernel_size  = kernel[nl-1], 
                                                          stride       = stride[nl-1], 
                                                          padding      = 0,
                                                          output_padding = 0))
 
        self.structured_decoder.add_module("tconv_ReLU_0", nn.ReLU())

        print(struct_outshape)
        outshape = utils.tconv1d_sequential_outshape(self.structured_decoder, 
                                                     self.num_structured_latent_channels, 
                                                     self.reshaped_shared_latent_width)
        print(outshape)

        for i in range(nl-2, 0, -1):
            outshape = utils.tconv1d_sequential_outshape(self.structured_decoder, 
                                                     self.num_structured_latent_channels, 
                                                     self.reshaped_shared_latent_width)
            w_in = outshape[2]
            new_target_width = conv_out_width[i-1]
            npad, noutpad = self.get_decode_paddings(new_target_width, w_in, stride[i], kernel[i])

            self.structured_decoder.add_module("conv1d_" + str(i), 
                                               nn.ConvTranspose1d(in_channels  = out_channels[i], 
                                                        out_channels = out_channels[i-1], 
                                                        kernel_size  = kernel[i], 
                                                        stride       = stride[i], 
                                                        padding      = npad,
                                                        output_padding= noutpad))
            
            print(utils.tconv1d_sequential_outshape(self.structured_decoder, 
                                                self.num_structured_latent_channels, 
                                                self.reshaped_shared_latent_width))

            self.structured_decoder.add_module("tconv_ReLU_" + str(i), nn.ReLU())

        # get correct padding and output_padding for final decoder layer
        outshape = utils.tconv1d_sequential_outshape(self.structured_decoder, 
                                                     self.num_structured_latent_channels, 
                                                     self.reshaped_shared_latent_width)
        

        width = outshape[2]
        new_target_width = structured_input_width
        pad, outpad = self.get_decode_paddings(new_target_width, width, stride[0], kernel[0])

        self.structured_decoder.add_module("struct_decoder_out", 
                                           nn.ConvTranspose1d(in_channels=out_channels[0], 
                                                              out_channels=self.num_structured_input_channel, 
                                                              kernel_size=kernel[0], stride=stride[0], padding=pad, 
                                                              output_padding=outpad))
        
        print(utils.tconv1d_sequential_outshape(self.structured_decoder, 
                                                     self.num_structured_latent_channels, 
                                                     self.reshaped_shared_latent_width))



    def forward(self, data: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        # Encode
        # data is a tuple (structured, unstructured)
        structured_encoded_x   = self.structured_encoder(data[0]) # (nbatch, nchannels, out_width)
        unstructured_encoded_x = self.unstructured_encoder(data[1]) # (nbatch, out_width)

        # Combine Latents
        flat_structured_encoded_x = structured_encoded_x.flatten(start_dim=1)
        combined_latent           = torch.cat((flat_structured_encoded_x, unstructured_encoded_x), dim=1)

        shared_latent = self.shared_layer(combined_latent)

        # Reshape for structured decoder (must have self.num_structured_latent_channels channels)
        reshaped_shared_latent = shared_latent.view(-1, self.num_structured_latent_channels, 
                                                          self.reshaped_shared_latent_width)

        # Decode
        unstructured_decoded_x = self.unstructured_decoder(shared_latent)
        structured_decoded_x   = self.structured_decoder(reshaped_shared_latent)

        return structured_decoded_x, unstructured_decoded_x
    
    def make_encoders(self):
        pass
    def make_decoders(self):
        pass

    def get_decode_paddings(self, w_out_target, w_in, s, k, d=1):
        # convtranspose1d formulat: 
        # w_out_target = (w_in - 1)s + d(k-1) + 1 + outpad - 2pad
        # returns the paddings necessary for target output width 
        E = s*(w_in - 1) + d*(k-1) + 1

        if (w_out_target - E) < 0:
            pad = (E - w_out_target)/2
            pad = int(pad + pad % 1)
            outpad = w_out_target - E + 2 * pad    #if (w_out_target - E + 2 * pad) <= (s-1) else 0
        elif (w_out_target - E) >=0 and (w_out_target - E) <= (s - 1):
            outpad = w_out_target - E
            pad = 0
        else:
            pad = (E - w_out_target + s - 1)/2
            pad = int(pad + pad % 1)
            outpad = s-1

        return pad, outpad
    
