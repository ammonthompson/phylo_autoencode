import torch
import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
from phyloencode import utils

class AECNN(nn.Module):
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
                 stride = [2,2],
                 kernel = [3,3],
                 out_channels = [16, 32],
                 latent_layer_type = "CNN"     # CNN, DENSE, GAUSS
                 ):
        
        # assumptions:
        # inputs are standardized
        # at least 2 convolution layers
        # all strides > 1
        # kernal[i] > stride[i]

        super().__init__()

        nl = len(out_channels)
        # ensure stride and kernel array lengths match the out_channel earray's length
        if len(stride) != nl:
            raise ValueError(f"Expected stride array length of {nl}, but got {len(stride)}.")

        if len(kernel) != nl:
            raise ValueError(f"Expected kernel array length of {nl}, but got {len(kernel)}.")

        # Ensure stride values are greater than 1
        if np.min(stride) <= 1:
            raise ValueError("All stride values must be greater than 1.")

        # Ensure kernel values are greater than stride values
        stride_arr, kernel_arr = np.array(stride), np.array(kernel)
        if np.min(kernel_arr - stride_arr) <= 0:
            raise ValueError("Each kernel value must be greater than the corresponding stride value.")

        # convolution layers parameters
        self.layer_params = {"out_channels": out_channels,
                            "kernel"      : kernel,
                            "stride"      : stride}
                
        self.latent_layer_type = latent_layer_type

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
        
        ''' 
        create NN architecture 
        '''
        # Unstructured Encoder
        self.unstructured_encoder = DenseEncoder(self.unstructured_input_width, 
                                                 self.unstructured_latent_width)

        # Structured Encoder
        print((1, self.num_structured_input_channel, self.structured_input_width))
        self.structured_encoder = CnnEncoder(self.num_structured_input_channel, 
                                             self.structured_input_width,
                                             self.layer_params)

        # Calculate final structured output width after encoder
        # get dims for latent shared concatenated layer
        struct_outshape = utils.conv1d_sequential_outshape(self.structured_encoder.cnn_layers, 
                                                           self.num_structured_input_channel, 
                                                           self.structured_input_width)

        self.combined_latent_width = struct_outshape[1] * struct_outshape[2] + self.unstructured_latent_width
        if self.latent_layer_type == "CNN":
            self.latent_layer = LatentCNN(struct_outshape)
        elif self.latent_layer_type == "GAUSS":
            self.latent_layer = LatentGauss(self.combined_latent_width)
        elif self.latent_layer_type == "DENSE":
            self.latent_layer = LatentDense(self.combined_latent_width)
        else:
            raise ValueError("""Must set latent_layer_type to either CNN, GAUSS or DENSE""")

        # Unstructured Decoder
        self.unstructured_decoder = DenseDecoder(self.combined_latent_width,
                                                 self.unstructured_input_width)

        # Structured Decoder
        self.reshaped_shared_latent_width = self.combined_latent_width // struct_outshape[1]
        self.structured_decoder = CnnDecoder(self.structured_encoder.conv_out_width,
                                             self.reshaped_shared_latent_width,
                                             self.num_structured_input_channel,
                                             self.structured_input_width,
                                             self.layer_params)

    def forward(self, data: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        # data is a tuple (structured, unstructured)
        
        # Encode
        structured_encoded_x   = self.structured_encoder(data[0]) # (nbatch, nchannels, out_width)
        unstructured_encoded_x = self.unstructured_encoder(data[1]) # (nbatch, out_width)

        # Combine Latents
        flat_structured_encoded_x = structured_encoded_x.flatten(start_dim=1)
        combined_latent           = torch.cat((flat_structured_encoded_x, unstructured_encoded_x), dim=1)

        reshaped_shared_latent = combined_latent.view(-1, self.num_structured_latent_channels, 
                                                          self.reshaped_shared_latent_width)
        
        if self.latent_layer_type == "CNN":
            shared_latent_out      = self.latent_layer(reshaped_shared_latent)
            structured_decoded_x   = self.structured_decoder(shared_latent_out)
            unstructured_decoded_x = self.unstructured_decoder(combined_latent)
            return structured_decoded_x, unstructured_decoded_x
        
        elif self.latent_layer_type == "GAUSS":
            shared_latent_out   = self.latent_layer(combined_latent)
            reshaped_latent_out = shared_latent_out.view(-1, self.num_structured_latent_channels, 
                                                               self.reshaped_shared_latent_width)
            structured_decoded_x   = self.structured_decoder(reshaped_latent_out)
            unstructured_decoded_x = self.unstructured_decoder(shared_latent_out)
            return structured_decoded_x, unstructured_decoded_x, shared_latent_out
        
        elif self.latent_layer_type == "DENSE":
            shared_latent_out   = self.latent_layer(combined_latent)
            reshaped_latent_out = shared_latent_out.view(-1, self.num_structured_latent_channels, 
                                                                self.reshaped_shared_latent_width)
            structured_decoded_x   = self.structured_decoder(reshaped_shared_latent)
            unstructured_decoded_x = self.unstructured_decoder(shared_latent_out)
            return structured_decoded_x, unstructured_decoded_x

# encoder classes
class DenseEncoder(nn.Module):
    def __init__(self, in_width, out_width):
        super().__init__()
        self.unstructured_encoder = nn.Sequential(
            nn.Linear(in_width, out_width),
            nn.ReLU()
        )
    def forward(self, x):
        return self.unstructured_encoder(x)

class CnnEncoder(nn.Module):
    def __init__(self, data_channels, data_width, layer_params):
        super().__init__()
        out_channels = layer_params['out_channels']
        kernel       = layer_params['kernel']
        stride       = layer_params['stride']
        nl = len(out_channels)
        self.cnn_layers = nn.Sequential()
        self.cnn_layers.add_module("conv1d_0", 
                                   nn.Conv1d(in_channels  = data_channels, 
                                             out_channels = out_channels[0], 
                                             kernel_size  = kernel[0], 
                                             stride       = stride[0], 
                                             padding      = 1))
        
        conv_out_shape = utils.conv1d_sequential_outshape(self.cnn_layers, 
                                                data_channels, 
                                                data_width)
        self.conv_out_width = [conv_out_shape[2]]
        print(conv_out_shape)

        # add more layers if depth greater than 1.
        if len(out_channels) > 1:
            self.cnn_layers.add_module("conv_ReLU_0", nn.ReLU())
            for i in range(1, nl):
                self.cnn_layers.add_module("conv1d_" + str(i), 
                                            nn.Conv1d(in_channels  = out_channels[i-1], 
                                                      out_channels = out_channels[i], 
                                                      kernel_size  = kernel[i], 
                                                      stride       = stride[i], 
                                                      padding      = 1))
                
                conv_out_shape = utils.conv1d_sequential_outshape(self.cnn_layers, 
                                                        data_channels, 
                                                        data_width)
                self.conv_out_width.append(conv_out_shape[2])                
                
                print(conv_out_shape)

                if i < (nl-1):
                    self.cnn_layers.add_module("conv_ReLU_" + str(i), nn.ReLU())

    def forward(self, x):
        return self.cnn_layers(x)
        

# decoder classes
class DenseDecoder(nn.Module):
    def __init__(self, in_width, out_width):
        super().__init__()
        self.unstructured_decoder = nn.Sequential(
            nn.Linear(in_width, out_width)
        )
    def forward(self, x):
        return self.unstructured_decoder(x)
    
class CnnDecoder(nn.Module):
    def __init__(self, 
                 encoder_layer_widths, 
                 latent_width, 
                 data_channels, 
                 data_width, 
                 layer_params):
        
        super().__init__()
        out_channels = layer_params['out_channels']
        kernel       = layer_params['kernel']
        stride       = layer_params['stride']
        nl = len(out_channels)

        num_cnn_latent_channels = out_channels[-1]

        self.tcnn_layers = nn.Sequential()

        # TODO: right now there has to be at least 2 conv layers. 
        # Implement this so one conv layer is possible
        if nl == 1:
            pass
        elif nl == 2:
            pass
        else:
            pass

        self.tcnn_layers.add_module("trans_conv1d_0", 
                                    nn.ConvTranspose1d(
                                        in_channels  = out_channels[nl-1], 
                                        out_channels = out_channels[nl-2], 
                                        kernel_size  = kernel[nl-1], 
                                        stride       = stride[nl-1], 
                                        padding      = 0,
                                        output_padding = 0))
 
        self.tcnn_layers.add_module("tconv_ReLU_0", nn.ReLU())

        outshape = utils.tconv1d_sequential_outshape(self.tcnn_layers, 
                                                     num_cnn_latent_channels, 
                                                     latent_width)
        print(outshape)

        for i in range(nl-2, 0, -1):
            outshape = utils.tconv1d_sequential_outshape(self.tcnn_layers, 
                                                     num_cnn_latent_channels, 
                                                     latent_width)
            w_in = outshape[2]
            new_target_width = encoder_layer_widths[i-1]
            npad, noutpad = self._get_decode_paddings(new_target_width, 
                                                     w_in, stride[i], kernel[i])

            self.tcnn_layers.add_module("conv1d_" + str(i), 
                                        nn.ConvTranspose1d(
                                            in_channels  = out_channels[i], 
                                            out_channels = out_channels[i-1], 
                                            kernel_size  = kernel[i], 
                                            stride       = stride[i], 
                                            padding      = npad,
                                            output_padding= noutpad))
            
            print(utils.tconv1d_sequential_outshape(self.tcnn_layers, 
                                                num_cnn_latent_channels, 
                                                latent_width))

            self.tcnn_layers.add_module("tconv_ReLU_" + str(i), nn.ReLU())

        # get correct padding and output_padding for final decoder layer
        outshape = utils.tconv1d_sequential_outshape(self.tcnn_layers, 
                                                    num_cnn_latent_channels, 
                                                    latent_width)
        

        width = outshape[2]
        new_target_width = data_width
        pad, outpad = self._get_decode_paddings(new_target_width, width, stride[0], kernel[0])

        self.tcnn_layers.add_module("struct_decoder_out", 
                                    nn.ConvTranspose1d(
                                        in_channels    = out_channels[0], 
                                        out_channels   = data_channels, 
                                        kernel_size    = kernel[0], 
                                        stride         = stride[0], 
                                        padding        = pad, 
                                        output_padding = outpad))
        
        print(utils.tconv1d_sequential_outshape(self.tcnn_layers, 
                                        num_cnn_latent_channels, 
                                        latent_width))
        
    def forward(self, x):
        return self.tcnn_layers(x)

    def _get_decode_paddings(self, w_out_target, w_in, s, k, d=1):
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
    
# Latent layer classes
class LatentCNN(nn.Module):

    def __init__(self, in_cnn_shape: Tuple[int, int]):
        # creates a tensor with shape (in_cnn_shape[1], in_cnn_shape[2] + 1)
        super().__init__()
        self.shared_layer = nn.Sequential( # preserve input shape
            nn.Conv1d(in_cnn_shape[1], in_cnn_shape[1], 
                      stride = 1, kernel_size = 1, padding=0),
            nn.ReLU()
        )
    
    def forward(self, x):
        return self.shared_layer(x)
    
class LatentDense(nn.Module):
    def __init__(self, in_width: int):
        super().__init__()
        # Shared Latent Layer
        self.shared_layer = nn.Sequential(
            nn.Linear(in_width, in_width),
            nn.ReLU()
        )
    
    def forward(self, x):
        return self.shared_layer(x)
    
class LatentGauss(nn.Module):
    def __init__(self, in_width: int):
        super().__init__()
        # Shared Latent Layer
        self.shared_layer = nn.Sequential(
            nn.Linear(in_width, in_width),
        )
    
    def forward(self, x):
        return self.shared_layer(x)

