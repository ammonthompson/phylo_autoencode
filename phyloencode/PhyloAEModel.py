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
                 num_chars = 0,
                 char_type = "categorical", # categorical, continuous
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

        self.char_type = char_type
        self.num_chars = num_chars
        
        # check that num_chars and num_channels are compatible
        if num_chars > 0 and num_structured_input_channel <= 2:
            self.num_chars = 0
            Warning("""num_chars > 0 but data_channels <= 2.
                                num_structured_input_channel must be greater
                                than 2 to use character data.
                          Setting num_chars to 0""")

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
            if self.latent_layer_type == "CNN":
                self.unstructured_latent_width = self.num_structured_latent_channels
            else: 
                self.unstructured_latent_width = 10

        # Validate divisibility of the two latent sizes if CNN latent layer is used
        if self.latent_layer_type == "CNN":
            if self.unstructured_latent_width % self.num_structured_latent_channels != 0:
                raise ValueError("""unstructured_latent_width must be an integer 
                                 multiple of num_structured_latent_channels""")
        
       
        ######################################
        # CREATE ENCODER AND DECODER LAYERS ##
        ######################################

        # Unstructured Encoder
        self.unstructured_encoder = DenseEncoder(self.unstructured_input_width, 
                                                 self.unstructured_latent_width)

        # Structured Encoder
        print("Structured autoencoder shapes:")
        print((1, self.num_structured_input_channel, self.structured_input_width))
        self.structured_encoder = CnnEncoder(self.num_structured_input_channel, 
                                             self.structured_input_width,
                                             self.layer_params)

        # Calculate final structured output width after encoder
        # get dims for latent shared concatenated layer
        # self.struct_outshape = utils.conv1d_sequential_outshape(self.structured_encoder.cnn_layers, 
        #                                                         self.num_structured_input_channel, 
        #                                                         self.structured_input_width)
        self.struct_outshape = utils.get_outshape(self.structured_encoder.cnn_layers, 
                                                  self.num_structured_input_channel, 
                                                  self.structured_input_width)
        self.struct_outwidth = self.struct_outshape[2]

        self.flat_struct_outwidth = self.struct_outshape[1] * self.struct_outshape[2]
        
        self.combined_latent_inwidth = self.flat_struct_outwidth + self.unstructured_latent_width

        if self.latent_layer_type == "CNN":
            self.latent_layer = LatentCNN(self.struct_outshape, kernel_size = self.layer_params['kernel'][-1])
            self.reshaped_shared_latent_width = self.combined_latent_inwidth // self.struct_outshape[1]
            self.latent_outwidth = self.combined_latent_inwidth

        elif self.latent_layer_type == "GAUSS":
            self.latent_layer = LatentGauss(self.combined_latent_inwidth, self.flat_struct_outwidth)
            self.reshaped_shared_latent_width = self.struct_outshape[2]
            self.latent_outwidth = self.flat_struct_outwidth

        elif self.latent_layer_type == "DENSE":
            self.latent_layer = LatentDense(self.combined_latent_inwidth, self.flat_struct_outwidth)
            self.reshaped_shared_latent_width = self.struct_outshape[2]
            self.latent_outwidth = self.flat_struct_outwidth

        else:
            raise ValueError("""Must set latent_layer_type to either CNN, GAUSS or DENSE""")

        self.unstructured_decoder = DenseDecoder(self.latent_outwidth, self.unstructured_input_width)

        # Structured Decoder
        self.structured_decoder = CnnDecoder(encoder_layer_widths = self.structured_encoder.conv_out_width,
                                             latent_width         = self.reshaped_shared_latent_width,
                                             data_channels        = self.num_structured_input_channel,
                                             data_width           = self.structured_input_width,
                                             layer_params         = self.layer_params,
                                             num_chars            = self.num_chars,
                                             char_type            = self.char_type)
        
    

    def forward(self, data: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:

        ''' 
        Push data through the encoder and decoder networks.
        '''

        # data is a tuple (structured, unstructured)

        # Encode
        structured_encoded_x   = self.structured_encoder(data[0]) # (nbatch, nchannels, out_width)
        unstructured_encoded_x = self.unstructured_encoder(data[1]) # (nbatch, out_width)

        # TODO: THe latent Classes should be able to handle the reshaping of the data
        # Combine encodings and reshape depending on the latent layer type
        flat_structured_encoded_x = structured_encoded_x.flatten(start_dim=1)
        # combined_latent           = torch.cat((flat_structured_encoded_x, unstructured_encoded_x), dim=1)
        combined_latent           = torch.cat((flat_structured_encoded_x, unstructured_encoded_x), dim=1)  # aux latent last

        
        if self.latent_layer_type == "CNN":
            reshaped_shared_latent = combined_latent.view(-1, self.num_structured_latent_channels, 
                                                              self.reshaped_shared_latent_width)
            shared_latent_out      = self.latent_layer(reshaped_shared_latent)
            structured_decoded_x   = self.structured_decoder(shared_latent_out)
            unstructured_decoded_x = self.unstructured_decoder(shared_latent_out.flatten(start_dim=1))
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
            structured_decoded_x   = self.structured_decoder(reshaped_latent_out)
            unstructured_decoded_x = self.unstructured_decoder(shared_latent_out)
            return structured_decoded_x, unstructured_decoded_x


# encoder classes
class DenseEncoder(nn.Module):
    def __init__(self, in_width, out_width):
        super().__init__()
        self.unstructured_encoder = nn.Sequential(
            nn.Linear(in_width, 10),
            nn.BatchNorm1d(10),# experimenting
            nn.ReLU(),
            nn.Linear(10, out_width),
            nn.BatchNorm1d(out_width)# experimenting
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
        
        # conv_out_shape = utils.conv1d_sequential_outshape(self.cnn_layers, 
        #                                         data_channels, 
        #                                         data_width)
        conv_out_shape = utils.get_outshape(self.cnn_layers, data_channels, data_width)
        
        self.conv_out_width = [conv_out_shape[2]]
        print(conv_out_shape)

        # experimenting
        self.cnn_layers.add_module("cnn_batchnorm_0", nn.BatchNorm1d(out_channels[0]))

        # add more layers if depth greater than 1.
        if len(out_channels) > 1:
            self.cnn_layers.add_module("conv_ReLU_0", nn.ReLU())
            # self.cnn_layers.add_module("conv_GELU_0", nn.GELU())
            # self.cnn_layers.add_module("conv_leakyrelu_0", nn.LeakyReLU(0.2))
            
            for i in range(1, nl):
                self.cnn_layers.add_module("conv1d_" + str(i), 
                                            nn.Conv1d(in_channels  = out_channels[i-1], 
                                                      out_channels = out_channels[i], 
                                                      kernel_size  = kernel[i], 
                                                      stride       = stride[i], 
                                                      padding      = 1))
                
                # conv_out_shape = utils.conv1d_sequential_outshape(self.cnn_layers, 
                #                                         data_channels, 
                #                                         data_width)
                conv_out_shape = utils.get_outshape(self.cnn_layers, data_channels, data_width)
                self.conv_out_width.append(conv_out_shape[2])                
                
                print(conv_out_shape)

                self.cnn_layers.add_module("cnn_batchnorm_" + str(i),
                                            nn.BatchNorm1d(out_channels[i])) # experimenting

                if i < (nl-1):
                    self.cnn_layers.add_module("conv_ReLU_" + str(i), nn.ReLU())
                    # self.cnn_layers.add_module("conv_GELU_" + str(i), nn.GELU())
                    # self.cnn_layers.add_module("conv_leakyrelu_" + str(i), nn.LeakyReLU(0.2))

    def forward(self, x):
        return self.cnn_layers(x)
        

# decoder classes
class DenseDecoder(nn.Module):
    def __init__(self, in_width, out_width):
        super().__init__()
        self.unstructured_decoder = nn.Sequential(
            nn.Linear(in_width, 10),
            nn.ReLU(),
            nn.Linear(10, out_width)
        )
    def forward(self, x):
        return self.unstructured_decoder(x)
    
class CnnDecoder(nn.Module):
    def __init__(self, 
                 encoder_layer_widths, 
                 latent_width, 
                 data_channels, 
                 data_width, 
                 layer_params,
                 num_chars,
                 char_type):
        
        super().__init__()

        # Set up decoder dimensions
        out_channels = layer_params['out_channels']
        kernel       = layer_params['kernel']
        stride       = layer_params['stride']
        num_cnn_latent_channels = out_channels[-1]
        self.char_type          = char_type
        self.num_chars          = num_chars
        self.data_channels      = data_channels
        nl = len(out_channels)


        # TODO: right now there has to be at least 2 conv layers. 
        # Implement this so one conv layer is possible
        if nl == 1:
            pass
        elif nl == 2:
            pass
        else:
            pass

        # get correct padding and output_padding for first decoder layer
        w_in = latent_width
        new_target_width = encoder_layer_widths[-2]
        npad, noutpad = self._get_paddings(new_target_width, w_in, stride[-1], kernel[-1])

        print((1, out_channels[nl-1], latent_width))
        ########################
        # build decoder layers #
        ########################
        self.tcnn_layers = nn.Sequential()        
        self.tcnn_layers.add_module("trans_conv1d_0", 
                                    nn.ConvTranspose1d(
                                        in_channels     = out_channels[nl-1], 
                                        out_channels    = out_channels[nl-2], 
                                        kernel_size     = kernel[nl-1], 
                                        stride          = stride[nl-1], 
                                        padding         = npad,
                                        output_padding  = noutpad 
                                        ))

 
        self.tcnn_layers.add_module("tconv_ReLU_0", nn.ReLU())

        # outshape = utils.tconv1d_sequential_outshape(self.tcnn_layers, 
        #                                              num_cnn_latent_channels, 
        #                                              latent_width)
        outshape = utils.get_outshape(self.tcnn_layers,  num_cnn_latent_channels, latent_width)
        print(outshape)

        for i in range(nl-2, 0, -1):
            # outshape = utils.tconv1d_sequential_outshape(self.tcnn_layers, 
            #                                          num_cnn_latent_channels, 
            #                                          latent_width)
            outshape = utils.get_outshape(self.tcnn_layers, num_cnn_latent_channels, latent_width)
            w_in = outshape[2]
            new_target_width = encoder_layer_widths[i-1]
            npad, noutpad = self._get_paddings(new_target_width, w_in, stride[i], kernel[i])

            self.tcnn_layers.add_module("conv1d_" + str(i), 
                                        nn.ConvTranspose1d(
                                            in_channels     = out_channels[i], 
                                            out_channels    = out_channels[i-1], 
                                            kernel_size     = kernel[i], 
                                            stride          = stride[i], 
                                            padding         = npad,
                                            output_padding  = noutpad
                                            ))
            
            # print(utils.tconv1d_sequential_outshape(self.tcnn_layers, 
            #                                     num_cnn_latent_channels, 
            #                                     latent_width))
            print(utils.get_outshape(self.tcnn_layers, num_cnn_latent_channels, latent_width))

            self.tcnn_layers.add_module("tconv_ReLU_" + str(i), nn.ReLU())

        # get correct padding and output_padding for final decoder layer
        # outshape = utils.tconv1d_sequential_outshape(self.tcnn_layers, num_cnn_latent_channels, latent_width)
        outshape = utils.get_outshape(self.tcnn_layers, num_cnn_latent_channels, latent_width)
        

        # final decoder layer
        width = outshape[2]
        new_target_width = data_width
        pad, outpad = self._get_paddings(new_target_width, width, stride[0], kernel[0])

        self.tcnn_layers.add_module("struct_decoder_out", 
                                    nn.ConvTranspose1d(
                                        in_channels    = out_channels[0], 
                                        out_channels   = data_channels, 
                                        kernel_size    = kernel[0], 
                                        stride         = stride[0], 
                                        padding        = pad, 
                                        output_padding = outpad
                                        ))
        
        # self.tcnn_layers.add_module("tconv_ReLU_out", nn.ReLU())
        # self.tcnn_layers.add_module("tconv_softclip_out", utils.SoftClip())
        # self.tcnn_layers.add_module("tconv_sigmoid_out", nn.Sigmoid())
        # self.tcnn_layers.add_module("tconv_softmax_out", nn.Softmax(dim=1))
        # self.tcnn_layers.add_module("tconv_Tanh_out", nn.Tanh())



        # print out shape
        # print(utils.tconv1d_sequential_outshape(self.tcnn_layers, num_cnn_latent_channels, latent_width))
        print(utils.get_outshape(self.tcnn_layers, num_cnn_latent_channels, latent_width))

    def forward(self, x):
        decoded_x = self.tcnn_layers(x)
        char_start_idx = decoded_x.shape[1] - self.num_chars

        # decoded_phylo = torch.sigmoid(decoded_x[:, :char_start_idx, :])
        decoded_phylo =decoded_x[:, :char_start_idx, :]

        # If has categorical character data, add a logistic layer
        # concatenate categorical output to phhylo output        
        if self.char_type == "categorical" and self.num_chars > 0 and self.data_channels > 2:
            decoded_cat = decoded_x[:, char_start_idx:, :]
            final_decoded_x = torch.cat((decoded_phylo, decoded_cat), dim = 1)
        else:
            final_decoded_x = torch.cat((decoded_phylo, decoded_x[:, char_start_idx:, :]), dim = 1)

        return final_decoded_x

    def _get_paddings(self, w_out_target, w_in, s, k, d=1):
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
# TODO: Should probably remove LatentDense and LatentCNN and just use LatentGauss
class LatentCNN(nn.Module):
    def __init__(self, in_cnn_shape: Tuple[int, int], kernel_size = 9):
        # creates a tensor with shape (in_cnn_shape[1], in_cnn_shape[2] + 1)
        super().__init__()
        odd_kernel = kernel_size - kernel_size % 2 + 1
        self.shared_layer = nn.Sequential( # preserve input shape
            nn.Conv1d(in_cnn_shape[1], in_cnn_shape[1], 
                      stride = 1, kernel_size = odd_kernel, 
                      padding = (odd_kernel-1)//2)
        )
    
    def forward(self, x):
        return self.shared_layer(x)
    
class LatentDense(nn.Module):
    def __init__(self, in_width: int, out_width: int):
        super().__init__()
        # Shared Latent Layer
        self.shared_layer = nn.Sequential(
            nn.Linear(in_width, out_width)
        )
        print((1, out_width))
    
    def forward(self, x):
        return self.shared_layer(x)
    
class LatentGauss(nn.Module):
    def __init__(self, in_width: int, out_width: int):
        super().__init__()
        # Shared Latent Layer
        self.shared_layer = nn.Sequential(
            nn.Linear(in_width, out_width)
        )
        print((1, out_width))
    
    def forward(self, x):
        return self.shared_layer(x)

