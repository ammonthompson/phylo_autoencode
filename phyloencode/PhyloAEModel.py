import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
from phyloencode import utils

class AECNN(nn.Module):
    """ This class uses a CNN and a dense layer to encode structured and unstructured data respectively
        The encodings are are concatenated in a latent layer which then gets decoded by a transpose CNN 
        and dense layer in parallel.
    """


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
                 latent_output_dim = None, # if None, then controled by structured latent channels
                 latent_layer_type = "CNN",     # CNN, DENSE, GAUSS
                 out_prefix = "out"):         
        """Constructor sets up network 

        Args:
            num_structured_input_channel (int): e.g. at least 2 for cblv formated trees
            structured_input_width (int): e.g. num tips for trees
            unstructured_latent_width (int, optional): number of auxiliary statistics or metadata. Defaults to None.
            num_chars (int, optional): number of channels that are characters (< num_chars). defaults to 0.
            char_type (str, optional): options are ["categorical", "continuous"]; int or float. Defaults to "categorical".
            stride (list, optional): stride at each CNN layer for structured data encoder. Defaults to [2,2].
            kernel (list, optional): kernel width for each layer for structured data encoder. Defaults to [3,3].
            out_channels (list, optional): number of channels for output for each layer for structured data encoder. Defaults to [16, 32].
            latent_output_dim (int, optional): Latent encoding dimension. Defaults to None.
            latent_layer_type (str, optional): options are ["CNN", "DENSE", "GAUSS"]. Defaults to "CNN".
            out_prefix (str, optional): prefix for output files. Defaults to "out".

        Raises:
            ValueError: if the stride array length does not equal the kernel array length
            ValueError: _description_
            ValueError: _description_
            Warning: num_chars > 0 but data_channesl <= 2. num_chars set to zero automatically.
            ValueError: _description_
        """
        
        # assumptions:
        # inputs are standardized
        # at least 2 convolution layers
        # all strides > 1
        # kernal[i] > stride[i]

        super().__init__()

        self.char_type = char_type
        self.num_chars = num_chars
        
        # check that num_chars and num_channels are compatible
        # TODO: fix this
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
        # if np.min(stride) <= 1:
        #     raise ValueError("All stride values must be greater than 1.")

        # Ensure kernel values are greater than stride values
        # stride_arr, kernel_arr = np.array(stride), np.array(kernel)
        # if np.min(kernel_arr - stride_arr) <= 0:
        #     raise ValueError("Each kernel value must be greater than the corresponding stride value.")

        # convolution layers parameters
        self.layer_params = {"out_channels": out_channels,
                            "kernel"      : kernel,
                            "stride"      : stride}
                
        self.latent_layer_type = latent_layer_type
        self.latent_layer_dim = latent_output_dim
        

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

        # TODO: nn.adaptivepooling might be better than flattening cnn encoder outputs before latent layers

        # Calculate final structured output width after encoder
        # get dims for latent shared concatenated layer
        self.struct_outshape = utils.get_outshape(self.structured_encoder.cnn_layers, 
                                                  self.num_structured_input_channel, 
                                                  self.structured_input_width)
        self.struct_outwidth = self.struct_outshape[2]

        self.flat_struct_outwidth = self.struct_outshape[1] * self.struct_outshape[2]
        
        self.combined_latent_inwidth = self.flat_struct_outwidth + self.unstructured_latent_width

        if self.latent_layer_type == "CNN":
            self.latent_layer = LatentCNN(self.struct_outshape, kernel_size = self.layer_params['kernel'][-1])
            self.reshaped_shared_latent_width = self.combined_latent_inwidth // self.struct_outshape[1]
            if self.latent_layer_dim is not None:
                raise Warning("""latent_layer_dim is set but not used.
                                Latent layer type is CNN, so reshaped_shared_latent_width is set to 
                                combined_latent_inwidth // num_structured_latent_channels""")
            self.latent_outwidth = self.combined_latent_inwidth
            self.latent_layer_decoder = LatentCNNDecoder(self.struct_outshape, kernel_size = self.layer_params['kernel'][-1])

        elif self.latent_layer_type == "GAUSS":
            self.reshaped_shared_latent_width = self.struct_outshape[2]
            self.latent_outwidth = self.flat_struct_outwidth if self.latent_layer_dim is None else self.latent_layer_dim
            self.latent_layer = LatentGauss(self.combined_latent_inwidth, self.latent_outwidth)
            self.latent_layer_decoder = LatentDenseDecoder(self.latent_outwidth, self.flat_struct_outwidth)

        elif self.latent_layer_type == "DENSE":
            self.reshaped_shared_latent_width = self.struct_outshape[2]
            self.latent_outwidth = self.flat_struct_outwidth if self.latent_layer_dim is None else self.latent_layer_dim
            self.latent_layer = LatentDense(self.combined_latent_inwidth, self.latent_outwidth)
            self.latent_layer_decoder = LatentDenseDecoder(self.latent_outwidth, self.flat_struct_outwidth)

        else:
            raise ValueError("""Must set latent_layer_type to either CNN, GAUSS or DENSE""")

        self.unstructured_decoder = DenseDecoder(self.flat_struct_outwidth, self.unstructured_input_width)

        # Structured Decoder
        self.structured_decoder = CnnDecoder(encoder_layer_widths = self.structured_encoder.conv_out_width,
                                             latent_width         = self.reshaped_shared_latent_width,
                                             data_channels        = self.num_structured_input_channel,
                                             data_width           = self.structured_input_width,
                                             layer_params         = self.layer_params,
                                             num_chars            = self.num_chars,
                                             char_type            = self.char_type)
        

        self.write_network_to_file(out_prefix + ".network.txt")
        
 
    def forward(self, data: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """        
        Push data through the encoder and decoder networks.

        Args:
            data (Tuple[torch.Tensor, torch.Tensor]): _description_

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: _description_
        """
        

        # data is a tuple (structured, unstructured)

        # Encode
        structured_encoded_x   = self.structured_encoder(data[0]) # (nbatch, nchannels, out_width)
        unstructured_encoded_x = self.unstructured_encoder(data[1]) # (nbatch, out_width)

        # TODO: THe latent Classes should be able to handle the reshaping of the data
        # Combine encodings and reshape depending on the latent layer type
        flat_structured_encoded_x = structured_encoded_x.flatten(start_dim=1)
        combined_latent           = torch.cat((flat_structured_encoded_x, unstructured_encoded_x), dim=1)  # aux latent last

        
        if self.latent_layer_type == "CNN":
            reshaped_shared_latent = combined_latent.view(-1, self.num_structured_latent_channels, 
                                                              self.reshaped_shared_latent_width)
            shared_latent_out      = self.latent_layer(reshaped_shared_latent)
            structured_decoded_x   = self.structured_decoder(shared_latent_out)
            unstructured_decoded_x = self.unstructured_decoder(shared_latent_out.flatten(start_dim=1))
        
        elif self.latent_layer_type == "GAUSS":
            shared_latent_out   = self.latent_layer(combined_latent)
           
            # testings
            decoded_shared_latent_out = self.latent_layer_decoder(shared_latent_out)

            reshaped_decoded_shared_latent_out = decoded_shared_latent_out.view(-1, self.num_structured_latent_channels, 
                                                                                    self.reshaped_shared_latent_width) 
            structured_decoded_x   = self.structured_decoder(reshaped_decoded_shared_latent_out)
            unstructured_decoded_x = self.unstructured_decoder(decoded_shared_latent_out)
        
        elif self.latent_layer_type == "DENSE":
            shared_latent_out   = self.latent_layer(combined_latent)            
            reshaped_latent_out = shared_latent_out.view(-1, self.num_structured_latent_channels, 
                                                                self.reshaped_shared_latent_width)
            structured_decoded_x   = self.structured_decoder(reshaped_latent_out)
            unstructured_decoded_x = self.unstructured_decoder(shared_latent_out)

        # separate the two type of structured decoded: tree and character data
        # maybe this should be done somewhere more downstream
        if self.num_chars > 0:
            phy_decoded_x = structured_decoded_x[:,:(structured_decoded_x.shape[1]-self.num_chars),:]
            char_decoded_x = structured_decoded_x[:,(structured_decoded_x.shape[1]-self.num_chars):,:]
        else:
            phy_decoded_x = structured_decoded_x
            char_decoded_x = None
        # model should output the latent layer if layer type is "GAUSS"
        if self.latent_layer_type != "GAUSS":
            shared_latent_out = None

        return phy_decoded_x, char_decoded_x, unstructured_decoded_x, shared_latent_out

    def write_network_to_file(self, out_fn) -> None:
        """Write network architecture to simple text file.

        Args:
            out_fn (_type_): _description_
        """

        with open(out_fn, "w") as f:  
            f.write("PHYLOGENETIC ENCODER AND DECODER:\n")    
            f.write(str(self.structured_encoder) + "\n")
            f.write(str(self.latent_layer) + "\n")
            f.write(str(self.latent_layer_decoder) + "\n")
            f.write(str(self.structured_decoder) + "\n")
            f.write("\n\nAUXILLIARY ENCODER AND DECODER:\n")
            f.write(str(self.unstructured_encoder) + "\n")
            f.write("\nSee latent layers above.\n")
            f.write(str(self.unstructured_decoder))

# encoder classes
class DenseEncoder(nn.Module):
    def __init__(self, in_width, out_width):
        super().__init__()
        self.unstructured_encoder = nn.Sequential(
            nn.Linear(in_width, 10),
            nn.BatchNorm1d(10),
            nn.ReLU(),
            nn.Linear(10, out_width),
            nn.BatchNorm1d(out_width)
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
                                             padding      = kernel[0]// 2))  # padding = kernel_size // 2 for same padding
                                        
        conv_out_shape = utils.get_outshape(self.cnn_layers, data_channels, data_width)
        self.conv_out_width = [conv_out_shape[2]]
        print(conv_out_shape)

        # experimenting
        self.cnn_layers.add_module("norm_0", nn.BatchNorm1d(out_channels[0]))

        # add more layers if depth greater than 1.
        if len(out_channels) > 1:
            self.cnn_layers.add_module("conv_ReLU_0", nn.ReLU())
            
            for i in range(1, nl):
                self.cnn_layers.add_module("conv1d_" + str(i), 
                                            nn.Conv1d(in_channels  = out_channels[i-1], 
                                                      out_channels = out_channels[i], 
                                                      kernel_size  = kernel[i], 
                                                      stride       = stride[i], 
                                                      padding      = kernel[i]// 2))                                                     
                
                conv_out_shape = utils.get_outshape(self.cnn_layers, data_channels, data_width)
                self.conv_out_width.append(conv_out_shape[2])  # bookkeeping           
                print(conv_out_shape)
                self.cnn_layers.add_module("norm_" + str(i), nn.BatchNorm1d(out_channels[i]))

                if i < (nl-1):
                    self.cnn_layers.add_module("conv_ReLU_" + str(i), nn.ReLU())


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


        ########################
        # build decoder layers #
        ########################

        # get correct padding and output_padding for first decoder layer
        w_in = latent_width
        new_target_width = encoder_layer_widths[-2]
        npad, noutpad = self._get_paddings(new_target_width, w_in, stride[-1], kernel[-1])
        print((1, out_channels[nl-1], latent_width))
        
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
 
        outshape = utils.get_outshape(self.tcnn_layers,  num_cnn_latent_channels, latent_width)
        print(outshape)

        if (nl - 2) > 0:
            self.tcnn_layers.add_module("tconv_norm_0", nn.BatchNorm1d(out_channels[nl-2]))


        self.tcnn_layers.add_module("tconv_ReLU_0", nn.ReLU())        

        for i in range(nl-2, 0, -1):
            outshape = utils.get_outshape(self.tcnn_layers, num_cnn_latent_channels, latent_width)
            w_in = outshape[2]
            new_target_width = encoder_layer_widths[i-1]
            npad, noutpad = self._get_paddings(new_target_width, w_in, stride[i], kernel[i])
            self.tcnn_layers.add_module("tconv1d_" + str(nl - i - 1), 
                                        nn.ConvTranspose1d(
                                            in_channels     = out_channels[i], 
                                            out_channels    = out_channels[i-1], 
                                            kernel_size     = kernel[i], 
                                            stride          = stride[i], 
                                            padding         = npad,
                                            output_padding  = noutpad
                                            ))  
                      
            print(utils.get_outshape(self.tcnn_layers, num_cnn_latent_channels, latent_width))

            if i > 1:
                self.tcnn_layers.add_module("tconv_norm_" + str(nl-i-1), 
                                            nn.BatchNorm1d(out_channels[i-1]))


            self.tcnn_layers.add_module("tconv_ReLU_" + str(nl - i - 1), nn.ReLU())

        # final decoder layer
        # get correct padding and output_padding for final decoder layer
        outshape = utils.get_outshape(self.tcnn_layers, num_cnn_latent_channels, latent_width)
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
        
        # print out shape
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
# TODO: Should probably remove LatentGauss and just use LatentDense
class LatentCNN(nn.Module):
    def __init__(self, in_cnn_shape: Tuple[int, int], kernel_size = 9):
        # creates a tensor with shape (in_cnn_shape[1], in_cnn_shape[2] + 1)
        super().__init__()
        odd_kernel = kernel_size - kernel_size % 2 + 1
        self.shared_layer = nn.Sequential( # preserve input shape
            nn.Conv1d(in_cnn_shape[1], 
                      in_cnn_shape[1], 
                      stride = 1, 
                      kernel_size = odd_kernel, 
                      padding = (odd_kernel-1)//2)
        )
    
    def forward(self, x):
        return self.shared_layer(x)
    
class LatentDense(nn.Module):
    def __init__(self, in_width: int, out_width: int):
        super().__init__()
        # Shared Latent Layer
        self.shared_layer = nn.Sequential(
            nn.Linear(in_width, out_width),
        )
        print((1, out_width))
    
    def forward(self, x):
        return self.shared_layer(x)
    
class LatentGauss(nn.Module):
    def __init__(self, in_width: int, out_width: int):
        super().__init__()
        # Shared Latent Layer
        self.shared_layer = nn.Sequential(
            nn.Linear(in_width, out_width),
            # nn.BatchNorm1d(out_width),
            nn.ReLU(),
            nn.Linear(out_width, out_width),
        )
        print((1, out_width))
    
    def forward(self, x):
        return self.shared_layer(x)
    
    def __iter__(self):
        return iter(self.shared_layer)
    
class LatentDenseDecoder(nn.Module): # same as LatentGauss
    def __init__(self, in_width: int, out_width: int):
        super().__init__()
        self.shared_layer = nn.Sequential(
            # nn.Identity(),
            # nn.Linear(in_width, out_width),
            nn.BatchNorm1d(in_width),
            # nn.ELU(),
            # SoftPower(0.1, 3),
            nn.Linear(in_width, out_width),
        )
        print((1, out_width))
    
    def forward(self, x):
        return self.shared_layer(x)
    
    def __iter__(self):
        return iter(self.shared_layer)
    
class LatentCNNDecoder(nn.Module):
    def __init__(self, in_cnn_shape: Tuple[int, int], kernel_size = 9):
        super().__init__()
        odd_kernel = kernel_size - kernel_size % 2 + 1
        self.shared_layer = nn.Sequential( # preserve input shape
            nn.Conv1d(in_cnn_shape[1], 
                      in_cnn_shape[1], 
                      stride = 1, 
                      kernel_size = odd_kernel, 
                      padding = (odd_kernel-1)//2)
        )
    
    def forward(self, x):
        return self.shared_layer(x)


# TODO: maybe belong in utils.py
class SoftPower(nn.Module):
    def __init__(self, alpha = 1., power = 2):
        super().__init__()
        self.alpha = alpha
        self.power = power
    
    def forward(self, x):
        return x + self.alpha * x**self.power

class SamePadConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        self.k = kernel_size
        self.s = stride
        self.conv = nn.Conv1d( in_channels , out_channels, kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        input_len = x.shape[-1]
        out_len = (input_len + self.s - 1) // self.s  # ceil division
        print(out_len)
        total_pad = max((out_len - 1) * self.s + self.k - input_len, 0)
        pad_left = total_pad // 2
        pad_right = total_pad - pad_left
        x = nn.functional.pad(x, (pad_left, pad_right))
        return self.conv(x)