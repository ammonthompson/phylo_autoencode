import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
from phyloencode import utils
from phyloencode.ResNet import ResCnnEncoder, ResCnnDecoder
import random
import sklearn

class AECNN(nn.Module):
    """ This class uses a CNN and a dense layer to encode structured and unstructured data respectively
        The encodings are concatenated in a latent layer which then gets decoded by a transpose CNN 
        and dense layer in parallel.
    """

    def __init__(self, 
                 num_structured_input_channel, 
                 structured_input_width,  # Input width for structured data
                 unstructured_input_width,
                 *,
                 unstructured_latent_width = None, # must be integer multiple of num_structured_latent_channels
                 aux_numtips_idx = None,
                 num_chars = 0,
                 char_type = "categorical", # categorical, continuous, integer (TODO: integer not implemented yet)
                 stride = [2,2],
                 kernel = [3,3],
                 out_channels = [16, 32],
                 latent_output_dim = None, # if None, then controled by structured latent channels
                 latent_layer_type = "CNN",     # CNN, DENSE, GAUSS
                 out_prefix = "out",
                 device = "auto",
                 seed = None,
                 phy_normalizer = None,
                 aux_normalizer = None):
                 
        """Constructor sets up all layers needed for network.  

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

        super().__init__()

        self.set_seed(seed=seed)

        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device


        self.char_type = char_type
        self.num_chars = num_chars
        self.aux_numtips_idx = aux_numtips_idx

        self.char_start_idx = num_structured_input_channel - self.num_chars
        
        # check that num_chars and num_channels are compatible
        # TODO: fix this
        if num_chars > 0 and num_structured_input_channel <= 2:
            self.num_chars = 0
            Warning("""num_chars > 0 but data_channels <= 2.
                                num_structured_input_channel must be greater
                                than 2 to use character data.
                          Setting num_chars to 0""")

        if not (len(stride) == len(kernel) == len(out_channels)):
            raise ValueError("stride, kernel  and out_channels array lengths should all be equal," + 
                             f" but got lengths {len(stride)}, {len(kernel)}, {len(out_channels)}.")

        # normalizers; sklearn.base.BaseEstimator
        # if (not isinstance(phy_normalizer, utils.StandardScalerPhyCategorical) or 
        #     not isinstance(aux_normalizer, sklearn.preprocessing.StandardScaler) ):
        #     raise TypeError("Normalizers need to be StandardScalerPhyCategorical and StandardScaler")
        self.phy_normalizer = phy_normalizer
        self.aux_normalizer = aux_normalizer

        # for ntips sigmoid layer in unstructured decoder (keep num tips within data bounds)
        ntip_mu = self.aux_normalizer.mean_[aux_numtips_idx]
        ntip_sd = np.sqrt(self.aux_normalizer.var_[aux_numtips_idx])
        min_width = 2.
        max_width = structured_input_width
        self.ntip_base = (min_width - ntip_mu) / ntip_sd
        self.ntip_scale = (max_width - min_width) / ntip_sd

        # for cblv phy decoder sigmoid layer (or something else). Data range is in [0,1].
        # transformed range is in [-mu/s, (1-mu)/s]
        # find base and scale to in transformed data to guarentee inverse_transformed stays in bounds.
        phy_mean = self.phy_normalizer.mean_.reshape((num_structured_input_channel - self.num_chars, 
                                                                        max_width), order = "F")
        phy_sd   = self.phy_normalizer.std_.reshape((num_structured_input_channel - self.num_chars, 
                                                                        max_width), order = "F")

        phy_lower_bound = torch.Tensor(-phy_mean / phy_sd).to(self.device)
        phy_upper_bound = torch.Tensor((1 - phy_mean) / phy_sd).to(self.device)

        self.phy_two_sided_ReLU = TwoSidedReLU(phy_lower_bound, phy_upper_bound)


        ######################################
        # Preliminaries for layer dimensions #
        ######################################

        # convolution layers parameters
        self.layer_params = {"out_channels": out_channels,
                            "kernel"      : kernel,
                            "stride"      : stride,
                            "latent_dim"  : latent_output_dim}
                
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
        
        # Calculate final structured output width after encoder
        # get shape for latent shared concatenated layer
        # TODO: utils.get_outshape here is just used to get the encoder out_width
        #       alternatively, this can be controled by using adaptive avg pooling's out_size
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

        elif self.latent_layer_type in {"GAUSS", "DENSE"}:
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

        self.to(self.device)
        
 
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
        latent = self.encode(data[0], data[1])
        # Decode
        structured_decoded_x, unstructured_decoded_x = self.decode(latent)

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
            latent = None

        return phy_decoded_x, char_decoded_x, unstructured_decoded_x, latent


    def encode(self, phy: torch.Tensor, aux: torch.Tensor, *,
               inference = False, detach = False ) -> torch.Tensor:
        """_summary_

        Args:
            phy (torch.Tensor): _description_
            aux (torch.Tensor): _description_
            inference (bool, optional): _description_. Defaults to False.
            detach (bool, optional): _description_. Defaults to False.

        Returns:
            torch.Tensor: _description_
        """
        
        phy = phy.to(self.device)
        aux = aux.to(self.device)

        is_training = self.training
        if inference:
            self.eval()
            grad_context = torch.no_grad()
        else:
            grad_context = torch.enable_grad()

        try:        
            with grad_context:
                # get latent unstrcutured and structured embeddings
                structured_encoded_x   = self.structured_encoder(phy)    # (N, nchannels, out_width)
                unstructured_encoded_x = self.unstructured_encoder(aux)  # (N, out_width)

                # reshape structured embeddings
                flat_structured_encoded_x = structured_encoded_x.flatten(start_dim=1)                
                combined_latent = torch.cat((flat_structured_encoded_x, unstructured_encoded_x), dim=1)
        
                # get combined latent output
                if self.latent_layer_type   == "CNN":
                    reshaped_shared_latent = combined_latent.view(-1, self.num_structured_latent_channels, 
                                                                    self.reshaped_shared_latent_width)
                    shared_latent_out = self.latent_layer(reshaped_shared_latent)

                elif self.latent_layer_type in {"GAUSS", "DENSE"}:
                    shared_latent_out = self.latent_layer(combined_latent)

            if inference and detach:
                shared_latent_out = shared_latent_out.detach()
                self.train(is_training)

            return(shared_latent_out.flatten(start_dim=1))
        
        finally:
            # restore original mode
            self.train(is_training)
    
    
    def decode(self, z: torch.Tensor, *,
               inference = False, detach = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """_summary_

        Args:
            z (torch.Tensor): _description_
            inference (bool, optional): _description_. Defaults to False.
            detach (bool, optional): _description_. Defaults to False.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: _description_
        """
        
        is_training = self.training
        z = z.to(self.device)

        if inference:
            self.eval()
            grad_context = torch.no_grad()
        else:
            grad_context = torch.enable_grad()

        try:
            with grad_context:
                decoded_latent_layer = self.latent_layer_decoder(z)
                # reshape
                reshaped_decoded_latent_layer = decoded_latent_layer.view(-1, self.num_structured_latent_channels, 
                                                                                self.reshaped_shared_latent_width) 
                decoded_tree = self.structured_decoder(reshaped_decoded_latent_layer)
                decoded_aux  = self.unstructured_decoder(decoded_latent_layer.flatten(start_dim=1))

                char_idx = self.char_start_idx
                tip_idx  = self.aux_numtips_idx

                # clamp phy cblv to [-mu/s, (1-mu)/s] which is [0,1] in untransformed space
                # see phyddle documentation for output format of phyddle -s F
                phy_clamp_left = self.phy_two_sided_ReLU(decoded_tree[:, :char_idx, :])
                not_phy_right  = decoded_tree[:, char_idx:, :]
                decoded_tree   = torch.cat([phy_clamp_left, not_phy_right], dim=1)

                # ensure ntips in [2, max tips].
                # TODO: look into two-sided clamp as above.
                left  = decoded_aux[:, :tip_idx]
                ntips = self.ntip_base + self.ntip_scale * torch.sigmoid(decoded_aux[:, tip_idx:tip_idx+1])
                right = decoded_aux[:, tip_idx+1:]
                decoded_aux = torch.cat([left, ntips, right], dim=1)

                if inference and self.char_type == "categorical" and self.num_chars > 0:
                    # softmax the char data
                    not_char_left      = decoded_tree[:, :char_idx, :]
                    char_softmax_right = torch.softmax(decoded_tree[:, char_idx:, :], dim = 1)
                    decoded_tree       = torch.cat([not_char_left, char_softmax_right], dim = 1)


                if inference and detach:
                    decoded_tree = decoded_tree.detach()
                    decoded_aux  = decoded_aux.detach()
                    
                return decoded_tree, decoded_aux
            
        finally:
            self.train(is_training)

    # TODO: should prob output torch.Tensors like everything else?
    def predict(self, phy: torch.Tensor, aux: torch.Tensor, *, 
                inference = False, detach = False) -> Tuple[np.ndarray, np.ndarray]:
        """_summary_

        Args:
            phy (torch.Tensor): _description_
            aux (torch.Tensor): _description_
            inference (bool, optional): _description_. Defaults to False.
            detach (bool, optional): _description_. Defaults to False.

        Returns:
            Tuple[np.ndarray, np.ndarray]: _description_
        """
        
        # pushes data through full autoencoder
        is_training = self.training
        if inference:
            self.eval()
            grad_context = torch.no_grad()
        else:
            grad_context = torch.enable_grad()
        
        try:
            with grad_context:
                phy = phy.to(self.device)
                aux = aux.to(self.device)
                # model PREDICTS here
                tree_pred, char_pred, aux_pred, _ = self((phy, aux)) 

                if char_pred is not None:
                    phy_pred =  torch.cat((tree_pred, char_pred), dim = 1)
                else:
                    phy_pred = tree_pred

                if inference and detach:
                    phy_pred = phy_pred.detach()
                    aux_pred = aux_pred.detach()

                return phy_pred.cpu().numpy(), aux_pred.cpu().numpy()
        finally:
            self.train(is_training)

    def set_normalizers(self, phy_norm, aux_norm):
        self.phy_normalizer = phy_norm
        self.aux_normalizer = aux_norm

    # inference machinery. Handles normalization too.
    def norm_and_encode(self, phy: np.array, aux: np.array) -> np.array:
        # phy shape = (N, nc, mt)
        # normalize
        # encode
        if isinstance(phy, torch.Tensor):
            phy = phy.numpy()
        if isinstance(aux, torch.Tensor):
            aux = aux.numpy()

        flat_phy = phy.reshape((phy.shape[0], -1), order = "F")
        flat_norm_phy = self.phy_normalizer.transform(flat_phy)
        norm_phy = torch.tensor(flat_norm_phy.reshape((flat_norm_phy.shape[0], 
                                          self.num_structured_input_channel, 
                                          self.structured_input_width), order = "F"), 
                                          dtype=torch.float32)
        norm_aux = torch.tensor(self.aux_normalizer.transform(aux), dtype = torch.float32)
        latent = self.encode(norm_phy, norm_aux, inference = True, detach = True)
        return latent.cpu().numpy()

    def decode_and_denorm(self, latent : np.array) -> Tuple[np.array, np.array]:
        # decode
        # inverse_transform
        # phy output shape is (N, nc, mt)
        latent = torch.tensor(latent, dtype = torch.float32)
        norm_phy, norm_aux = self.decode(latent, inference = True, detach = True)
        flat_norm_phy = norm_phy.cpu().numpy().reshape((norm_phy.shape[0], -1), order = "F")
        phy = self.phy_normalizer.inverse_transform(flat_norm_phy)
        aux = self.aux_normalizer.inverse_transform(norm_aux.cpu().numpy())
        phy = phy.reshape(norm_phy.shape, order = "F")
        return phy, aux

    def norm_predict_denorm(self, phy : np.array, aux : np.array) -> Tuple[np.array, np.array]:
        # phy shape = (N, nc, mt)
        # phy output shape = (N, nc, mt)
        return self.decode_and_denorm( self.norm_and_encode(phy, aux) )
        

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

    def set_seed(self, seed = None):
        if seed is None:
                return  # use module-level RNGs as-is

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


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
                                             padding      = kernel[0]// 2,
                                             bias         = False))  # padding = kernel_size // 2 for same padding
                                        
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
                                                      padding      = kernel[i]// 2,
                                                      bias         = i >= (nl-1)))                                                     
                
                conv_out_shape = utils.get_outshape(self.cnn_layers, data_channels, data_width)
                self.conv_out_width.append(conv_out_shape[2])  # bookkeeping           
                print(conv_out_shape)

                if i < (nl-1):
                    self.cnn_layers.add_module("norm_" + str(i), nn.BatchNorm1d(out_channels[i]))
                    self.cnn_layers.add_module("conv_ReLU_" + str(i), nn.ReLU())


        # TODO: adaptive average pooling TESTING
        # out_size = layer_params['latent_dim'] // out_channels[-1]
        # self.cnn_layers.add_module("adapt_avg_pool", torch.nn.AdaptiveAvgPool1d(output_size = out_size))
        # conv_out_shape = utils.get_outshape(self.cnn_layers, data_channels, data_width)
        # self.conv_out_width.append(conv_out_shape[2])
        # print(conv_out_shape)



    def forward(self, x):
        return self.cnn_layers(x)
        

# decoder classes
class DenseDecoder(nn.Module):
    def __init__(self, in_width, out_width):
        super().__init__()
        self.unstructured_decoder = nn.Sequential(
            # TODO: 10 probably should be controlled by a parameter...
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
                                        output_padding  = noutpad,
                                        bias            = (nl - 2) <= 0 
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
                                            output_padding  = noutpad,
                                            bias            = i <= 1
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
        
        # for character logit rescaling
        # self.phy_head_out = Head(data_channels - num_chars, data_width)
        self.char_head_out = Head(num_chars, data_width)

        
        # print out shape
        print(utils.get_outshape(self.tcnn_layers, num_cnn_latent_channels, latent_width))

    def forward(self, x):
        decoded_x = self.tcnn_layers(x)
        char_start_idx = decoded_x.shape[1] - self.num_chars

        decoded_phylo = decoded_x[:, :char_start_idx, :]
        # decoded_phylo = self.phy_head_out(decoded_x[:, :char_start_idx, :])


        # If has categorical character data, add layers for scaling for logit outputs then
        # concatenate categorical output to phhylo output        
        if self.char_type == "categorical" and self.num_chars > 0 and self.data_channels > 2:
            # decoded_char = decoded_x[:, char_start_idx:, :]
            decoded_char = self.char_head_out(decoded_x[:, char_start_idx:, :])
            # print(self.head_out.Lin[2].weight[50:51,50:60])
            final_decoded_x = torch.cat((decoded_phylo, decoded_char), dim = 1)
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

class Head(nn.Module):
    def __init__(self, c, w):
        super().__init__()
        self.c = c
        self.w = w
        self.scale = nn.Parameter(torch.ones(c * w))
        # self.head = nn.Linear(d, d)
        # self.head = nn.Sequential(nn.Linear(d, d),
        #                          nn.ReLU(),
        #                          nn.Linear(d,d))


    def forward(self, x):
        # print(torch.max(self.scale))
        flat_x = x.view((x.shape[0], self.c * self.w))
        scaled_flat_x = self.scale * flat_x
        scaled_x = scaled_flat_x.view(x.shape[0], self.c, self.w)
        return scaled_x
        # print(torch.max(self.Lin.weight))
        # return(self.head(x))
    


# Latent layer classes
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
            nn.ReLU(),
            nn.Linear(out_width, out_width),
        )
        print((1, out_width))
    
    def forward(self, x):
        return self.shared_layer(x)
    
    def __iter__(self):
        return iter(self.shared_layer)

    
# might be more work than its worth
class LatentPool(nn.Module):
    def __init__(self, 
                 struct_encoder_out_shape : Tuple[int, int], 
                 unstruct_encoder_out_width : int, 
                 latent_dim: int):
        super().__init__()
        struct_flat_width = struct_encoder_out_shape[0] * struct_encoder_out_shape[1]
        avg_pool_out_size = struct_flat_width // latent_dim
        dense_in_width = avg_pool_out_size + unstruct_encoder_out_width
        self.avg_pool_layer = nn.AdaptiveAvgPool1d(output_size=avg_pool_out_size)
        self.shared_layer = nn.Sequential(
                nn.Linear(dense_in_width, latent_dim),
            )        
        print((1, latent_dim))


    def forward(self, struct, unstruct):
        #
        avg_pool = self.avg_pool_layer(struct)
        flat_avg_pool = avg_pool.flatten(start_dim=1)                
        combined_latent = torch.cat((flat_avg_pool, unstruct), dim=1)
        return self.shared_layer(combined_latent)

    
class LatentDenseDecoder(nn.Module): # same as LatentGauss
    def __init__(self, in_width: int, out_width: int):
        super().__init__()
        self.shared_layer = nn.Sequential(
            # nn.Identity(),
            # nn.Linear(in_width, out_width),
            # nn.BatchNorm1d(in_width),
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



# Dev
class TwoSidedReLU(nn.Module):
    def __init__(self, min_val, max_val):
        # shape of min_val and max_val is the same as the cblv tensor (N, C, W)
        super().__init__()
        self.min_val = min_val
        self.max_val = max_val

    def forward(self, x):
        return torch.clamp(x, self.min_val, self.max_val)

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
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        input_len = x.shape[-1]
        out_len = (input_len + self.s - 1) // self.s  # ceil division
        print(out_len)
        total_pad = max((out_len - 1) * self.s + self.k - input_len, 0)
        pad_left = total_pad // 2
        pad_right = total_pad - pad_left
        x = nn.functional.pad(x, (pad_left, pad_right))
        return self.conv(x)
 
