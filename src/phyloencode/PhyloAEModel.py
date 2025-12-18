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
    """Autoencoder for structured phylogenetic tensors plus auxiliary features.

    The model encodes:

    - Structured input (e.g. CBLV-like tree tensors) with a 1D CNN (``CnnEncoder``).
    - Unstructured/auxiliary input with a small MLP (``DenseEncoder``).

    The two embeddings are concatenated and passed through a shared latent layer, then
    decoded back into:

    - Structured output with a 1D transposed CNN (``CnnDecoder``).
    - Auxiliary output with a small MLP (``DenseDecoder``).

    Character channels:
        If ``num_chars > 0``, the last ``num_chars`` structured channels are treated as
        "character" channels. In ``forward()``, the structured output is split into
        ``(phy_decoded, char_decoded)``.

    Normalization:
        Training typically expects inputs already normalized with the provided scalers.
        Use ``norm_and_encode()``, ``decode_and_denorm()``, or ``norm_predict_denorm()`` when
        you want the model to handle normalization/denormalization.
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
            num_structured_input_channel (int): Number of structured channels.
                For CBLV-like tree tensors this is typically ``>= 2`` (more when including
                extra features and/or character channels).
            structured_input_width (int): Structured input width (e.g., maximum number of tips).
            unstructured_input_width (int): Number of auxiliary features per sample.
            unstructured_latent_width (Optional[int]): Width of the auxiliary embedding produced
                by the unstructured encoder. If ``latent_layer_type == "CNN"``, this must be an
                integer multiple of the final structured latent channel count (``out_channels[-1]``).
                If None, defaults to ``out_channels[-1]`` for CNN latent layers, otherwise 10.
            aux_numtips_idx (Optional[int]): Column index in the auxiliary vector that contains
                the number of taxa/tips (e.g. ``num_taxa``). Required (cannot be None): used to
                constrain the decoded value to ``[2, structured_input_width]`` in normalized space.
            num_chars (int): Number of character channels in the structured input. If > 0, the
                last ``num_chars`` channels are treated as character channels. Defaults to 0.
            char_type (str): Character data type. Common values are ``"categorical"`` and
                ``"continuous"``. If ``"categorical"`` and ``inference=True``, character channels
                are softmaxed in ``decode()``. Defaults to ``"categorical"``.
            stride (List[int]): Convolution stride for each structured encoder layer. Defaults to
                ``[2, 2]``.
            kernel (List[int]): Convolution kernel size for each structured encoder layer. Defaults
                to ``[3, 3]``.
            out_channels (List[int]): Output channels for each structured encoder layer. Defaults to
                ``[16, 32]``.
            latent_output_dim (Optional[int]): Size of the shared latent vector when
                ``latent_layer_type`` is ``"DENSE"`` or ``"GAUSS"``. If None, defaults to the
                flattened structured embedding width.
            latent_layer_type (str): Shared latent layer type: ``"CNN"``, ``"DENSE"``, or ``"GAUSS"``.
                Defaults to ``"CNN"``.
            out_prefix (str): Prefix for output files written during initialization (currently the
                ``.network.txt`` architecture dump). Defaults to ``"out"``.
            device (str): ``"auto"``, ``"cpu"``, or ``"cuda"``. If ``"auto"``, selects CUDA when
                available. Defaults to ``"auto"``.
            seed (Optional[int]): Random seed for Python, NumPy, and PyTorch. Defaults to None.
            phy_normalizer: Required. Fitted scaler for structured data with ``mean_`` and ``std_``
                attributes and ``transform``/``inverse_transform`` methods (sklearn-like).
            aux_normalizer: Required. Fitted scaler for auxiliary data with ``mean_`` and ``var_``
                attributes and ``transform``/``inverse_transform`` methods (sklearn-like).

        Raises:
            ValueError: If ``stride``, ``kernel``, and ``out_channels`` lengths differ.
            ValueError: If ``latent_layer_type == "CNN"`` and ``unstructured_latent_width`` is not
                divisible by ``out_channels[-1]``.
            ValueError: If ``latent_layer_type`` is not one of ``{"CNN", "DENSE", "GAUSS"}``.
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
        """Run a full autoencoder pass: encode then decode.

        Args:
            data (Tuple[torch.Tensor, torch.Tensor]): ``(phy, aux)`` input tensors. ``phy`` has
                shape ``(N, C, W)`` and ``aux`` has shape ``(N, A)``. Inputs are moved to
                ``self.device`` inside ``encode()``.

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, Optional[torch.Tensor]]:
            ``(phy_decoded, char_decoded, aux_decoded, latent)`` where:

            - ``phy_decoded``: Phylogeny reconstruction excluding character channels.
            - ``char_decoded``: Character reconstruction (or None if ``num_chars == 0``).
            - ``aux_decoded``: Auxiliary reconstruction.
            - ``latent``: The latent representation if ``latent_layer_type == "GAUSS"``, otherwise None.
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
        """Encode structured and auxiliary inputs into a shared latent representation.

        Args:
            phy (torch.Tensor): PHylogeny + charecter (cblv+s) input tensor with shape ``(N, C, W)``.
            aux (torch.Tensor): Auxiliary input tensor with shape ``(N, A)``.
            inference (bool, optional): If True, runs in eval mode and disables gradients
                for the duration of the call. Defaults to False.
            detach (bool, optional): If True (and ``inference`` is True), detaches the returned
                latent tensor from the computation graph. Defaults to False.

        Returns:
            torch.Tensor: Latent tensor with shape ``(N, latent_dim)`` (flattened).
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
        """Decode a latent representation back to structured and auxiliary outputs.

        Decoding applies a few constraints in normalized space:

        - Structured phylogenetic channels are clamped so their inverse-transformed values
          remain within the original ``[0, 1]`` bounds.
        - The auxiliary "num tips" field (``aux_numtips_idx``) is passed through a sigmoid-based
          transform to keep it within ``[2, structured_input_width]`` after inverse transform.
        - If ``inference=True`` and ``char_type == "categorical"``, character channels are softmaxed.

        Args:
            z (torch.Tensor): Latent tensor with shape ``(N, latent_dim)``.
            inference (bool, optional): If True, runs in eval mode and disables gradients
                for the duration of the call. Defaults to False.
            detach (bool, optional): If True (and ``inference`` is True), detaches the returned
                tensors from the computation graph. Defaults to False.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: ``(decoded_structured, decoded_aux)`` where
            ``decoded_structured`` has shape ``(N, C, W)`` and includes both phylogenetic and
            (optionally) character channels, and ``decoded_aux`` has shape ``(N, A)``.
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
        """Reconstruct inputs and return NumPy arrays on CPU.

        This is a convenience wrapper around calling the module directly and converting
        outputs to NumPy. Outputs are in the same (typically normalized) space as the model
        was trained in; use ``norm_predict_denorm()`` if you want predictions in the original
        input scale.

        Args:
            phy (torch.Tensor): Structured input tensor with shape ``(N, C, W)``.
            aux (torch.Tensor): Auxiliary input tensor with shape ``(N, A)``.
            inference (bool, optional): If True, runs in eval mode and disables gradients.
                Defaults to False.
            detach (bool, optional): If True (and ``inference`` is True), detaches the outputs
                before converting to NumPy. Defaults to False.

        Returns:
            Tuple[np.ndarray, np.ndarray]: ``(phy_pred, aux_pred)`` arrays on CPU. ``phy_pred`` has
            shape ``(N, C, W)`` and ``aux_pred`` has shape ``(N, A)``.
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
        """Set the fitted normalizers used by the inference helpers.

        Args:
            phy_norm: Structured-data normalizer (sklearn-like).
            aux_norm: Auxiliary-data normalizer (sklearn-like).
        """
        self.phy_normalizer = phy_norm
        self.aux_normalizer = aux_norm

    # inference machinery. Handles normalization too.
    def norm_and_encode(self, phy: np.array, aux: np.array) -> np.array:
        """Normalize raw inputs and return their latent encoding.

        Args:
            phy (np.ndarray or torch.Tensor): Structured input with shape ``(N, C, W)``.
            aux (np.ndarray or torch.Tensor): Auxiliary input with shape ``(N, A)``.

        Returns:
            np.ndarray: Latent encodings with shape ``(N, latent_dim)``.
        """
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
        """Decode latent vectors and inverse-transform to original data space.

        Args:
            latent (np.ndarray or torch.Tensor): Latent vectors with shape ``(N, latent_dim)``.

        Returns:
            Tuple[np.ndarray, np.ndarray]: ``(phy, aux)`` in the original (inverse-transformed)
            scale. ``phy`` has shape ``(N, C, W)`` and ``aux`` has shape ``(N, A)``.
        """
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
        """Normalize inputs, run the autoencoder, and inverse-transform outputs.

        Args:
            phy (np.ndarray or torch.Tensor): Structured input with shape ``(N, C, W)``.
            aux (np.ndarray or torch.Tensor): Auxiliary input with shape ``(N, A)``.

        Returns:
            Tuple[np.ndarray, np.ndarray]: ``(phy_pred, aux_pred)`` in the original input scale.
        """
        # phy shape = (N, nc, mt)
        # phy output shape = (N, nc, mt)
        return self.decode_and_denorm( self.norm_and_encode(phy, aux) )
        

    def write_network_to_file(self, out_fn) -> None:
        """Write network architecture to simple text file.

        Args:
            out_fn (str): Output path for the text file.
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
        """Seed Python, NumPy, and PyTorch RNGs for reproducibility.

        Notes:
            This mutates global RNG state (``random``, ``numpy.random``, and ``torch``) and sets
            cuDNN to deterministic mode.

        Args:
            seed (Optional[int]): Seed value. If None, this is a no-op. Defaults to None.
        """
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
    """Small MLP encoder for auxiliary (unstructured) features."""
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
        """Encode auxiliary inputs.

        Args:
            x (torch.Tensor): Auxiliary tensor of shape ``(N, in_width)``.

        Returns:
            torch.Tensor: Encoded auxiliary tensor of shape ``(N, out_width)``.
        """
        return self.unstructured_encoder(x)

class CnnEncoder(nn.Module):
    """1D CNN encoder for structured inputs (e.g. phylogenetic tensors)."""
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
        """Encode structured inputs.

        Args:
            x (torch.Tensor): Structured tensor of shape ``(N, C, W)``.

        Returns:
            torch.Tensor: Encoded structured tensor of shape ``(N, out_channels[-1], W_out)``.
        """
        return self.cnn_layers(x)
        

# decoder classes
class DenseDecoder(nn.Module):
    """Small MLP decoder for auxiliary (unstructured) features."""
    def __init__(self, in_width, out_width):
        super().__init__()
        self.unstructured_decoder = nn.Sequential(
            # TODO: 10 probably should be controlled by a parameter...
            nn.Linear(in_width, 10),
            nn.ReLU(),
            nn.Linear(10, out_width)
        )
    def forward(self, x):
        """Decode auxiliary outputs.

        Args:
            x (torch.Tensor): Decoded latent tensor of shape ``(N, in_width)``.

        Returns:
            torch.Tensor: Reconstructed auxiliary tensor of shape ``(N, out_width)``.
        """
        return self.unstructured_decoder(x)
    
class CnnDecoder(nn.Module):
    """1D transposed-CNN decoder for structured outputs.

    This decoder mirrors the structured encoder and uses ``torch.nn.ConvTranspose1d`` layers with
    computed padding/output-padding to hit requested intermediate widths and the final
    ``data_width``.
    """
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
        """Decode structured outputs from a structured latent tensor.

        Args:
            x (torch.Tensor): Structured latent tensor of shape
                ``(N, out_channels[-1], latent_width)``.

        Returns:
            torch.Tensor: Reconstructed structured tensor of shape ``(N, data_channels, data_width)``.
        """
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
        """Compute ``padding`` and ``output_padding`` for ``torch.nn.ConvTranspose1d``.

        Args:
            w_out_target (int): Desired output width.
            w_in (int): Input width to the transpose convolution.
            s (int): Stride.
            k (int): Kernel size.
            d (int, optional): Dilation. Defaults to 1.

        Returns:
            Tuple[int, int]: ``(padding, output_padding)`` values.
        """
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
    """Learned per-element scaling for structured output channels.

    Used to rescale character logits before concatenation with phylogenetic channels.
    """
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
        """Scale an input tensor elementwise.

        Args:
            x (torch.Tensor): Tensor of shape ``(N, c, w)``.

        Returns:
            torch.Tensor: Tensor of shape ``(N, c, w)`` with learned scaling applied.
        """
        # print(torch.max(self.scale))
        flat_x = x.view((x.shape[0], self.c * self.w))
        scaled_flat_x = self.scale * flat_x
        scaled_x = scaled_flat_x.view(x.shape[0], self.c, self.w)
        return scaled_x
        # print(torch.max(self.Lin.weight))
        # return(self.head(x))
    


# Latent layer classes
class LatentCNN(nn.Module):
    """CNN latent layer that preserves ``(channels, width)`` shape."""
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
        """Apply the latent CNN.

        Args:
            x (torch.Tensor): Tensor of shape ``(N, C, W)``.

        Returns:
            torch.Tensor: Tensor of shape ``(N, C, W)``.
        """
        return self.shared_layer(x)
    
class LatentDense(nn.Module):
    """Dense latent layer operating on flattened embeddings."""
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
        """Apply the latent MLP.

        Args:
            x (torch.Tensor): Tensor of shape ``(N, in_width)``.

        Returns:
            torch.Tensor: Tensor of shape ``(N, out_width)``.
        """
        return self.shared_layer(x)
    
    def __iter__(self):
        return iter(self.shared_layer)

    
# might be more work than its worth
class LatentPool(nn.Module):
    """Experimental latent layer using adaptive average pooling plus a linear map."""
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
        """Apply pooled latent mapping.

        Args:
            struct (torch.Tensor): Structured embedding tensor.
            unstruct (torch.Tensor): Unstructured embedding tensor.

        Returns:
            torch.Tensor: Latent tensor of shape ``(N, latent_dim)``.
        """
        #
        avg_pool = self.avg_pool_layer(struct)
        flat_avg_pool = avg_pool.flatten(start_dim=1)                
        combined_latent = torch.cat((flat_avg_pool, unstruct), dim=1)
        return self.shared_layer(combined_latent)

    
class LatentDenseDecoder(nn.Module): # same as LatentGauss
    """Dense decoder that expands the latent vector back to a flattened structured embedding."""
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
        """Decode latent vectors.

        Args:
            x (torch.Tensor): Tensor of shape ``(N, in_width)``.

        Returns:
            torch.Tensor: Tensor of shape ``(N, out_width)``.
        """
        return self.shared_layer(x)
    
    def __iter__(self):
        return iter(self.shared_layer)
    
class LatentCNNDecoder(nn.Module):
    """CNN decoder counterpart to ``LatentCNN`` that preserves ``(channels, width)`` shape."""
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
        """Apply the latent CNN decoder.

        Args:
            x (torch.Tensor): Tensor of shape ``(N, C, W)``.

        Returns:
            torch.Tensor: Tensor of shape ``(N, C, W)``.
        """
        return self.shared_layer(x)



# Dev
class TwoSidedReLU(nn.Module):
    """Clamp an input tensor elementwise between per-element min/max bounds."""
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
 
