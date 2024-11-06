#  Copyright (c) 2024, Salesforce, Inc.
#  SPDX-License-Identifier: Apache-2
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

from functools import partial


import torch
import torch.nn.functional as F
from huggingface_hub import PyTorchModelHubMixin
from hydra.utils import instantiate
from jaxtyping import Bool, Float, Int
from torch import nn
from torch.distributions import Distribution
from torch.utils._pytree import tree_map

from typing import Union
import sys
from .data_adapter import unpack_data

# from uni2ts.common.torch_util import mask_fill, packed_attention_mask
# from uni2ts.distribution import DistributionOutput
# from uni2ts.module.norm import RMSNorm
# from uni2ts.module.packed_scaler import PackedNOPScaler, PackedStdScaler
# from uni2ts.module.position import (
#     BinaryAttentionBias,
#     QueryKeyProjection,
#     RotaryProjection,
# )
# from uni2ts.module.transformer import TransformerEncoder
# from uni2ts.module.ts_embed import MultiInSizeLinear
from uni2ts.module.elastst.ElasTST_backbone import ElasTST_backbone
from uni2ts.module.elastst.utils import convert_to_list, InstanceNorm

class ElasTSTModule(
    nn.Module,
    PyTorchModelHubMixin
):
    """
    Contains components of Moirai, to ensure implementation is identical across models.
    Subclasses huggingface_hub.PyTorchModelHubMixin to support loading from HuggingFace Hub.
    """

    def __init__(
        self,
        patch_sizes: tuple[int, ...],  # tuple[int, ...] | list[int]
        l_patch_size: Union[str, int, list] = '8_16_32',
        k_patch_size: int = 1,
        stride: int = None,
        rotate: bool = True, 
        addv: bool = False,
        bin_att: bool = False,
        rope_theta_init: str = 'exp',
        min_period: float = 1, 
        max_period: float = 1000,
        learn_tem_emb: bool = False,
        learnable_rope: bool = True, 
        abs_tem_emb: bool = False,
        structured_mask: bool = True,
        max_seq_len: int = 512,
        theta_base: float = 10000,
        t_layers: int = 1, 
        v_layers: int = 0,
        patch_share_backbone: bool = True,
        n_heads: int = 16, 
        d_k: int = 8, 
        d_v: int = 8,
        d_inner: int = 256, 
        dropout: float = 0.,
        in_channels: int = 1,
        f_hidden_size: int = 40,
        use_norm: bool = True,
        **kwargs
    ):
        """
        :param distr_output: distribution output object
        :param d_model: model hidden dimensions
        :param num_layers: number of transformer layers
        :param patch_sizes: sequence of patch sizes
        :param max_seq_len: maximum sequence length for inputs
        :param attn_dropout_p: dropout probability for attention layers
        :param dropout_p: dropout probability for all other layers
        :param scaling: whether to apply scaling (standardization)
        """
        super().__init__()
        self.l_patch_size = convert_to_list(l_patch_size)
        self.use_norm = use_norm
        self.patch_sizes=patch_sizes
        self.max_seq_len = max_seq_len
        # Model
        self.model = ElasTST_backbone(l_patch_size=self.l_patch_size, 
            stride=stride, 
            k_patch_size=k_patch_size, 
            in_channels=in_channels,
            t_layers=t_layers, 
            v_layers=v_layers, 
            hidden_size=f_hidden_size, 
            d_inner=d_inner,
            n_heads=n_heads, 
            d_k=d_k, 
            d_v=d_v,
            dropout=dropout,
            rotate=rotate, 
            max_seq_len=max_seq_len, 
            theta=theta_base,
            addv=addv, 
            bin_att=bin_att,
            learn_tem_emb=learn_tem_emb, 
            abs_tem_emb=abs_tem_emb, 
            learnable_theta=learnable_rope, 
            structured_mask=structured_mask,
            rope_theta_init=rope_theta_init, 
            min_period=min_period, 
            max_period=max_period,
            patch_share_backbone=patch_share_backbone
        )
        
        self.instance_norm = InstanceNorm()

    def forward(
        self,
        target: Float[torch.Tensor, "*batch seq_len max_patch"],
        observed_mask: Bool[torch.Tensor, "*batch seq_len max_patch"],
        sample_id: Int[torch.Tensor, "*batch seq_len"],
        time_id: Int[torch.Tensor, "*batch seq_len"],
        variate_id: Int[torch.Tensor, "*batch seq_len"],
        prediction_mask: Bool[torch.Tensor, "*batch seq_len"],
        patch_size: Int[torch.Tensor, "*batch seq_len"],
    ) -> Distribution:
        """
        This method expects processed inputs.

        1. Apply scaling to observations
        2. Project from observations to representations
        3. Replace prediction window with learnable mask
        4. Apply transformer layers
        5. Project from representations to distribution parameters
        6. Return distribution object

        :param target: input data
        :param observed_mask: binary mask for missing values, 1 if observed, 0 otherwise
        :param sample_id: indices indicating the sample index (for packing)
        :param time_id: indices indicating the time index
        :param variate_id: indices indicating the variate index
        :param prediction_mask: binary mask for prediction horizon, 1 if part of the horizon, 0 otherwise
        :param patch_size: patch size for each token
        :return: predictive distribution
        """
        
        unpacked_sequences = unpack_data(target,
        observed_mask,
        sample_id,
        time_id,
        variate_id,
        prediction_mask,
        patch_size)

        B, L, K = unpacked_sequences['target'].shape

        new_pred_len = L
        for p in self.l_patch_size:
            new_pred_len = self.check_divisibility(new_pred_len, p)
        
        pad_length = new_pred_len - L
        
        # past_target = unpadded_sequences[]
        # past_observed_values = batch_data.past_observed_values
        
        if self.use_norm:
            x = self.instance_norm(unpacked_sequences['target'], 'norm', mask=~unpacked_sequences['prediction_mask'])
            unpacked_sequences['target'] = self.instance_norm(unpacked_sequences['target'], 'norm', mask=~unpacked_sequences['prediction_mask'])

        past_value_indicator = ~unpacked_sequences['prediction_mask'] * unpacked_sequences['observed_mask']
        x[~past_value_indicator] = 0
        # # future_observed_values is the mask indicate whether there is a value in a position
        # future_observed_values = torch.zeros([B, new_pred_len, K]).to(batch_data.future_observed_values.device)

        # pred_len = batch_data.future_observed_values.shape[1]
        # future_observed_values[:,:pred_len] = batch_data.future_observed_values

        # # target placeholder
        # future_placeholder = torch.zeros([B, new_pred_len, K]).to(batch_data.past_target_cdf.device)
        
        x = F.pad(x, (0, 0, 0, pad_length))
        past_value_indicator = F.pad(past_value_indicator, (0, 0, False, pad_length))
        observed_mask = F.pad(unpacked_sequences['observed_mask'], (0, 0, False, pad_length))
        
        pred, pred_list = self.model(x, past_value_indicator, observed_mask)
        pred = pred[:, :L]
        
        # if self.use_norm:
        #     pred = self.instance_norm(pred, 'denorm')
            
        return unpacked_sequences, pred
    
    def check_divisibility(self, pred_len, patch_size):
        if pred_len % patch_size == 0:
            return pred_len
        else:  
            return (pred_len // patch_size + 1) * patch_size  