import torch
from einops import rearrange
import sys 
import torch.nn.functional as F
from collections import Counter

def unpack_data(
        target,
        observed_mask,
        sample_id,
        time_id,
        variate_id,
        prediction_mask,
        patch_size):
    '''
    input: 
        target: [b n p_max]
        observed_mask: [b n p_max]
        sample_id: [b n]
        time_id: [b n]
        variate_id: [b n]
        prediction_mask: [b n]
        patch_size: [b n]
    output: dict
        {
        'target': [b_new, l_max, K],
        'observed_mask': [b_new, l_max, K],
        'time_id': [b_new, l_max, K],
        'variate_id': [b_new, l_max, K],
        'prediction_mask': [b_new, l_max, K],
        }
    '''
    
    packed_data = {
            'target': target,
            'observed_mask': observed_mask,
            'time_id': time_id,
            'variate_id': variate_id,
            'prediction_mask': prediction_mask,
            'patch_size': patch_size
        }
    
    unpadded_sequences = {
            'target': [],
            'observed_mask': [],
            # 'time_id': [],
            'variate_id': [],
            'prediction_mask': [],
        }
    

    max_len = 0
    B = len(sample_id)

    for b in range(B):
        # unpack
        unique_sample_ids = torch.unique(sample_id[b])
        for target_sample_id in unique_sample_ids:
            if target_sample_id == 0:
                continue
            
            mask = (sample_id[b] == target_sample_id)

            # unflatten
            unique_var_ids = torch.unique(variate_id[b][mask])
            for target_var_id in unique_var_ids:
                if target_var_id == 0:
                    continue
                
                var_mask = (variate_id[b][mask] == target_var_id)

                patch_size = packed_data['patch_size'][b][mask][var_mask][0]
                for key in unpadded_sequences:
                    selected_sample = packed_data[key][b][mask][var_mask]

                    # unpatch
                    if key in ['target', 'observed_mask']:
                        selected_sample = selected_sample[:,:patch_size]
                        selected_sample = rearrange(selected_sample, 'n p -> (n p)')
                        max_len = max(max_len, selected_sample.size(0))
                    elif key in ['variate_id', 'prediction_mask']:
                        selected_sample = torch.repeat_interleave(selected_sample, patch_size)

                    unpadded_sequences[key].append(selected_sample)
                    

    padded_sequences = {
        'target': [],
        'observed_mask': [],
        'variate_id': [],
        'prediction_mask': []
    }
    
    for key in unpadded_sequences:
        for seq in unpadded_sequences[key]:
            padded_seq = F.pad(seq, (0, max_len - len(seq)), value=0)
            padded_sequences[key].append(padded_seq)

        # Stack the padded sequences into a single tensor

        padded_sequences[key] = torch.stack(padded_sequences[key])
        padded_sequences[key] = rearrange(padded_sequences[key], 'b l -> b l 1')
    

    return padded_sequences
