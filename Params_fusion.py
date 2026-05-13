# -*- coding: utf-8 -*-
import numpy as np
import torch
import copy
import os
from scipy import spatial
from tqdm import tqdm

"""
Fedsoda Parameter Fusion Module
This module implements a dynamic weighting mechanism for federated learning.
It calculates the similarity between different client models using synthetic 
data and middle-layer activations to adjust the contribution of each client 
to the global model.
"""

def output_middle_layer(net, x):
    """
    Captures middle-layer activations using forward hooks.
    This is used to compute 'model fingerprints' for similarity comparison.
    """
    output_data = {}
    
    def get_activation(name):
        def hook(model, input, output):
            output_data[name] = output.detach()
        return hook

    # Layer names to hook into. Must match names in Network.py
    layer_names = ['conv1', 'conv3', 'conv5', 'conv7', 'conv9', 'conv11', 'conv13', 'out_conv']
    hooks = []
    for name in layer_names:
        if hasattr(net, name):
            hooks.append(getattr(net, name).register_forward_hook(get_activation(name)))

    device = next(net.parameters()).device
    # Move synthetic input to the correct device
    _ = net(torch.from_numpy(x).to(device))
    
    outputs = []
    for name in layer_names:
        if name in output_data:
            outputs.append(output_data[name].flatten())
    
    # Clean up hooks to prevent memory leaks
    for hook in hooks:
        hook.remove()
        
    return outputs


def For_one_centre(C_list, args, net, num_clients):
    """
    Calculates the cosine similarity between each client's model across 
    different layers using synthetic data generated from client-specific stats.
    """
    # Load mean/std for each client to generate representative synthetic data
    mean_std_list = []
    for i in range(num_clients):
        suffix = str(i) if i > 0 else ""
        path = getattr(args, f'Meanstd_dir{suffix}')
        mean_std_list.append(np.load(path))

    num_samples = 4
    x_in_list = []
    
    os.makedirs(args.save_path_array, exist_ok=True)

    # Generate or update synthetic inputs that represent each client's distribution
    for i in range(num_clients):
        npy_path = os.path.join(args.save_path_array, f'x_in_{i}.npy')
        mean, std = mean_std_list[i]
        
        x_new = np.random.normal(mean, std, num_samples * 3 * args.ROI_shape * args.ROI_shape)
        x_new = x_new.reshape((num_samples, 3, args.ROI_shape, args.ROI_shape)).astype(dtype=np.float32)
        
        if os.path.exists(npy_path):
            # EMA style update for synthetic data consistency
            x_former = np.load(npy_path)
            x_in = 0.75 * x_new + 0.25 * x_former
        else:
            x_in = x_new
        
        np.save(npy_path, x_in)
        x_in_list.append(x_in)

    pbar = tqdm(total=num_clients * 8, desc="Calculating Model Similarities")
    cos_sim_all = []
    
    for i in range(num_clients):
        cos_sim_single = []
        net.load_state_dict(C_list[i])
        net = net.cuda()

        # Get fingerprints of the current model on all clients' data distributions
        client_outputs = [output_middle_layer(net, x) for x in x_in_list]

        for layer_idx in range(len(client_outputs[0])):
            cos_sim = []
            anchor_output = client_outputs[i][layer_idx].data.cpu().numpy().flatten()
            
            for other_client_idx in range(num_clients):
                target_output = client_outputs[other_client_idx][layer_idx].data.cpu().numpy().flatten()
                # Cosine similarity between activations
                sim = 1 - spatial.distance.cosine(anchor_output, target_output)
                cos_sim.append(sim)
            
            # Remove self-similarity (always 1.0)
            del cos_sim[i]
            cos_sim_single.append(cos_sim)
            pbar.update(1)
        cos_sim_all.append(cos_sim_single)
        
    pbar.close()
    return cos_sim_all


def Fuse(base_dir, input_param_list, output_param_list, args, net):
    """
    Orchestrates the fusion of model weights.
    Calculates weights dynamically based on similarities and updates client models.
    """
    num_clients = len(input_param_list) - 1 # Exclude server entry
    
    with torch.no_grad():
        C_list = []
        for i in range(1, len(input_param_list)):
            C_list.append(torch.load(os.path.join(base_dir, input_param_list[i]), map_location='cpu'))

        # 1. Calculate similarity matrix
        cos_sim_all = For_one_centre(C_list, args, net, num_clients)

        # 2. Perform dynamic fusion
        C_avg = copy.deepcopy(C_list[0])
        C_list_new = [copy.deepcopy(c) for c in C_list]
        
        param_keys = list(C_list[0].keys())
        
        with tqdm(total=len(param_keys) * num_clients, desc="Fusing Model Weights") as pbar:
            for k_idx, k in enumerate(param_keys):
                # Mapping parameters to layer-specific similarity weights
                if k_idx <= 7: layer_idx = 0
                elif k_idx <= 15: layer_idx = 1
                elif k_idx <= 23: layer_idx = 2
                elif k_idx <= 31: layer_idx = 3
                elif k_idx <= 39: layer_idx = 4
                elif k_idx <= 47: layer_idx = 5
                elif k_idx <= 55: layer_idx = 6
                else: layer_idx = 7
                
                for client_idx in range(num_clients):
                    weights = cos_sim_all[client_idx][layer_idx]
                    out = fusion_logic(C_list, weights, client_idx, k)
                    C_list_new[client_idx][k].data = out
                    pbar.update(1)

        # 3. Final global average for the server model
        for k in param_keys:
            sum_data = torch.zeros_like(C_list_new[0][k].data).float()
            for client_idx in range(num_clients):
                sum_data += C_list_new[client_idx][k].data.float()
            C_avg[k].data = sum_data / num_clients

        # 4. Save updated parameters
        torch.save(C_avg, os.path.join(base_dir, output_param_list[0]))
        for i in range(num_clients):
            # In Fedsoda, clients typically pull the fused weights for their next step
            torch.save(C_avg, os.path.join(base_dir, output_param_list[i+1]))


def fusion_logic(C_list, weights, client_idx, k):
    """
    Implements the core fusion formula:
    (1 - dynamic_w) * Local_Weights + dynamic_w * Sum(Similarity_Weighted_Other_Weights)
    """
    weights = np.array(weights)
    sum_weights = np.sum(weights) + 1e-8
    normalized_weights = weights / sum_weights
    
    dynamic_w = 0.4 # Tuning parameter for global vs local balance
    C_list_others = [C_list[i] for i in range(len(C_list)) if i != client_idx]
    
    out = (1 - dynamic_w) * C_list[client_idx][k].data
    
    others_contribution = torch.zeros_like(out)
    for i, w in enumerate(normalized_weights):
        others_contribution += w * C_list_others[i][k].data
        
    out += dynamic_w * others_contribution
    return out
