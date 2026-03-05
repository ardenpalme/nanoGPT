from torch import nn
import numpy as np
import torch
import wandb

def log_attention_svd(model, wandb_run, step=None):
    """Log singular values of all 'c_attn' Linear modules."""
    with torch.no_grad():
        log_data = {}
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear) and name.endswith('c_attn'):
                weight = module.weight.cpu()
                S = torch.linalg.svdvals(weight).cpu().numpy()
                safe_name = name.replace('.', '/')

                log_data[f"c_attn_max/{safe_name}"] = S.max()
                log_data[f"c_attn_min/{safe_name}"] = S.min()
                if len(S) > 1:
                    log_data[f"c_attn_cond/{safe_name}"] = S[0] / S[-1]
        wandb_run.log(log_data, step=step)

