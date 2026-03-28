from torch import nn
import numpy as np
import torch
import wandb
import os

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


def get_batch(split, data_dir, device, device_type, block_size, batch_size):
    # We recreate np.memmap every batch to avoid a memory leak, as per
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
    if split == 'train':
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y
