import time
from torch import nn
import numpy as np
import torch
import wandb
import os
import pandas as pd
import pickle as pkl

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

def save_common_batch(model, dataset, device, n_samples=250, filepath="saved_batch.pkl"):
    data_dir = os.path.join('data', dataset)
    X_arr = []
    Y_arr = []
    for _ in range(n_samples):
        X, Y = get_batch('val', data_dir, device, 'gpu', model.config.block_size, model.config.batch_size)
        X_arr.append(X.cpu())
        Y_arr.append(Y.cpu())

    with open(filepath, 'wb') as f:
        pkl.dump({'X': X_arr, 'Y': Y_arr}, f)

def calc_perplexity(model, device, filepath="saved_batch.pkl", elapsed_time=0):
    '''
    Must be called from top-level project dir where results.csv results resides
    '''
    with open(filepath, 'rb') as f:
        batch = pkl.load(f)
    
    X = [x.to(device) for x in batch['X']]
    Y = [y.to(device) for y in batch['Y']]
    n = len(X)

    perp = torch.zeros(n)
    for i in range(n):
        logits, loss = model(X[i], Y[i])
        perp[i] = torch.exp(loss)

    metrics = {
        'n': model.config.block_size, # sequence length
        'd': model.config.n_embd,
        'h': model.config.n_head, 
        'dp': model.config.dp,
        'n_params': model.get_num_params(),
        'mlp_width': model.config.mlp_width,
        'training_time':elapsed_time,
        'perplexity': float(torch.mean(perp)),
    }
    df = pd.read_csv("results.csv")
    df_new = pd.concat([df, pd.DataFrame([metrics])], ignore_index=True)
    df_new.to_csv("results.csv", index=False)

    return metrics

'''
32,128,4,32,664320,68,7.761092185974121,1024
32,128,8,32,795392,82,7.605432510375977,1024
32,128,16,32,1057536,81,7.482982158660889,1024
32,128,32,32,1581824,106,7.527788162231445,1024
32,128,40,32,1843968,108,7.5979814529418945,1024
32,128,64,32,2630400,134,7.542094230651855,1024
32,256,4,32,2377216,0,6.753637790679932,2048
32,256,8,32,2639360,0,6.547783851623535,2048
32,256,16,32,3163648,0,6.7082719802856445,2048
32,256,32,32,4212224,0,6.693365097045898,2048
32,256,40,32,4736512,0,6.632531642913818,2048
32,256,64,32,6309376,0,6.611839771270752,2048
'''
