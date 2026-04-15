import time
from torch import nn
import numpy as np
import torch
import wandb
import os
import pandas as pd
import pickle as pkl
from model import GPTConfig, GPT
import scipy as sc
import scipy.optimize as opt

def get_batch(split, data_dir, device, block_size, batch_size, device_type='cpu'):
    # We recreate np.memmap every batch to avoid a memory leak, as per
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
    if split == 'train':
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')

    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
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

def get_ckpt_model(ckpt_fname, device='cpu', ckpt_dir='out-shakespeare-char'):
    torch.manual_seed(1337)
    ckpt_path = os.path.join(ckpt_dir, ckpt_fname)
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    model.eval() # disables dropout
    model.to(device)
    return model

def calc_optval(model, out_dir, out_fname):
    device='cpu'
    dataset='shakespeare_char'
    torch.manual_seed(1337)

    # load the checkpointed model state from last train save
    ckpt_path = os.path.join(out_dir, out_fname)
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)

    model.eval() # disables dropout
    model.to(device)

    n = model.config.block_size
    h = model.config.n_head
    hs = model.config.head_size
    X, Y = get_batch('eval', os.path.join('data', dataset), device, n, model.config.batch_size)
    A,T,v = model.get_matricies(X,0)
    res = two_phase_opt(n,A,T,X.numpy())
    if(res != None):
        print(f"Optimal objective: {res[0]}")

def two_phase_opt(n,A,T,X,col_idx=0):
    T_aug = np.concatenate([T, np.ones((n, 1))], axis=1)
    basis_LN_base = sc.linalg.null_space(X)
    basis_LN = sc.linalg.null_space(T_aug.T)

    a = A[:, col_idx].copy()
    B = basis_LN
    E = basis_LN_base

    k = B.shape[1]
    eps = 1e-8

    def get_x(lam):
        return a + B @ lam

    def objective(lam):
        x = get_x(lam)
        if np.any(x <= 0):
            return 1e12
        vals = (E.T @ np.log(x)) / (E.T @ np.ones(n))
        return float(np.max(np.abs(vals - vals[0])))

    c_feas  = np.zeros(k + 1); 
    c_feas[-1] = -1.0  # minimizing -s
    A_ub = np.hstack([-B, np.ones((n, 1))]) # -B @ lam + s <= a
    lp = opt.linprog(c_feas, A_ub=A_ub, b_ub=a, bounds=[(None,None)]*k + [(None,None)])
    lam0 = lp.x[:k]

    results = []
    for trial in range(50):
        if trial == 0:
            lam_init = lam0
        else:
            lam_init = lam0 + np.random.randn(k) * 0.01

        # discard any initial lambdas with subcomponents < tolerance
        if np.any(get_x(lam_init) <= eps):
            continue

        res = opt.minimize(
            objective,
            lam_init,
            method='SLSQP',
            constraints={'type': 'ineq', 'fun': lambda l: get_x(l) - eps},
            options={'ftol': 1e-8, 'maxiter': 2500, 'disp': False}
        )
        if res.fun < 1e11:  # filter out infeasible
            results.append(res)

    if(len(results) == 0):
        return None

    best = min(results, key=lambda r: r.fun)
    a_opt = get_x(best.x)

    if(best.success): 
        return best.fun, a_opt
    else:
        return None
