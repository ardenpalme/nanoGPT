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


def calc_optval(model):
    device='cpu'
    dataset='shakespeare_char'
    checkpoint_dir='out-shakespeare-char'
    torch.manual_seed(1337)

    # load the checkpointed model state from last train save
    ckpt_path = os.path.join(checkpoint_dir, 'ckpt.pt')
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
    T_aug = np.concatenate([T, np.ones((n, 1))], axis=1)
    basis_LN_base = sc.linalg.null_space(T.T)
    basis_LN = sc.linalg.null_space(T_aug.T)

    a_orig = A[:, 1].copy()
    B = basis_LN
    E = basis_LN_base

    k = B.shape[1]
    n = len(a_orig)
    e_dot_1 = E.T @ np.ones(n)
    eps = 1e-8

    def get_x(lam):
        return a_orig + B @ lam

    def objective(lam):
        x = get_x(lam)
        if np.any(x <= 0):
            return 1e12
        vals = (E.T @ np.log(x)) / e_dot_1
        return float(np.max(np.abs(vals - vals[0])))

    # Step 1: LP warm start — find interior feasible point
    # maximize s s.t. a + B @ lam >= s, i.e. -B @ lam + s <= a
    c_feas  = np.zeros(k + 1); c_feas[-1] = -1.0
    A_feas  = np.hstack([-B, np.ones((n, 1))])
    lp      = opt.linprog(c_feas, A_ub=A_feas, b_ub=a_orig,
                      bounds=[(None,None)]*k + [(None,None)],
                      method='highs')
    lam0    = lp.x[:k]
    print(f"LP slack:       {-lp.fun:.6f}")
    print(f"Objective(lam0): {objective(lam0):.6f}")

    # Step 2: SLSQP with multiple restarts
    results = []
    for trial in range(50):
        lam_init = lam0 + np.random.randn(k) * (0 if trial == 0 else 0.01)
        if np.any(get_x(lam_init) <= eps):
            continue
        res = opt.minimize(
            objective,
            lam_init,
            method='SLSQP',
            constraints={'type': 'ineq', 'fun': lambda l: get_x(l) - eps},
            options={'ftol': 1e-12, 'maxiter': 5000, 'disp': False}
        )
        if res.fun < 1e11:  # filter out infeasible
            results.append(res)

    best = min(results, key=lambda r: r.fun)
    a_opt = get_x(best.x)

    print(f"\nOptimal objective: {best.fun:.6f}")
    print(f"Min component:     {np.min(a_opt):.8f}")
    print(f"Solver success:    {best.success}")
    print(f"Solver message:    {best.message}")
