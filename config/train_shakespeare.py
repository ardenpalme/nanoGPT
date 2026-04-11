# train a miniature shakespeare model with BPE embeddings

import datetime

out_dir = 'out-shakespeare'
eval_interval = 250 # keep frequent because we'll overfit
eval_iters = 100
log_interval = 10 # don't print too too often

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = False

wandb_log = False # override via command line if you like
wandb_project = 'shakespeare'
wandb_run_name = 'mini-gpt ' + datetime.datetime.now().strftime("%m/%d/%y %H:%M:%S")

dataset = 'shakespeare'
gradient_accumulation_steps = 1

batch_size = 16
block_size = 256 # input sequence length
n_layer = 4
n_head = 4
head_size = 64
n_embd = 256   # embedding dimension
dropout = 0.2
mlp_width = 4 * n_embd

learning_rate = 1e-3 # with baby networks can afford to go a bit higher
max_iters = 5000
lr_decay_iters = 5000 # make equal to max_iters usually
min_lr = 1e-4 # learning_rate / 10 usually
beta2 = 0.99 # make a bit bigger because number of tokens per iter is small

warmup_iters = 100 # not super necessary potentially

device = 'xpu'  
compile = False # do not torch compile the model
