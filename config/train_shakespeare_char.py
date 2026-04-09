# train a miniature character-level shakespeare model
# good for debugging and playing on macbooks and such

import datetime

out_dir = 'out-shakespeare-char/fixedmultihead'
eval_interval = 250 # keep frequent because we'll overfit
eval_iters = 100
log_interval = 10 # don't print too too often

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = False

wandb_log = False # override via command line if you like
wandb_project = 'shakespeare-char'
wandb_run_name = 'mini-gpt ' + datetime.datetime.now().strftime("%m/%d/%y %H:%M:%S")

dataset = 'shakespeare_char'
gradient_accumulation_steps = 1
batch_size = 16
block_size = 32 # input sequence length

n_layer = 2
n_head = 3
head_size = 40
n_embd = 256
dropout = 0.2
mlp_width = 4 * n_embd

learning_rate = 1e-3 # with baby networks can afford to go a bit higher
max_iters = 5000
lr_decay_iters = 5000 # make equal to max_iters usually
min_lr = 1e-4 # learning_rate / 10 usually
beta2 = 0.99 # make a bit bigger because number of tokens per iter is small

warmup_iters = 100 # not super necessary potentially

# on macbook also add
device = 'xpu'  # run on cpu only
compile = False # do not torch compile the model
