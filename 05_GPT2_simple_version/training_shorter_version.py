
"""
A much shorter version of train.py for benchmarking
"""
import os
from os import path
from contextlib import nullcontext
import numpy as np
import time
import torch
from model import GPTConfig, GPT

from torch.utils import tensorboard as tb


def train(args):
    seed = 1337
    batch_size = args.batch_size
    sent_len = args.sent_len
    real_data = args.read_data
    profile = args.profile # # use pytorch profiler, or just simple benchmarking?

    train_logger, valid_logger = None, None
    if args.log_dir:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, "train"), flush_secs=1)
        # valid_logger = tb.SummaryWriter(path.join(args.log_dir, "valid"), flush_secs=1)

    device = "cpu" #'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
    dtype = 'bfloat16' # 'float32' or 'bfloat16' or 'float16'
    compile = True if device != "cpu" else False # use PyTorch 2.0 to compile the model to be faster

    # Dynamically running a piece of Python code for config
    # exec(open('configurator.py').read()) # overrides from command line or config file
    # -----------------------------------------------------------------------------

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
    device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    # data loading init
    if real_data:
        dataset = "shakespeare"  # 'openwebtext'
        data_dir = os.path.join('data', dataset)
        train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r') # here we need first to run the python script in data\\shakespears\prepare.py to build the train.bin
        def get_batch(split):
            data = train_data # note ignore split in benchmarking script
            ix = torch.randint(len(data) - sent_len, (batch_size,))
            x = torch.stack([torch.from_numpy((data[i:i+sent_len]).astype(np.int64)) for i in ix])
            y = torch.stack([torch.from_numpy((data[i+1:i+1+sent_len]).astype(np.int64)) for i in ix])
            if device != "cpu":
                x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
            return x, y
    else:
        # alternatively, if fixed data is desired to not care about data loading
        x = torch.randint(50304, (batch_size, sent_len), device=device)
        y = torch.randint(50304, (batch_size, sent_len), device=device)
        get_batch = lambda split: (x, y)

    # model config
    gptconf = GPTConfig(block_size=args.sent_len, vocab_size=args.vocab_size, n_layer=args.n_layer, n_head=args.n_head,
                        n_embd=args.emb_size, dropout=args.dropout, # for determinism
                        bias=args.bias,)
    model = GPT(gptconf)
    model.to(device)

    optimizer = model.configure_optimizers(weight_decay=1e-2, learning_rate=args.learning_rate, betas=(0.9, 0.95), device_type=device_type)

    if compile:
        print("Compiling model...")
        model = torch.compile(model) # pytorch 2.0

    if profile:
        # useful docs on pytorch profiler:
        # - tutorial https://pytorch.org/tutorials/intermediate/tensorboard_profiler_tutorial.html
        # - api https://pytorch.org/docs/stable/profiler.html#torch.profiler.profile
        wait, warmup, active = 5, 5, 5
        num_steps = wait + warmup + active
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(wait=wait, warmup=warmup, active=active, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(path.join(args.log_dir, "bench_log")),
            record_shapes=False,
            profile_memory=False,
            with_stack=False, # incurs an additional overhead, disable if not needed
            with_flops=True,
            with_modules=False, # only for torchscript models atm
        ) as prof:

            X, Y = get_batch('train')
            for k in range(num_steps):
                with ctx:
                    logits, loss = model(X, Y)
                X, Y = get_batch('train')
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                lossf = loss.item()
                print(f"{k}/{num_steps} loss: {lossf:.4f}")

                prof.step() # notify the profiler at end of each step
    else:
        # simple benchmarking
        if device != "cpu":
            torch.cuda.synchronize()

        for stage, num_steps in enumerate([10, 20]): # burnin, then benchmark
            t0 = time.time()
            X, Y = get_batch('train') # X & Y in shape: Bs X sent_len
            for k in range(num_steps):
                with ctx:
                    logits, loss = model(X, Y) # logits: Bs X sent_len X #Vocab_size && loss is a single value tensor
                X, Y = get_batch('train')
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                lossf = loss.item()
                print(f"{k}/{num_steps} loss: {lossf:.4f}")
                if args.log_dir is not None:
                    train_logger.add_scalar("AvgLoss/Train", lossf, num_steps)

            t1 = time.time()
            dt = t1-t0
            if device != "cpu":
                torch.cuda.synchronize()
                mfu = model.estimate_mfu(batch_size * 1 * num_steps, dt)
            if stage == 1:
                print(f"time per iteration: {dt/num_steps*1000:.4f}ms, MFU: {mfu*100:.2f}%")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("-bs", "--batch_size", type=int, default=12, help="batch size") 
    parser.add_argument("-sl", "--sent_len", type=int, default=35,
                        help="Length of sentences to be passed into the encoder network") # 1024
    parser.add_argument("-vcs", "--vocab_size", type=int, default=50304, help="GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency") 
    parser.add_argument("-nlr", "--n_layer", type=int, default=2, help="# of multi-attention layer in encoder") #12
    parser.add_argument("-nhd", "--n_head", type=int, default=2, help="# of multi-attention layer in encoder")  # 12
    parser.add_argument("-esz", "--emb_size", type=int, default=8, help="embedding vector size")  # 768
    parser.add_argument("-drp", "--dropout", type=float, default=0.0,
                        help="Dropout in feedforward layer within transformer")  # 0.2
    parser.add_argument("-lr", "--learning_rate", type=float, default=1.E-4, help="Learning rate for optimizer")

    parser.add_argument("-bis", "--bias", type=bool, default=False, help="Include bias?")
    parser.add_argument("-rdt", "--read_data", type=bool, default=True, help="Reading Data")
    parser.add_argument("-prf", "--profile", type=bool, default=False, help="Whether or not to profile the code?")

    parser.add_argument("--log_dir")

    args = parser.parse_args()

    train(args)