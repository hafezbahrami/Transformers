from os import path
import copy
import time
import math

import torch
from torch import Tensor
import torch.utils.tensorboard as tb
import torch.nn as nn

# from transformer_model import TransformerModel, generate_square_subsequent_mask
from model_simpler_version import TransformerLanguageModel
from data_loader import get_train_test_eval_data, vocab, get_batch_x_y_sents
from torchtext.vocab import Vocab


def train(args):
    n_epochs = args.n_epochs
    batch_size = args.batch_size
    eval_batch_size = args.eval_batch_size
    emb_size = args.emb_size
    n_layers = args.n_layers
    n_head = args.n_head
    dropout = args.dropout
    sent_len = args.sent_len
    lr = args.learning_rate
    log_print_interval = args.log_print_interval
    batch_size_first = args.batch_size_first

    train_logger, valid_logger = None, None
    if args.log_dir:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, "train"), flush_secs=1)
        valid_logger = tb.SummaryWriter(path.join(args.log_dir, "valid"), flush_secs=1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # train_data shape=[102499, Bs]  && val_data shape=[ , val_Bs]
    train_data, val_data, test_data = get_train_test_eval_data(batch_size, eval_batch_size, device,
                                                               batch_size_first=batch_size_first)

    ntokens = len(vocab)  # size of vocabulary: for WikiText2 ~ 28,800 words
    _get_itos = Vocab(vocab).get_itos()

    model = TransformerLanguageModel(vocab_size=ntokens, emb_size=emb_size, sent_len=sent_len, n_head=n_head,
                                n_layer=n_layers, dropout=dropout).to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=1.0, gamma=args.lr_decay_factor)

    iter = 0
    for epoch in range(n_epochs):
        epoch_start_time = time.time()
        model.train()  # in train mode
        total_loss = 0.
        start_time = time.time()

        len_each_batch = train_data.size(1) if batch_size_first else train_data.size(0)

        num_sent_per_batches = len_each_batch // sent_len
        for batch, i_start_sent in enumerate(range(0, len_each_batch-1, sent_len)):
            iter += 1
            # both data and target are in shape: Bs X sent_len. Target is shifted to the right as much as one word
            x_data_sent, targets = get_batch_x_y_sents(train_data, i_start_sent, sent_len, batch_size_first)
            output, loss = model(x_data_sent, targets)  # output shape: Bs X sent_len X #_of_words    [ntokens=28782]

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            total_loss += loss.item()
            if batch % log_print_interval == 0 and batch > 0:
                lr = scheduler.get_last_lr()[0]
                ms_per_batch = (time.time() - start_time) * 1000 / log_print_interval
                cur_loss = total_loss / log_print_interval
                ppl = math.exp(cur_loss)
                print(f'| epoch {epoch:3d} | {batch:5d}/{num_sent_per_batches:5d} batches | '
                      f'lr {lr:02.2f} | ms/batch {ms_per_batch:5.2f} | '
                      f'loss {cur_loss:5.2f} | ppl {ppl:8.2f}')
                total_loss = 0
                start_time = time.time()

                if args.log_dir is not None:
                    train_logger.add_scalar("AvgLoss/Train", cur_loss, iter)

                scheduler.step()

        ##########################################################################
        # Evaluation of the model after each epoch
        model.eval()  # in evaluation mode
        total_loss = 0.
        len_each_batch = val_data.size(1) if batch_size_first else val_data.size(0)
        with torch.no_grad():
            for i in range(0, len_each_batch-1, sent_len):
                x_val_data_sent, val_targets = get_batch_x_y_sents(val_data, i, sent_len, batch_size_first)
                val_output, val_loss = model(x_val_data_sent, val_targets)
                total_loss += val_loss.item()

        val_avg_loss = total_loss / (len_each_batch - 1)
        val_ppl = math.exp(val_avg_loss)
        elapsed = time.time() - epoch_start_time
        print('-' * 89)
        print(f'| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | '
              f'valid loss {val_avg_loss:5.4f} | valid ppl {val_ppl:8.2f}')
        print('-' * 89)
        if args.log_dir is not None:
            valid_logger.add_scalar("AvgLoss/Eval", val_avg_loss, iter)

    # generate from the model
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    xx = model.generate(context, max_len_sentence_to_be_created=2000)
    generated_txt_in_lst = [_get_itos[i] for i in xx[0].tolist()]
    print(" ".join(generated_txt_in_lst))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("-epc", "--n_epochs", type=int, default=50, help="# of epochs")
    parser.add_argument("-bs", "--batch_size", type=int, default=20, help="batch size")  # 20
    parser.add_argument("-ebs", "--eval_batch_size", type=int, default=20, help="batch size for evaluation part")  # 10
    parser.add_argument("-esz", "--emb_size", type=int, default=300, help="embedding vector size")  # 200
    parser.add_argument("-ffd", "--dim_feedforward", type=int, default=200, help="Feed forward linear layer size")  # 200
    parser.add_argument("-nlr", "--n_layers", type=int, default=5, help="# of multi-attention layer in encoder")
    parser.add_argument("-nhd", "--n_head", type=int, default=5, help="# of multi-attention layer in encoder")  # 2
    parser.add_argument("-drp", "--dropout", type=float, default=0.2,
                        help="Dropout in feedforward layer within transformer")  # 0.2
    parser.add_argument("-sl", "--sent_len", type=int, default=35,
                        help="Length of sentences to be passed into the encoder network")  # 35
    parser.add_argument("-lr", "--learning_rate", type=float, default=5.0, help="Learning rate for optimizer")
    parser.add_argument("-lrs", "--lr_decay_factor", type=float, default=0.992, help="Decaying factor for lr") # 0.995

    parser.add_argument("--log_print_interval", type=int, default=500, help="interval to print some results") #200
    parser.add_argument("--batch_size_first", type=bool, default=True, help="If we want tensors with Batch size first.")

    parser.add_argument("--log_dir")

    args = parser.parse_args()
    train(args)
