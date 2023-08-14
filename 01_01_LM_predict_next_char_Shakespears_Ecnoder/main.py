from os import path
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.utils.tensorboard as tb

from data_loader import get_train_test_data
from model_simpler_version import BigramLanguageModel

torch.manual_seed(1337)


def train(args):
    batch_size = args.batch_size
    sent_len = args.sent_len
    n_epochs = args.n_epochs
    loss_eval_epoch_interval = args.loss_eval_epoch_interval
    learning_rate = args.learning_rate
    emb_size = args.emb_size
    n_head = args.n_head
    n_layer = args.n_layer
    dropout = args.dropout

    train_logger, valid_logger = None, None
    if args.log_dir:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, "train"), flush_secs=1)
        valid_logger = tb.SummaryWriter(path.join(args.log_dir, "valid"), flush_secs=1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    encode_func, decode_func, vocab_size, train_data, val_data = get_train_test_data('input.txt')

    # data loading
    def get_rand_batch_per_epoch(split):
        """
        generate a small random batch of data of inputs x and targets y, per training epoch.
        For instance, for a batch_size of 4, and whole text size of 100, we get 4 random index to pick 4 sentences starting
          at these 4 random indices:
                                    torch.randint(100, (4,)) => tensor([74, 15, 61, 94])
        """
        data = train_data if split == 'train' else val_data
        ix = torch.randint(len(data) - sent_len, (batch_size,))
        x = torch.stack([data[i:i + sent_len] for i in ix])
        y = torch.stack([data[i + 1:i + sent_len + 1] for i in ix])
        x, y = x.to(device), y.to(device)
        return x, y

    @torch.no_grad()
    def estimate_loss():
        """
        This is just for some sort of aggregation/averaging the loss to reduce the noise in the loss, and hence, better
        showing the loss graph.
        """
        out = {}
        model.eval()  # setting the model in eval, since we have dropout, ....
        for split in ['train', 'val']:
            losses = torch.zeros(loss_eval_epoch_interval)
            for k in range(loss_eval_epoch_interval):
                X, Y = get_rand_batch_per_epoch(split)  # X: Bs X sent_len && Y: Bs X sent_len
                logits, loss = model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        model.train()  # setting the model back to training mode
        return out

    model = BigramLanguageModel(vocab_size=vocab_size, emb_size=emb_size, sent_len=sent_len, n_head=n_head,
                                n_layer=n_layer, dropout=dropout, sinusoidal_pos_encoding=True)
    m = model.to(device)
    # print the number of parameters in the model
    print(sum(p.numel() for p in m.parameters()) / 1e6, 'M parameters')

    # create a PyTorch optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    for epoch in range(n_epochs):
        # every once in a while evaluate the loss on train and val sets
        if epoch % loss_eval_epoch_interval == 0 or epoch == n_epochs - 1:
            losses = estimate_loss()
            print(f"step {epoch}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            if args.log_dir:
                train_logger.add_scalar("AvgLoss/Train", losses['train'], epoch)
                valid_logger.add_scalar("AvgLoss/Valid", losses['val'], epoch)

        # sample a batch of data
        xb, yb = get_rand_batch_per_epoch('train')

        # evaluate the loss
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    # generate from the model
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    print(decode_func(m.generate(context, max_len_sentence_to_be_created=2000)[0].tolist()))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("-epc", "--n_epochs", type=int, default=10000, help="# of epochs")
    parser.add_argument("-bs", "--batch_size", type=int, default=64, help="batch size")
    parser.add_argument("-sl", "--sent_len", type=int, default=256,
                        help="Length of sentences to be passed into the encoder network")
    parser.add_argument("-ebs", "--loss_eval_epoch_interval", type=int, default=200,
                        help="On what epoch interval to evaluate the avg loss")
    parser.add_argument("-lr", "--learning_rate", type=float, default=1e-3, help="Learning rate for optimizer")
    parser.add_argument("-esz", "--emb_size", type=int, default=384, help="embedding vector size") # 384/n_head = head_embed_size
    parser.add_argument("-nhd", "--n_head", type=int, default=6, help="# of multi-attention head in encoder") #4
    parser.add_argument("-nlr", "--n_layer", type=int, default=6, help="# of multi-attention layer in encoder")
    parser.add_argument("-drp", "--dropout", type=float, default=0.2,
                        help="Dropout in feedforward layer within transformer")
    parser.add_argument("--log_dir")

    args = parser.parse_args()
    train(args)