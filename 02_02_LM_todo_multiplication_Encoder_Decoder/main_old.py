from os import path
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.utils.tensorboard as tb

from data_loader import data_loader, get_multiplication_answer
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
    vocab_size = args.vocab_size
    n_multiplication_digit = args.n_multiplication_digit

    train_logger, valid_logger = None, None
    if args.log_dir:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, "train"), flush_secs=1)
        valid_logger = tb.SummaryWriter(path.join(args.log_dir, "valid"), flush_secs=1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Train dataset
    # each loader is a list of len XXX. Each item has Batch# of string for multiplications
    train_data_loader_obj = data_loader(count=10000, begin=0, n_digit=n_multiplication_digit, batch_size=batch_size, shuffle=True)
    valid_data_loader_obj = data_loader(count=100, begin=100000, n_digit=n_multiplication_digit, batch_size=batch_size, shuffle=True)

    # encode_func, decode_func, vocab_size, train_data, val_data = get_train_test_data('input.txt')

    # data loading
    def _get_x_y(lst_str):
        """
        Get a list of size of batch size. Each element in this list batch# of strings.
        First convert it to torch.tensor, while padding with zero to get all into the same dimension. We convert the
        string into ascii values. This is why vocab_size=128.
        At the end shape x and y. For y, we only shift the char one to the right.
        x, y --> shape=[Bs, max_len_str_in_this_batch]
        """
        lengths = [len(s) for s in lst_str]
        length = max(lengths)
        data = torch.stack([torch.tensor(bytearray(s.ljust(length, '\0'), 'ascii'), device=device) for s in lst_str])

        x, y = data[:, :length-1], data[:, 1:length]
        x, y = x.to(device), y.to(device)
        return x, y

    def _calculate_loss(logits, targets):
        Bs, T, c_vocab_size = logits.shape  # (Bs,T,emb_size)
        logits = logits.view(Bs * T, c_vocab_size)
        targets = targets.contiguous().view(Bs * T)
        loss = F.cross_entropy(logits, targets)
        return loss

    model = BigramLanguageModel(vocab_size=vocab_size, emb_size=emb_size, sent_len=sent_len, n_head=n_head,
                                n_layer=n_layer, dropout=dropout)
    m = model.to(device)
    # print the number of parameters in the model
    print(sum(p.numel() for p in m.parameters()) / 1e6, 'M parameters')

    # create a PyTorch optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    for epoch in range(n_epochs):
        model.train()
        # get a list of batch_of_strings
        for lst_str in train_data_loader_obj:
            xb, yb = _get_x_y(lst_str)

            logits, _ = model(xb) # logits' shape=[Bs, max_len, vocab_size=128]
            loss_train = _calculate_loss(logits, yb)

            optimizer.zero_grad(set_to_none=True)
            loss_train.backward()
            optimizer.step()

        if epoch % loss_eval_epoch_interval == 0 or epoch == n_epochs - 1:
            generated_expr = model.generate_multiplication("12*21", sent_len, device=device)
            predicted_val = get_multiplication_answer(generated_expr)
            print(
                f"epoch {epoch}: train loss {loss_train:.4f}. Predicted multiplication "
                f"for 12*21 is {predicted_val}. The actual result is {str(12*21)}. ")
            print(f"Generate expression is {generated_expr}")
            if args.log_dir:
                train_logger.add_scalar("AvgLoss/Train", loss_train.detach().cpu().item(), epoch)

        perform_validate = False
        if perform_validate:
            model.eval()
            # get a list of batch_of_strings
            for lst_str in valid_data_loader_obj:
                xb, yb = _get_x_y(lst_str)

                logits, _ = model(xb)  # logits' shape=[Bs, max_len, vocab_size=128]
                loss_val = _calculate_loss(logits, yb)

            if epoch % loss_eval_epoch_interval == 0 or epoch == n_epochs - 1:
                print(f"epoch {epoch}: train loss {loss_val:.4f}")
                if args.log_dir:
                    valid_logger.add_scalar("AvgLoss/Valid", loss_val.detach().cpu().item(), epoch)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("-epc", "--n_epochs", type=int, default=10000, help="# of epochs")
    parser.add_argument("-bs", "--batch_size", type=int, default=128, help="batch size")
    parser.add_argument("-sl", "--sent_len", type=int, default=160,
                        help="Max length of string-sentences representing the multiplications") # for 5-digit = 1024, for 3-digit = 350, for 2-digits = 150
    parser.add_argument("-ebs", "--loss_eval_epoch_interval", type=int, default=5,
                        help="On what epoch interval to evaluate the avg loss")
    parser.add_argument("-lr", "--learning_rate", type=float, default=1e-3, help="Learning rate for optimizer")
    parser.add_argument("-esz", "--emb_size", type=int, default=128, help="embedding vector size") # 384/n_head = head_embed_size
    parser.add_argument("-vcs", "--vocab_size", type=int, default=128, help="vocab size")
    parser.add_argument("-ndg", "--n_multiplication_digit", type=int, default=2, help="number of digits of multipliers.")
    parser.add_argument("-nhd", "--n_head", type=int, default=4, help="# of multi-attention layer in encoder") #4
    parser.add_argument("-nlr", "--n_layer", type=int, default=4, help="# of multi-attention layer in encoder")
    parser.add_argument("-drp", "--dropout", type=float, default=0.2,
                        help="Dropout in feedforward layer within transformer")
    parser.add_argument("--log_dir")

    args = parser.parse_args()
    train(args)