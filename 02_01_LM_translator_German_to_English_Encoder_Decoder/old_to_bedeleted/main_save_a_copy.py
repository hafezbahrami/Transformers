from os import path
from timeit import default_timer as timer
import torch
import torch.nn as nn
import torch.utils.tensorboard as tb

from torchtext.datasets import Multi30k

from data_loader import get_token_vocab, data_loader, german_to_english_samples
from transformer_model_torch_tutorial import Seq2SeqTransformer, translate_torch_tutorial
from transformers_encoder_decoder import TransformerModel, translate, create_mask

torch.manual_seed(0)


def train(args):
    src_language = args.src_language
    tgt_language = args.tgt_language
    emb_size = args.emb_size
    n_head = args.n_head
    batch_size = args.batch_size
    n_layers = args.n_layers
    n_epochs = args.n_epochs

    if emb_size % n_head != 0:
        emb_size = (emb_size//n_head) * n_head
        print("embed size is changes to have a compatible head numbers in self_attention within the transformer.")

    train_logger, valid_logger = None, None
    if args.log_dir:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, "train"), flush_secs=1)
        valid_logger = tb.SummaryWriter(path.join(args.log_dir, "valid"), flush_secs=1)

    # Make sure the tokens are in order of their indices to properly insert them in vocab
    special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']
    special_symbol_idx = {"UNK_IDX": 0, "PAD_IDX": 1, "BOS_IDX": 2, "EOS_IDX": 3}

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tokenizer, vocab = get_token_vocab(src_language=src_language,
                                       tgt_language=tgt_language,
                                       special_symbols=special_symbols,
                                       special_symbol_idx=special_symbol_idx)

    src_vocab_size = len(vocab[src_language]) # > 19k words in German
    tgt_vocab_size = len(vocab[tgt_language]) # > 10k words in English

    if args.torch_tutorial_model:
        print("*** Using Pytorch Tutorial Transformer model")
        model = Seq2SeqTransformer(num_encoder_layers=n_layers, num_decoder_layers=n_layers, emb_size=emb_size,
                                    nhead=n_head, src_vocab_size=src_vocab_size, tgt_vocab_size=tgt_vocab_size,
                                    dim_feedforward=args.ffn_hid_dim, dropout=args.dropout)
    else:
        print("*** Using locally written Transformer model")
        model = TransformerModel(vocab_size_enc=src_vocab_size, vocab_size_dec=tgt_vocab_size, emb_size=emb_size,
                                    n_head=n_head, n_layers=n_layers, dropout=args.dropout, d_ff=args.ffn_hid_dim)

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    model = model.to(device)

    pad_idx = special_symbol_idx["PAD_IDX"]
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=pad_idx)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.98), eps=1e-9)

    ######################################################################
    for epoch in range(n_epochs):
        model.train()
        train_losses = 0
        train_iter = Multi30k(split='train', language_pair=(src_language, tgt_language))
        train_dataloader = data_loader(data_iter=train_iter, batch_size=batch_size,
                                       special_symbol_idx=special_symbol_idx, tokenizer=tokenizer, vocab=vocab,
                                       src_language=src_language, tgt_language=tgt_language,
                                       shuffle=True)

        for src, tgt in train_dataloader:
            """
            Go and call the collate_fn function to get transformed batch sentences. Sentences will be in vertical 
            columns: src: src_sent_len X Bs  && tgt: tgt_sent_len X Bs
            """
            src, tgt = src.to(device), tgt.to(device) # we have padded (same length) src. Similarly, for tgt.
            tgt_input = tgt[:-1, :]

            src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input, pad_idx, device)

            if args.torch_tutorial_model:
                if args.turn_off_padding:
                    src_padding_mask, tgt_padding_mask, src_padding_mask = None, None, None
                logits = model(src, tgt_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask) # Logits: tgt_sent_len X Bs X tgt_vocab_size. tgt_vocab_size is the # of output classes
            else:
                logits = model(src, tgt_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask) # Logits: tgt_sent_len X Bs X tgt_vocab_size. tgt_vocab_size is the # of output classes

            optimizer.zero_grad()

            tgt_out = tgt[1:, :] # tgt_sent_length X Bs ==> ground truth for classes
            loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
            loss.backward()

            optimizer.step()
            train_losses += loss.item()

        train_loss = train_losses / len(list(train_dataloader))

        #################################################################
        # EVALUATION
        model.eval()
        eval_losses = 0

        val_iter = Multi30k(split='valid', language_pair=(src_language, tgt_language))
        val_dataloader = data_loader(data_iter=val_iter, batch_size=batch_size,
                                    special_symbol_idx=special_symbol_idx, tokenizer=tokenizer, vocab=vocab,
                                    src_language=src_language, tgt_language=tgt_language,
                                    shuffle=True)

        for src, tgt in val_dataloader:
            src, tgt = src.to(device), tgt.to(device)
            tgt_input = tgt[:-1, :]

            src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input, pad_idx, device)

            if args.torch_tutorial_model:
                if args.turn_off_padding:
                    src_padding_mask, tgt_padding_mask, src_padding_mask = None, None, None
                logits = model(src, tgt_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask)

            else:
                logits = model(src, tgt_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask)

            tgt_out = tgt[1:, :]

            loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
            eval_losses += loss.item()

        val_loss = eval_losses / len(list(val_dataloader))

        print(f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}")
        if args.log_dir is not None:
            train_logger.add_scalar("AvgLoss/Train", train_loss, epoch)
        if args.log_dir is not None:
            valid_logger.add_scalar("AvgLoss/Valid", val_loss, epoch)

    if args.torch_tutorial_model:
        translate_func = translate_torch_tutorial
    else:
        translate_func = translate

    samples_ger_to_en = german_to_english_samples()
    for german_sent, eng_sent in samples_ger_to_en.items():
        translated_sents = translate_func(model=model,
                        tgt_language=tgt_language, src_language=src_language,
                        src_sentence=german_sent,
                        special_symbol_idx=special_symbol_idx,
                        tokenizer=tokenizer,
                        vocab=vocab,
                        device=device,
                        beam_search_generation_approach = args.beam_search_text_generation,
                        beam_size=args.beam_search_beam_size,
                        n_results=args.beam_search_n_results,
                        max_length=args.beam_search_max_length,
                        )
        print(f"\"{german_sent}\" ==> Predicted Sentence: \"{translated_sents}\". ==> Ground Truth:  \"{eng_sent}\"")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("-epc", "--n_epochs", type=int, default=1, help="# of epochs") # 30
    parser.add_argument("-bs", "--batch_size", type=int, default=2, help="batch size")  # 128
    parser.add_argument("-lr", "--learning_rate", type=float, default=1.E-4, help="Learning rate for optimizer")
    parser.add_argument("-drp", "--dropout", type=float, default=0.2,
                        help="Dropout in feedforward layer within transformer")  # 0.2
    parser.add_argument("-srl", "--src_language", type=str, default="de", help="Source language that we want to "
                                            "translate it to another one. de is for Germann, and en for English.")
    parser.add_argument("-tgl", "--tgt_language", type=str, default="en", help="Target language that we want to "
                                        "translate the source language into. de is for Germann, and en for English.")
    parser.add_argument("-esz", "--emb_size", type=int, default=6, help="embedding vector size")  # 512
    parser.add_argument("-nhd", "--n_head", type=int, default=2, help="# of heads within self_attention")  # 8
    parser.add_argument("-ffd", "--ffn_hid_dim", type=int, default=10, help="Feed forward linear layer size")  # 512
    parser.add_argument("-ndl", "--n_layers", type=int, default=3, help="# of self-attention layer in decoder and encoder") #6
    parser.add_argument("--torch_tutorial_model", type=bool, default=False, help="Use torch tutorial transformer model")
    parser.add_argument("--turn_off_padding", type=bool, default=False, help="Turn off the padding in pytorch tutorial transformer model")

    parser.add_argument("--beam_search_text_generation", type=bool, default=True, help="Use beam search to generate "
                                                    "text after training model, in opposed to using greedy approach.")
    parser.add_argument("--beam_search_beam_size", type=int, default=50, help="beam size search")
    parser.add_argument("--beam_search_n_results", type=int, default=5, help="number of final results kept after beam search")
    parser.add_argument("--beam_search_max_length", type=int, default=50, help="Max length of a sentence in a beam search before hitting a end of sentences sign.")

    parser.add_argument("--log_dir")

    args = parser.parse_args()
    train(args)
