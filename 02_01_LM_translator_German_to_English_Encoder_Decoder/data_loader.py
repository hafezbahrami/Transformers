# Data Sourcing and Processing
# ----------------------------
#
# `torchtext library <https://pytorch.org/text/stable/>`__ has utilities for creating datasets that can be easily
# iterated through for the purposes of creating a language translation
# model. In this example, we show how to use torchtext's inbuilt datasets,
# tokenize a raw text sentence, build vocabulary, and numericalize tokens into tensor. We will use
# `Multi30k dataset from torchtext library <https://pytorch.org/text/stable/datasets.html#multi30k>`__
# that yields a pair of source-target raw sentences.
#
# To access torchtext datasets, please install torchdata following instructions at https://github.com/pytorch/data.
#
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

torch.utils.data.datapipes.utils.common.DILL_AVAILABLE = torch.utils._import_utils.dill_available()  # Was added due to some error in loading Multi30K dataset from torchtext

from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.datasets import multi30k, Multi30k
from typing import Iterable, List


# We need to modify the URLs for the dataset since the links to the original dataset are broken
# Refer to https://github.com/pytorch/text/issues/1756#issuecomment-1163664163 for more info
# "multi30k" is an extension of the Flickr30K dataset (Young et al., 2014) with 31,014 German translations of English
# descriptions and 155,070 independently collected German descriptions.
multi30k.URL["train"] = "https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/training.tar.gz"
multi30k.URL["valid"] = "https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/validation.tar.gz"


def get_token_vocab(src_language: str = "de", tgt_language: str = "en", special_symbols=['<unk>', '<pad>', '<bos>', '<eos>'],
              special_symbol_idx={"UNK_IDX": 0, "PAD_IDX": 1, "BOS_IDX": 2, "EOS_IDX": 3}):
    """
    Below are some examples of how to use them:
    (I)
            for item in train_iter:
                print(item)
                break
    
    (II)
    len(vocab["de"]) # > 19k words in German  ==> to for the embedding table
    
    """
    # Place-holders
    tokenizer = {}
    vocab = {}

    ###################################################################################
    # Create source and target language tokenizer. Make sure to install the dependencies.
    #
    # .. code-block:: python
    #
    #    pip install -U torchdata
    #    pip install -U torchtext
    #    pip install -U portalocker        
    #    pip install -U spacy
    #    python -m spacy download en_core_web_sm
    #    python -m spacy download de_core_news_sm

    # we should also install/download: python -m spacy download en_core_web_sm  
    # we should also install/download: python -m spacy download de_core_news_sm
    
    # (1) Building tokenizer
    tokenizer[src_language] = get_tokenizer('spacy', language='de_core_news_sm') # => Example: tokenizer["en"]("I am good") => ['I', 'am', 'good']
    tokenizer[tgt_language] = get_tokenizer('spacy', language='en_core_web_sm')

    # (2) Building vocab
    language_index = {src_language: 0, tgt_language: 1}
    # helper function to yield list of tokens
    def yield_tokens(data_iter: Iterable, language: str) -> List[str]:
        ln_idx=language_index[language]
        
        for data_sample in data_iter:                               # data_sample => ('Zwei junge ....', 'Two youg ....')
            X_sentence = data_sample[ln_idx]                        # 'Zwei junge ...'
            tokenized_sentence = tokenizer[language](X_sentence)    # ['Zwei', 'junge', ...]
            yield tokenized_sentence

    # Training data Iterator
    train_iter = Multi30k(split='train', language_pair=(src_language, tgt_language)) # > 30k training sentences
    for ln in [src_language, tgt_language]:
        # Create torchtext's Vocab object
        vocab[ln] = build_vocab_from_iterator(yield_tokens(train_iter, ln),
                                                        min_freq=1,
                                                        specials=special_symbols,
                                                        special_first=True)

    # Set ``UNK_IDX`` as the default index. This index is returned when the token is not found.
    # If not set, it throws ``RuntimeError`` when the queried token is not found in the Vocabulary.
    for ln in [src_language, tgt_language]:
        vocab[ln].set_default_index(special_symbol_idx["UNK_IDX"])

    return tokenizer, vocab, train_iter


def get_transformed_text(tokenizer, vocab, src_language, tgt_language, special_symbol_idx):
    """This method is similar to Data Augmentation in Computer Vision"""
    
    def sequential_transforms(*transforms):
        """helper function to club together sequential operations"""
        def func(txt_input):
            for transform in transforms:
                txt_input = transform(txt_input)
            return txt_input

        return func

    def tensor_transform(token_ids: List[int]):
        """function to add BOS/EOS and create tensor for input sequence indices"""
        bos_idx = special_symbol_idx["BOS_IDX"]
        eos_idx = special_symbol_idx["EOS_IDX"]
        return torch.cat((torch.LongTensor([bos_idx]),
                          torch.LongTensor(token_ids),
                          torch.LongTensor([eos_idx])))

    # ``src`` and ``tgt`` language text will be transformed from raw strings into tensors indices
    transformed_text = {}
    for ln in [src_language, tgt_language]:
        transformed_text[ln] = sequential_transforms(tokenizer[ln],  # Tokenization
                                                   vocab[ln],  # Numericalization
                                                   tensor_transform)  # Add BOS/EOS and create tensor

    return transformed_text

def data_loader(data_iter, batch_size, special_symbol_idx, 
                tokenizer, vocab, 
                src_language, tgt_language,
                shuffle):
    # transformed_text["en"]("I am book and he is not XXX") ==> 
    #                              tensor([   2, 1166, 3426,  285,   11,  210,   10,  978,    0,    3]). 
    # The EOS gets index=2, EOS gets index=3, and the unknown word, XXX, hets index=0
    transformed_text = get_transformed_text(tokenizer, vocab, src_language, tgt_language, special_symbol_idx)

    # function to collate data samples into batch tensors
    def collate_fn(batch_sents):
        """
        General: Why collate_fn: https://stackoverflow.com/questions/65279115/how-to-use-collate-fn-with-dataloaders
                 This method will get called every time within train-epoc we want a batch of data/sentences out of our
                 already-created train-iter or eval-iter
                 collate_fn will act similar to __get_item__ in DataLoader

        Example:
        If we assume batch size is 1, then the batch_sents comes in like:
        [('Zwei junge weiße Männer sind im Freien in der Nähe vieler Büsche.',
          'Two young, White males are outside near many bushes.')]
        Then, we simply, convert this sentences into a series of transformation, to get the final transformed text:
        src_batch[0] => tensor([   2,   21,   85,  257,   31,   87,   22,   94,    7,   16,  112, 7910,
                                   3209,    4,    3])
        tgt_batch[0] => tensor([   2,   19,   25,   15, 1169,  808,   17,   57,   84,  336, 1339,    5,
                                   3])
        """
        src_batch, tgt_batch = [], []
        for src_sample, tgt_sample in batch_sents: #for Bs=3, batch_sents: List[(src1,tgt1), (src1,tgt1), (src1,tgt1)]
            src_batch.append(transformed_text[src_language](src_sample.rstrip("\n")))
            tgt_batch.append(transformed_text[tgt_language](tgt_sample.rstrip("\n")))

        # as an example, if the src_batch includes 3 sentences (for Bs=3), they will have different length. In the next
        # section, we simply pad all smaller-length sentences to the length og longest sentence.
        pad_idx = special_symbol_idx["PAD_IDX"]
        src_batch = pad_sequence(src_batch, padding_value=pad_idx)
        tgt_batch = pad_sequence(tgt_batch, padding_value=pad_idx)
        return src_batch, tgt_batch
    data_loader_obj = DataLoader(data_iter, batch_size=batch_size, collate_fn=collate_fn, shuffle=shuffle)

    return data_loader_obj


def german_to_english_samples():
    samples_german_to_en = {}
    samples_german_to_en["Ich möchte heute nach der Arbeit einen Freund besuchen."] = "I want to go to see a friend today after work."
    samples_german_to_en["Ich treibe Sport, um mich zu entspannen."] = "I play sport to relax."
    samples_german_to_en["Eine Gruppe von Menschen steht vor einem Iglu"] = "A group of people stand in front of an igloo."
    samples_german_to_en["Es ist mein Traum, meinen Vater und meine Onkel noch einmal zu sehen."] = "It is my dream to see my dad and uncles one more time."
    return samples_german_to_en


