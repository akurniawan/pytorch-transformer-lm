import os
import torch

from collections import namedtuple

from torchtext import data
from torchtext import datasets
from .iterator import BPTTIterator

_LMDataReturn = namedtuple(
    "WikiTextReturn", ["train_iter", "valid_iter", "test_iter", "text_field"])


def wikitext2(batch_size, bptt_len):
    TEXT = data.Field(lower=True)
    train, valid, test = datasets.WikiText2.splits(TEXT)

    if os.path.exists("./checkpoint/vocab.data"):
        TEXT.vocab = torch.load("./checkpoint/vocab.data")
    else:
        TEXT.build_vocab(train)
        # torch.save(TEXT.vocab, "./checkpoint/vocab.data")

    train_iter, valid_iter, test_iter = BPTTIterator.splits(
        (train, valid, test),
        batch_size=batch_size,
        bptt_len=bptt_len,
        repeat=False)

    return _LMDataReturn(train_iter, valid_iter, test_iter, TEXT)


def wikitext103(batch_size, bptt_len):
    TEXT = data.Field(lower=True)
    train, valid, test = datasets.WikiText103.splits(TEXT)

    if os.path.exists("./checkpoint/vocab.data"):
        TEXT.vocab = torch.load("./checkpoint/vocab.data")
    else:
        TEXT.build_vocab(train)
        # torch.save(TEXT.vocab, "./checkpoint/vocab.data")

    train_iter, valid_iter, test_iter = data.BPTTIterator.splits(
        (train, valid, test),
        batch_size=batch_size,
        bptt_len=bptt_len,
        repeat=False)

    return _LMDataReturn(train_iter, valid_iter, test_iter, TEXT)
