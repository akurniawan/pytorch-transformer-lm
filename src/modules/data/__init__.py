import os
import torch

from torchtext import data


class Dataset(object):
    def __init__(self, batch_size, bptt_len, dataset_cls):
        TEXT = data.Field(lower=True)
        train, valid, test = dataset_cls.splits(TEXT)

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

        self.tex_field = TEXT
        self.train_iter = train_iter
        self.valid_iter = valid_iter
        self.text_iter = test_iter
