import math
from torchtext.data import Iterator
from torchtext.data import Batch
from torchtext.data import Dataset


class BPTTIterator(Iterator):
    """Defines an iterator for language modeling tasks that use BPTT.

    Provides contiguous streams of examples together with targets that are
    one timestep further forward, for language modeling training with
    backpropagation through time (BPTT). Expects a Dataset with a single
    example and a single field called 'text' and produces Batches with text and
    target attributes.

    Attributes:
        dataset: The Dataset object to load Examples from.
        batch_size: Batch size.
        bptt_len: Length of sequences for backpropagation through time.
        sort_key: A key to use for sorting examples in order to batch together
            examples with similar lengths and minimize padding. The sort_key
            provided to the Iterator constructor overrides the sort_key
            attribute of the Dataset, or defers to it if None.
        train: Whether the iterator represents a train set.
        repeat: Whether to repeat the iterator for multiple epochs. Default: False.
        shuffle: Whether to shuffle examples between epochs.
        sort: Whether to sort examples according to self.sort_key.
            Note that repeat, shuffle, and sort default to train, train, and
            (not train).
        device (str or torch.device): A string or instance of `torch.device`
            specifying which device the Variables are going to be created on.
            If left as default, the tensors will be created on cpu. Default: None.
    """

    def __init__(self, dataset, batch_size, bptt_len, **kwargs):
        self.bptt_len = bptt_len
        super(BPTTIterator, self).__init__(dataset, batch_size, **kwargs)

    def __len__(self):
        return math.ceil(
            (len(self.dataset[0].text) / self.batch_size - 1) / self.bptt_len)

    def __iter__(self):
        text = self.dataset[0].text
        TEXT = self.dataset.fields['text']
        TEXT.eos_token = None
        text = text + ([TEXT.pad_token] * int(
            math.ceil(len(text) / self.batch_size) * self.batch_size -
            len(text)))
        data = TEXT.numericalize([text], device=self.device)
        data = data.view(self.batch_size, -1).t().contiguous()
        dataset = Dataset(
            examples=self.dataset.examples,
            fields=[('text', TEXT), ('target', TEXT)])
        while True:
            for i in range(0, len(self) * self.bptt_len):
                self.iterations += 1
                seq_len = min(self.bptt_len, len(data) - i - 1)
                yield Batch.fromvars(
                    dataset,
                    self.batch_size,
                    text=data[i:i + seq_len],
                    target=data[i + 1:i + 1 + seq_len])
            if not self.repeat:
                return