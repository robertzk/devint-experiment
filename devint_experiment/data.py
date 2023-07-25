import torch
from torch import nn, Tensor
from torchtext.datasets import WikiText2
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator, Vocab
from torch.utils.data import IterableDataset
from typing import List, Optional, Tuple

from devint_experiment.constants import CONTEXT_WINDOW_LENGTH, DEVICE

def get_wikitext_vocab_iterator() -> Vocab:
    train_iter = WikiText2(split="train")
    tokenizer = get_tokenizer("basic_english")
    vocab = build_vocab_from_iterator((tokenizer(x) for x in train_iter), specials=["<unk>"])
    vocab.set_default_index(vocab["<unk>"])
    return vocab

def get_data_tensor(vocab: Vocab, raw_text_iter: IterableDataset) -> Tensor:
    """Converts raw text into a flat Tensor."""
    tokenizer = get_tokenizer("basic_english")
    data = [torch.tensor(vocab(tokenizer(item)), dtype=torch.long) for item in raw_text_iter]
    return torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))

def batchify(data: Tensor, batch_size: int) -> Tensor:
    """
    Divides the data into ``batch_size`` separate sequences, removing extra elements
    that would not cleanly fit.
    
    Arguments:
        data: Tensor, shape [N]
        batch_size: int, batch size
    
    Returns:
        Tensor of shape ``[N // batch_size, batch_size]``
    """
    seq_len = data.size(0) // batch_size
    data = data[:seq_len * batch_size]
    data = data.view(batch_size, seq_len).t().contiguous()
    return data.to(DEVICE)

def get_batched_data(vocab: Vocab, batch_size: int = 20, eval_batch_size: int = 10) -> Tuple[Tensor, Tensor, Tensor]:
    train_iter, val_iter, test_iter = WikiText2()
    train_data = get_data_tensor(vocab, train_iter)
    val_data = get_data_tensor(vocab, val_iter)
    test_data = get_data_tensor(vocab, test_iter)

    train_data = batchify(train_data, batch_size)
    val_data = batchify(val_data, eval_batch_size)
    test_data = batchify(test_data, eval_batch_size)
    return train_data, val_data, test_data

def get_batch(source: Tensor, i: int, chunk_lengths: int = CONTEXT_WINDOW_LENGTH, batch_shift_offsets: Optional[List[int]] = None) -> Tuple[Tensor, Tensor]:
    """
    Args:
        source: Tensor, shape ``[full_seq_len, batch_size]``
        i: int
        batch_shift_offsets: Optional[List[int]]
    
        Returns:
            tuple (data, target), where data has shape ``[seq_len, batch_size]`` and
            target has shape ``[seq_len * batch_size]``.

            If `batch_shift_offsets` are provided, we shift each batch item by
            the appropriate offset. This ensures batches don't end up identical
            for different epochs (albeit traversed in different orders).
    """
    if batch_shift_offsets is not None:
        tensors = []
        targets = []
        for j in range(len(batch_shift_offsets)):
            new_ix = i + batch_shift_offsets[j]
            if new_ix >= source.shape[0]:
                new_ix -= source.shape[0]
            seq_len = min(chunk_lengths, len(source) - chunk_lengths - new_ix)
            if seq_len != chunk_lengths:
                # We are near the last or penultimate batch.
                # There is no easy way to fix this with batch_shift_offsets,
                # so just throw this data point away and re-use the previous one.
                if len(tensors) == 0 or len(targets) == 0:
                    return None, None
                tensors.append(tensors[-1])
                targets.append(targets[-1])
            else:
                tensors.append(source[new_ix:new_ix + seq_len, j])
                targets.append(source[(new_ix + 1):(new_ix + 1 + seq_len), j])

        data = torch.concat(tensors).reshape([source.shape[1], chunk_lengths]).t()
        target = torch.concat(targets).reshape([source.shape[1], chunk_lengths]).t().reshape(-1)
        return data, target
    else:
        seq_len = min(chunk_lengths, len(source) - chunk_lengths - i)
        data = source[i:i + seq_len]
        target = source[(i + 1):(i + 1 + seq_len)].reshape(-1)
        return data, target
