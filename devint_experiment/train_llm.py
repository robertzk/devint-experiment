import json
import math
import os
import pickle
import random
import sys
import time
from json import JSONEncoder
from tempfile import TemporaryDirectory
from typing import List, NamedTuple

import einops
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import dataset

from devint_experiment.constants import CONTEXT_WINDOW_LENGTH, DEVICE
from devint_experiment.data import (get_batch, get_batched_data,
                                    get_wikitext_vocab_iterator)
from devint_experiment.model import (PositionalEncoding, TransformerModel,
                                     generate_square_subsequent_mask)


class TrainLogItem:

    def __init__(self, grads: dict, data_batch: List[str], targets: List[str],
                    epoch: int, batch_num: int, train_loss: int):
        self.grads = grads
        self.data_batch = data_batch
        self.targets = targets
        self.epoch = epoch
        self.batch_num = batch_num
        self.train_loss = train_loss

class TrainLogJSONEncoder(JSONEncoder):
    
    def default(self, obj):
        if type(obj) == Tensor:
            return obj.tolist()
        else:
                return obj.__dict__

def log_grads(prefix: str, grads: dict, include_embedding: bool=False) -> None:
    os.makedirs(prefix, exist_ok=True)
    for key, item in grads.items():
        if (not include_embedding) and key in ("encoder", "decoder"):
            continue
        if isinstance(item, dict):
            log_grads(os.path.join(prefix, key), item, include_embedding)
        elif isinstance(item, Tensor):
            with open(os.path.join(prefix, key), "wb") as f:
                    f.write(pickle.dumps(item))


if __name__ == "__main__":
    log_prefix = "single-step/"
    log_single_steps = True
    log_single_tokens = False
    # Log gradients for only these tokens if `log_single_tokens` is True.
    tokens_to_log = ("(", ")")

    # Whether or not to include the encoder/decoder units. These are considerably
    # larger than the rest so are typically worth switching off to avoid running
    # out of space.
    include_embedding = False
    log_embedding_every_n = 100

    vocab = get_wikitext_vocab_iterator()
    num_tokens = len(vocab)
    embedding_dimension = 200
    hidden_dimension = 200
    num_layers = 3
    num_heads = 4
    dropout = 0.2
    
    # Set to True for double-checking that recomputed single-datum gradients
    # match batch gradients when manually added and averaged. Will slow down
    # performance by a factor of 2x but worth running for a few steps to verify
    # single-datum capture is working as expected.
    apply_grad_sanity_check = False
    debug_after_train = False

    model = TransformerModel(num_tokens, embedding_dimension, num_heads,
                             hidden_dimension, num_layers, dropout).to(DEVICE)

    if log_single_steps:
        loss_criterion = nn.CrossEntropyLoss(reduction="none")
    else:
        loss_criterion = nn.CrossEntropyLoss()

    lr = 5.0
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)
    train_batch_size = 20
    eval_batch_size = 10

    epochs = 5

    train_data, val_data, test_data = get_batched_data(vocab, train_batch_size, eval_batch_size)

    def train(model: nn.Module, train_data, epoch: int=0, lr: float=5.0) -> None:
        model.train()
        total_loss = 0.
        log_interval = 200
        start_time = time.time()
        train_log: List[TrainLogItem] = []

        num_batches = len(train_data) // CONTEXT_WINDOW_LENGTH
        training_range = range(0, train_data.size(0) - 1, CONTEXT_WINDOW_LENGTH)

        # Apply a seed for deterministic training runs.
        random.seed(epoch * 42)

        # We provide `batch_shift_offsets` , which  shift each batch item by
        # the appropriate offset. This ensures batches don't end up identical
        # for different epochs (albeit traversed in different orders).
        batch_shift_offsets = [0] + [CONTEXT_WINDOW_LENGTH * int(random.uniform(0, num_batches)) for _ in range(train_batch_size - 1)]

        cur_lr = lr # Used to track current learning rate of scheduler.
        output = None

        if log_single_tokens:
            token_num_values = vocab(list(tokens_to_log))

        for batch, i in enumerate(random.sample(list(training_range), k=num_batches)):
            print(".", end="")
            sys.stdout.flush()

            src_mask = generate_square_subsequent_mask(CONTEXT_WINDOW_LENGTH).to(DEVICE)
            if batch >= num_batches - 2:
                # We are near the last or penultimate batch.
                # There is no easy way to adjust this when using batch_shift_offsets,
                # so just throw these data points away.
                continue
            else:
                data, targets = get_batch(train_data, i, CONTEXT_WINDOW_LENGTH, batch_shift_offsets)
                if data is None or targets is None:
                    continue

            grads_record = {}
            if not log_single_steps:
                # Do not log embeddings in single steps for now due to high
                # space storage utilization.
                if include_embedding and batch % log_embedding_every_n == 0:
                    grads_record["encoder_weights"] = next(model.encoder.parameters()).data.detach().clone()
                    grads_record["decoder_weights"] = next(model.decoder.parameters()).data.detach().clone()
                    grads_record["transformer_encoder_weights"] = {
                        name: param.data
                        for name, param in model.transformer_encoder.named_parameters()
                    }

            seq_len = data.size(0)
            if seq_len != CONTEXT_WINDOW_LENGTH:
                src_mask = src_mask[:seq_len, :seq_len]
            output = model(data, src_mask)

            if log_single_steps:
                loss = loss_criterion(output.view(-1, num_tokens), targets)

                total_grads = {
                    name: torch.zeros_like(param) for name, param in model.named_parameters()
                }

                # At this point, `targets` and `loss` have been flattened such that
                # the tokens for the ith data point occur at every `train_batch_size`
                # position. For example, if `train_batch_size` == 20, then 
                # `targets[::20]` contains the target tokens for the first datum.
                # To store gradient attribution for a datum, we store the
                # averaged gradient over the context window (`CONTEXT_WINDOW_LENGTH`).
                # Note `loss.shape[0] == train_batch_size * CHUNK_LENGHTS`.
                for i in range(train_batch_size):
                    new_grads = {
                        name: torch.zeros_like(param) for name, param in model.named_parameters()
                        if include_embedding or name not in ("encoder.weight", "decoder.weight", "decoder.bias")
                    }
                    for j in range(CONTEXT_WINDOW_LENGTH):
                        k = j * train_batch_size + i
                        log_immediately = log_single_tokens and j > 0 and (int(data[j, i]) in token_num_values or int(targets[k]) in token_num_values)

                        new_grads = {
                            name: torch.zeros_like(param) for name, param in model.named_parameters()
                            if include_embedding or name not in ("encoder.weight", "decoder.weight", "decoder.bias")
                        }

                        loss[k].backward(retain_graph=True)

                        # In the batch step, this is where we could perform gradient clipping
                        # as per below. Instead, we do this after applying all single-datum grads.
                        #   torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                        for name, param in model.named_parameters():
                            if include_embedding or name not in ("encoder.weight", "decoder.weight", "decoder.bias"):
                                new_grads[name].add_(param.grad.data * -cur_lr / loss.size(0))
                            total_grads[name].add_(param.grad.data)
                        
                        if log_immediately:
                            log_grads(f"{log_prefix}grads_{epoch}/{batch}/{i}/{j}",
                                      new_grads, include_embedding and batch % log_embedding_every_n == 0)
                        model.zero_grad()
                    
                    if not log_single_tokens:
                        log_grads(f"{log_prefix}grads_{epoch}/{batch}/{i}",
                                  new_grads, include_embedding and batch % log_embedding_every_n == 0)
                
                # Conserve on some space. The backwards graph keeps references to these
                # so delete them explicitly to trigger GC on the GPU.
                for key in list(new_grads.keys()):
                    del new_grads[key]
                del new_grads

                if DEVICE == "cuda":
                    torch.cuda.empty_cache()
                
                # First we apply the averaged gradient, then we clip norms.
                for name, param in model.named_parameters():
                    param.grad = total_grads[name] / loss.size(0)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                
                def sanity_check(loss_size):
                    # Verify that the individual summed grads are equal
                    # to a single pass with a CrossEntropyLoss(reduction="mean").
                    model.zero_grad()
                    loss_criterion2 = nn.CrossEntropyLoss(reduction="mean")
                    loss2 = loss_criterion2(output.view(-1, num_tokens), targets)
                    loss2.backward(retain_graph=True)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                    params = {name: param for name, param in model.named_parameters() } 
                    assert all(
                        ((params[name].grad - total_grads[name] / loss_size).abs() < 1.5e-4).all()
                        for name, _ in model.named_parameters()
                        if name != "decoder.weight"
                    )

                if apply_grad_sanity_check:
                    sanity_check(loss.size(0))
                
                optimizer.step()
                curr_loss = loss.mean().item()
                if not math.isnan(curr_loss):
                    total_loss += curr_loss

                # Clean up some more GPU mem usage explicitly.
                del loss
                for key in list(total_grads.keys()):
                    del total_grads[key]
                del total_grads

            else:
                loss = loss_criterion(output.view(-1, num_tokens), targets)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)

                optimizer.step()
                curr_loss = loss.item()
                total_loss += curr_loss

            cur_lr = scheduler.get_last_lr()[0]

            if not log_single_steps:
                grads_record["encoder"] = next(model.encoder.parameters()).grad.data * -cur_lr
                grads_record["transformer_encoder"] = {
                    name: param.grad.data * -cur_lr
                    for name, param in model.transformer_encoder.named_parameters()
                }
                grads_record["decoder"] = next(model.decoder.parameters()).grad.data * -cur_lr

                log_grads(f"{log_prefix}grads_{epoch}/{batch}", grads_record, include_embedding and batch % log_embedding_every_n == 0)

            # Convert training data and target tokens to strings.
            data_batch = [vocab.lookup_tokens(data[:, i].tolist()) for i in range(data.shape[1])]
            targets_str = vocab.lookup_tokens(targets.tolist())
            train_log.append(TrainLogItem({}, data_batch, targets_str, epoch, batch, curr_loss))

            if batch % log_interval == 0 and batch > 0:
                lr = scheduler.get_last_lr()[0]
                ms_per_batch = (time.time() - start_time) * 1_000 / log_interval
                cur_loss = total_loss / log_interval
                try:
                    ppl = math.exp(cur_loss)
                except:
                    ppl = -1.0
                print(f"| epoch {epoch:3d} | {batch:5d}/{num_batches:5d} batches | " + 
                       f"|lr {lr:02.02f} | ms/batch {ms_per_batch:5.2f} |" + 
                       f"loss {cur_loss:5.2f} | ppl {ppl:8.2f}")
                total_loss = 0
                start_time = time.time()

            with open(f"{log_prefix}grads_{epoch}.json", "w") as f:
                json.dump(train_log, f, cls=TrainLogJSONEncoder)

    def evaluate(model: nn.Module, eval_data: Tensor) -> float:
        model.eval()
        total_loss = 0.
        src_mask = generate_square_subsequent_mask(CONTEXT_WINDOW_LENGTH).to(DEVICE)

        with torch.no_grad():
            for i in range(0, eval_data.size(0) - 1, CONTEXT_WINDOW_LENGTH):
                data, targets = get_batch(eval_data, i)
                if data.shape[0] == 0:
                    continue
                seq_len = data.size(0)
                if seq_len != CONTEXT_WINDOW_LENGTH:
                    src_mask = src_mask[:seq_len, :seq_len]
                output = model(data, src_mask)
                output_flat = output.view(-1, num_tokens)
                loss_criterion = nn.CrossEntropyLoss()
                total_loss += seq_len * loss_criterion(output_flat, targets).item()
        return total_loss / (len(eval_data) - 1)

    best_val_loss = float("inf")

    print("Starting model training\n")
    with TemporaryDirectory() as tempdir:
        best_model_params_path = os.path.join(tempdir, "best_model_params.pt")

        for epoch in range(1, epochs + 1):
            epoch_start_time = time.time()
            train(model, train_data, epoch, lr)
            try:
                val_loss = evaluate(model, val_data)
                val_ppl = math.exp(val_loss)
                elapsed = time.time() - epoch_start_time

                print("-" * 89)
                print(f"| end of epoch {epoch:3d} | time: {elapsed: 5.2f}s | " +
                    f"valid loss {val_loss:5.2f} | valid ppl {val_ppl:8.2f}")
                print("-" * 89)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(model.state_dict(), best_model_params_path)
            except:
                print(f"Exception during evaluate: {e}")
            
            scheduler.step()
        
        model.load_state_dict(torch.load(best_model_params_path))
    
    test_loss = evaluate(model, test_data)
    test_ppl = math.exp(test_loss)
    print("=" * 89)
    print(f"| End of training | test loss {test_loss:5.2f}" +
          f"| test ppl {test_ppl:8.2f}")
    print("=" * 89)

    torch.save(model.state_dict(), f"{log_prefix}trained_model.pt")

    # Leave a debugger on in case we want to examine any local variables
    # after training.
    if debug_after_train:
        import pdb; pdb.set_trace()

