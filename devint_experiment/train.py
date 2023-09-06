import json
import math
import os
import random
from tempfile import TemporaryDirectory
import time
import torch
from torch import Tensor, nn
from torchtext.vocab import Vocab
from typing import List
import sys

from devint_experiment.constants import (CONTEXT_WINDOW_LENGTH, DEVICE)
from devint_experiment.data import (get_batch, get_batched_data, get_wikitext_vocab_iterator)
from devint_experiment.model import (TransformerModel, generate_square_subsequent_mask)
from devint_experiment.train_log import (TrainLogItem, TrainLogJSONEncoder, log_grads)

def progress_printer(*args, wrapper: bool = False):
    if wrapper:
        print("-" * 89)
    print("| " + " | ".join(args))
    if wrapper:
        print("-" * 89)

class Trainer:

    def __init__(self, log_prefix: str, log_single_steps: bool, log_single_tokens: bool,
                 include_embedding: bool, vocab: Vocab, log_embedding_every_n: int = 100,
                 embedding_dimension: int =  200, hidden_dimension: int = 200, 
                 num_layers: int = 3, num_heads: int = 4, dropout: float = 0.2,
                 lr: float = 5.0, lr_step: float = 1.0, lr_gamma: float = 0.95,
                 train_batch_size: int = 20, eval_batch_size: int = 10, epochs: int = 5,
                 *, tokens_to_log: List[str] = ("(", ")"), apply_grad_sanity_check: bool = False,
                 debug_after_train: bool = False):
        self.log_prefix = log_prefix
        self.log_single_steps = log_single_steps
        self.log_single_tokens = log_single_tokens
        self.tokens_to_log = tokens_to_log

        self.include_embedding = False
        self.log_embedding_every_n = log_embedding_every_n
        
        self.vocab = vocab
        self.num_tokens = len(vocab)
        self.embedding_dimension = embedding_dimension
        self.hidden_dimension = hidden_dimension
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout

        self.apply_grad_sanity_check = apply_grad_sanity_check
        self.debug_after_train = debug_after_train

        self.initialize_model()

        self.lr = lr
        # Only use cross-entropy loss for now.
        if log_single_steps:
            self.loss_criterion = nn.CrossEntropyLoss(reduction="none")
        else:
            self.loss_criterion = nn.CrossEntropyLoss()

        # Only use SGD for now.
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)

        self.lr_step = lr_step
        self.lr_gamma = lr_gamma
        scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, lr_step, gamma=lr_gamma)
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size

        self.epochs = epochs

    def initialize_model(self):
        self.model = TransformerModel(self.num_tokens, self.embedding_dimension, self.num_heads,
                                      self.hidden_dimension, self.num_layers, self.dropout).to(DEVICE)
    
    def train(self):
        train_data, val_data, test_data = get_batched_data(self.vocab, self.train_batch_size, self.eval_batch_size)

        best_val_loss = float("inf")

        print("Starting model training\n")
        with TemporaryDirectory() as tempdir:
            best_model_params_path = os.path.join(tempdir, "best_model_params.pt")

            for epoch in range(1, self.epochs + 1):
                epoch_start_time = time.time()
                self.train_epoch(train_data, epoch)
                try:
                    val_loss = self.evaluate(val_data)
                    val_ppl = math.exp(val_loss)
                    elapsed = time.time() - epoch_start_time

                    progress_printer(
                        f"end of epoch {epoch:3d}",
                        f"time: {elapsed: 5.2f}s",
                        f"valid loss {val_loss:5.2f}",
                        f"valid ppl {val_ppl:8.2f}"
                    )

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        torch.save(self.model.state_dict(), best_model_params_path)
                except Exception as e:
                    print(f"Exception during evaluate: {e}")
                
                self.scheduler.step()
            
            self.model.load_state_dict(torch.load(best_model_params_path))
        
        test_loss = self.evaluate(test_data)
        test_ppl = math.exp(test_loss)
        progress_printer(
            "End of training",
            f"test loss {test_loss:5.2f}",
            f"test ppl {test_ppl:8.2f}"
        )

        torch.save(self.model.state_dict(), f"{self.log_prefix}trained_model.pt")

        # Leave a debugger on in case we want to examine any local variables
        # after training.
        if self.debug_after_train:
            import pdb; pdb.set_trace()
    
    def train_epoch(self, train_data, epoch: int):
        # Used to track current learning rate of scheduler.
        cur_lr = self.scheduler.get_last_lr()[0]
        
        self.model.train()

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
        batch_shift_offsets = [0] + [CONTEXT_WINDOW_LENGTH * int(random.uniform(0, self.num_batches)) for _ in range(self.train_batch_size - 1)]

        output = None

        if self.log_single_tokens:
            token_num_values = self.vocab(list(self.tokens_to_log))

        for batch, i in enumerate(random.sample(list(training_range), k=self.num_batches)):
            print(".", end="")
            sys.stdout.flush()

            src_mask = generate_square_subsequent_mask(CONTEXT_WINDOW_LENGTH).to(DEVICE)
            if batch >= self.num_batches - 2:
                # We are near the last or penultimate batch.
                # There is no easy way to adjust this when using batch_shift_offsets,
                # so just pass here.
                continue
            else:
                data, targets = get_batch(train_data, i, CONTEXT_WINDOW_LENGTH, batch_shift_offsets)
                if data is None or targets is None:
                    continue

            grads_record = {}
            if not self.log_single_steps:
                # Do not log embeddings in single steps for now due to high
                # space storage utilization.
                if self.include_embedding and batch % self.log_embedding_every_n == 0:
                    grads_record["encoder_weights"] = next(self.model.encoder.parameters()).data.detach().clone()
                    grads_record["decoder_weights"] = next(self.model.decoder.parameters()).data.detach().clone()
                    grads_record["transformer_encoder_weights"] = {
                        name: param.data
                        for name, param in self.model.transformer_encoder.named_parameters()
                    }

            seq_len = data.size(0)
            if seq_len != CONTEXT_WINDOW_LENGTH:
                src_mask = src_mask[:seq_len, :seq_len]
            output = self.model(data, src_mask)

            if self.log_single_steps:
                loss = self.loss_criterion(output.view(-1, self.num_tokens), targets)

                total_grads = {
                    name: torch.zeros_like(param) for name, param in self.model.named_parameters()
                }

                # At this point, `targets` and `loss` have been flattened such that
                # the tokens for the ith data point occur at every `train_batch_size`
                # position. For example, if `train_batch_size` == 20, then 
                # `targets[::20]` contains the target tokens for the first datum.
                # To store gradient attribution for a datum, we store the
                # averaged gradient over the context window (`CONTEXT_WINDOW_LENGTH`).
                # Note `loss.shape[0] == train_batch_size * CHUNK_LENGHTS`.
                for i in range(self.train_batch_size):
                    new_grads = {
                        name: torch.zeros_like(param) for name, param in self.model.named_parameters()
                        if self.include_embedding or name not in ("encoder.weight", "decoder.weight", "decoder.bias")
                    }
                    for j in range(CONTEXT_WINDOW_LENGTH):
                        k = j * self.train_batch_size + i
                        log_immediately = self.log_single_tokens and j > 0 and (int(data[j, i]) in token_num_values or int(targets[k]) in token_num_values)

                        new_grads = {
                            name: torch.zeros_like(param) for name, param in self.model.named_parameters()
                            if self.include_embedding or name not in ("encoder.weight", "decoder.weight", "decoder.bias")
                        }

                        loss[k].backward(retain_graph=True)

                        # In the batch step, this is where we could perform gradient clipping
                        # as per below. Instead, we do this after applying all single-datum grads.
                        #   torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                        for name, param in self.model.named_parameters():
                            if self.include_embedding or name not in ("encoder.weight", "decoder.weight", "decoder.bias"):
                                new_grads[name].add_(param.grad.data * -cur_lr / loss.size(0))
                            total_grads[name].add_(param.grad.data)
                        
                        if log_immediately:
                            log_grads(f"{self.log_prefix}grads_{epoch}/{batch}/{i}/{j}",
                                      new_grads, self.include_embedding and batch % self.log_embedding_every_n == 0)
                        self.model.zero_grad()
                    
                    if not self.log_single_tokens:
                        log_grads(f"{self.log_prefix}grads_{epoch}/{batch}/{i}",
                                  new_grads, self.include_embedding and batch % self.log_embedding_every_n == 0)
                
                # Conserve on some space. The backwards graph keeps references to these
                # so delete them explicitly to trigger GC on the GPU.
                for key in list(new_grads.keys()):
                    del new_grads[key]
                del new_grads

                if DEVICE == "cuda":
                    torch.cuda.empty_cache()
                
                # First we apply the averaged gradient, then we clip norms.
                for name, param in self.model.named_parameters():
                    param.grad = total_grads[name] / loss.size(0)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                
                def sanity_check(loss_size):
                    # Verify that the individual summed grads are equal
                    # to a single pass with a CrossEntropyLoss(reduction="mean").
                    self.model.zero_grad()
                    loss_criterion2 = nn.CrossEntropyLoss(reduction="mean")
                    loss2 = loss_criterion2(output.view(-1, self.num_tokens), targets)
                    loss2.backward(retain_graph=True)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                    params = {name: param for name, param in self.model.named_parameters() } 
                    assert all(
                        ((params[name].grad - total_grads[name] / loss_size).abs() < 1.5e-4).all()
                        for name, _ in self.model.named_parameters()
                        if name != "decoder.weight"
                    )

                if self.apply_grad_sanity_check:
                    sanity_check(loss.size(0))
                
                self.optimizer.step()
                curr_loss = loss.mean().item()
                if not math.isnan(curr_loss):
                    total_loss += curr_loss

                # Clean up some more GPU mem usage explicitly.
                del loss
                for key in list(total_grads.keys()):
                    del total_grads[key]
                del total_grads

            else: # Log gradients for entire batch instead of individual datums. This is easy!
                loss = self.loss_criterion(output.view(-1, self.num_tokens), targets)
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)

                self.optimizer.step()
                curr_loss = loss.item()
                total_loss += curr_loss

            cur_lr = self.scheduler.get_last_lr()[0]

            if not self.log_single_steps:
                grads_record["encoder"] = next(self.model.encoder.parameters()).grad.data * -cur_lr
                grads_record["transformer_encoder"] = {
                    name: param.grad.data * -cur_lr
                    for name, param in self.model.transformer_encoder.named_parameters()
                }
                grads_record["decoder"] = next(self.model.decoder.parameters()).grad.data * -cur_lr

                log_grads(f"{self.log_prefix}grads_{epoch}/{batch}", grads_record, self.include_embedding and batch % self.log_embedding_every_n == 0)

            # Convert training data and target tokens to strings.
            data_batch = [self.vocab.lookup_tokens(data[:, i].tolist()) for i in range(data.shape[1])]
            targets_str = self.vocab.lookup_tokens(targets.tolist())
            train_log.append(TrainLogItem({}, data_batch, targets_str, epoch, batch, curr_loss))

            if batch % self.log_interval == 0 and batch > 0:
                lr = self.scheduler.get_last_lr()[0]
                ms_per_batch = (time.time() - start_time) * 1_000 / log_interval
                cur_loss = total_loss / log_interval
                try:
                    ppl = math.exp(cur_loss)
                except:
                    ppl = -1.0

                progress_printer(
                    f"epoch {epoch:3d}",
                    f"{batch:5d}/{num_batches:5d} batches",
                    f"lr {lr:02.02f}",
                    f"ms/batch {ms_per_batch:5.2f}",
                    f"loss {cur_loss:5.2f}",
                    f"ppl {ppl:8.2f}",
                    wrapper=False
                )

                total_loss = 0
                start_time = time.time()

            with open(f"{self.log_prefix}grads_{epoch}.json", "w") as f:
                json.dump(train_log, f, cls=TrainLogJSONEncoder)

    def evaluate(self, eval_data) -> float:
        self.model.eval()
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
                output = self.model(data, src_mask)
                output_flat = output.view(-1, self.num_tokens)
                loss_criterion = nn.CrossEntropyLoss()
                total_loss += seq_len * loss_criterion(output_flat, targets).item()

        return total_loss / (len(eval_data) - 1)
