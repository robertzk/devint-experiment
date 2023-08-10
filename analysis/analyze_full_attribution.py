import functools
import math
import os
import pickle
import random
import re
import tempfile
from collections import Counter, defaultdict, namedtuple
from itertools import chain, groupby
from typing import Any, Callable, Dict, Generator, List, Optional, Set, Tuple

import _io
import pandas as pd
import torch
import tqdm
from torch import Tensor
import math
from typing import List, Optional, Tuple

from devint_experiment.model import TransformerModel, generate_square_subsequent_mask
from devint_experiment.data import get_wikitext_vocab_iterator, get_batched_data, get_data_tensor
from devint_experiment.constants import DEVICE, CONTEXT_WINDOW_LENGTH

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import IterableDataset, dataset
from torchtext.data.utils import get_tokenizer
from torchtext.datasets import WikiText2
from torchtext.vocab import Vocab, build_vocab_from_iterator

log_prefix = "single-step"

config = {
    "num_layers": 3,
    "hidden_dimension": 200,
    "parameter_batch_size": 100,
    "parameter_batches_limit": 50,
    "mlp_neuron_analysis_limit": 50,
    "random_train_data_sample_size": 2000,
    "metadata_per_epoch_file": os.path.join(log_prefix, "metadata_per_epoch.pkl"),
    "full_data_attribution_dir": os.path.join(log_prefix, "analysis", str(1000)),
    "relative_distributions_file": os.path.join(log_prefix, "analysis", "relative_distributions.csv"),
    "full_distributions_file": os.path.join(log_prefix, "analysis", "full_distributions.csv"),
    "ablation_analysis_file": os.path.join(log_prefix, "analysis", "ablation_analysis.pkl"),
    "trained_model_path": os.path.join(log_prefix, "trained_model.pt"),
}

parameter_group = namedtuple("parametergroup", "unit flat_index")

def read_grads(path: str) -> Tensor:
    with open(path, "rb") as f:
        return pickle.loads(f.read()).to("cuda")

def get_2d_tensors():
    tensors_2d = []
    tensors_1d = []
    prefix = f"{log_prefix}/grads_1/0/0"
    for dir, dirs, files in os.walk(prefix):
        for file in files:
            curpath = os.path.join(dir, file)
            grads = read_grads(curpath)
            if "layers." in file:
                layer = int(file.split("layers.")[1].split(".")[0])
                # Ignore layers that were constructed from previous runs.
                if layer >= config["num_layers"]:
                    continue
            if len(grads.shape) == 1:
                tensors_1d.append(curpath[len(prefix)+1:])
            else:
                tensors_2d.append(curpath[len(prefix)+1:])
    
    return tensors_2d

def get_counts_by_parameter(df: pd.DataFrame) -> Counter:
    counts = Counter()
    for _, row in df.iterrows():
        counts[parameter_group(row.unit, row.flat_index)] += 1
    return counts

def full_data_attribution_files():
    for root, dirs, files in os.walk(config["full_data_attribution_dir"]):
        for name in files:
            if re.match("^[0-9]+$", name):
                yield os.path.join(root, name)

def get_counts_by_full_attribution_file(filepath: str, tensors_2d: List[str]=get_2d_tensors()) -> Counter:
    # Filter to only 2d tensors
    return get_counts_by_parameter(
        filtered_data_attribution(filepath, lambda line: any(f"{t}," in line for t in tensors_2d),
                                  pd_args={"usecols": ["unit", "flat_index"]})
    )

def get_counts_by_filtered_data_attribution(filepath: str, filter: Callable[[str], bool], *, pd_args: Dict[Any, Any] = {}) -> pd.DataFrame:
    counter = Counter()
    with open(filepath, "r") as f:
        lines = f.readlines()
        header = True
        for line in lines:
            if header or filter(line):
                if not header:
                    line = line.split(",")
                    counter[parameter_group(line[4], int(line[5]))] += 1
                header = False
    
    return counter

def get_most_common_parameters():
    most_common_parameters_path = os.path.join(config["full_data_attribution_dir"], "most_common_parameters.pkl")
    tensors_2d = get_2d_tensors()

    if not os.path.exists(most_common_parameters_path):
        print("Calculating most common parameters...\n")
        counter = Counter()
        for file in tqdm.tqdm(full_data_attribution_files()):
            counter += get_counts_by_filtered_data_attribution(file, lambda line: any(f"{t}," in line for t in tensors_2d))
        
        print("\nSaving most common parameters\n")
        counter = { (gp.unit, gp.flat_index): value for (gp, value) in counter.items() }
        with open(most_common_parameters_path, "wb") as f:
            pickle.dump(counter, f)
        return counter
    else:
        with open(most_common_parameters_path, "rb") as f:
            return Counter(pickle.load(f))

### 
# Analysis of parameter-datum and neuron-datum attribution
###

training_item = namedtuple("training_item", "id datum")

def training_datum_id_to_datum_map() -> pd.DataFrame:
    metadata_per_epoch = pickle.load(open(config["metadata_per_epoch_file"], "rb"))
    rows = []

    for epoch, steps in metadata_per_epoch.items():
        for batch_num, batch in enumerate(steps):
            for datum_ind, item in enumerate(batch["data_batch"]):
                datum = " ".join(item.datum)
                rows.append((epoch, batch_num, datum_ind, item.id, datum))

    return pd.DataFrame(rows, columns=["epoch", "batch_num", "datum_ind", "datum_id", "datum"])

def extract_parameter_from_full_attribution_file(unit: str, flat_index: int, filepath: str,
                                                 outputfile: _io.TextIOWrapper, *, header: bool = False) -> None:
    filter_str = f",{unit},{flat_index},"
    with open(filepath, "r") as inputfile:
        lines = inputfile.readlines()
        for line in lines:
            if header:
                outputfile.write(f"\n{line}")
                header = False
            elif filter_str in line:
                outputfile.write(f"{line}\n")
        outputfile.flush()

def extract_parameter_from_full_attribution(unit: str, flat_index: int) -> pd.DataFrame:
    with tempfile.NamedTemporaryFile("w") as f:
        header = True
        for file in tqdm.tqdm(full_data_attribution_files()):
            extract_parameter_from_full_attribution_file(unit, flat_index, file, f, header=header)
            header = False

        return pd.read_csv(f.name)

def extract_parameter_from_full_attribution(unit: str, flat_index: int) -> pd.DataFrame:
    with tempfile.NamedTemporaryFile("w") as f:
        header = True
        for file in tqdm.tqdm(full_data_attribution_files()):
            extract_parameter_from_full_attribution_file(unit, flat_index, file, f, header=header)
            header = False

        return pd.read_csv(f.name)

def extract_parameter_from_full_attribution_file_batch(params: List[Tuple[str, int]], filepath: str,
                                                       outputfile: _io.TextIOWrapper, *, header: bool = False,
                                                       header_cols: Optional[List[str]] = None) -> List[str]:
    params = { (str(p[0]), str(p[1])) for p in params }
    if not header and not header_cols:
        raise ValueError("Provide header_cols when not extracting header")
    # filter_strs = [f",{unit},{flat_index}," for unit, flat_index in params]
    if header_cols:
        unit_loc = header_cols.index("unit")
        flat_index_loc = header_cols.index("flat_index")

    with open(filepath, "r") as inputfile:
        lines = inputfile.readlines()
        first = True
        for line in lines:
            row = line.split(",")
            if first and header:
                outputfile.write(f"\n{line}")
                header_cols = line.strip().split(",")
                unit_loc = header_cols.index("unit")
                flat_index_loc = header_cols.index("flat_index")
                first = False
            elif (row[unit_loc], row[flat_index_loc]) in params:
                outputfile.write(f"{line}\n")
        outputfile.flush()
    
    return header_cols

def extract_parameter_from_full_attribution_batch(params: List[Tuple[str, int]]) -> pd.DataFrame:
    params = set(params)
    with tempfile.NamedTemporaryFile("w") as f:
        header = True
        header_cols = None
        for file in tqdm.tqdm(list(full_data_attribution_files())):
            header_cols = extract_parameter_from_full_attribution_file_batch(params, file, f, header=header, header_cols=header_cols)
            header = False

        return pd.read_csv(f.name)

class DistributionShiftCalculator:
    """
    Calculates some distribution shift statistics of an attributed datum batch vs the training dataset.
    """

    ABSOLUTE_COUNT_CUTOFF = 20
    RELATIVE_FREQ_SORT_CUTOFF = 200
    TOP_N_CUTOFF = 5

    def __init__(self):
        self.calculate_training_id_to_datum_map()
        self.calculate_baseline()
    
    def calculate_training_id_to_datum_map(self):
        self.training_id_to_datum_map = training_datum_id_to_datum_map()
    
    def calculate_baseline(self):
        datums = set(self.training_id_to_datum_map.datum.tolist())
        self.baseline_counts = Counter(chain(*[datum.split(" ") for datum in datums]))
        self.baseline_denom = sum(x for x in self.baseline_counts.values())
    
    def random_train_data_sample(self, n: int) -> List[str]:
        datums = list(set(self.training_id_to_datum_map.datum.tolist()))
        return random.sample(datums, n)

    def replace_datum_ids_and_add_datum(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Replace the `datum_id` column with the appropriate values, as these are not guaranteed
        to match the data from `metadata_per_epoch` due to use of the unstable hash().
        """
        df = df.merge(self.training_id_to_datum_map[["epoch", "batch_num", "datum_ind", "datum_id", "datum"]],
            on=["epoch", "batch_num", "datum_ind"])
        df.rename(columns={"datum_id_y": "datum_id"}, inplace=True)
        df.drop(columns=["datum_id_x"], inplace=True)
        return df

    def get_relative_distribution(self, attribution_frame: pd.DataFrame,
                                  frequency_cutoff: int = 1,
                                  compute_mdistance: bool = False) -> pd.DataFrame:
        """
        Produce token distribution for parameter attributed example 
        and relative frequencies to full training data.
        """
        attribution_frame = self.replace_datum_ids_and_add_datum(attribution_frame)
        relative_counts = Counter(" ".join(attribution_frame.datum.tolist()).split(" "))
        relative_total = sum(relative_counts.values())
        relative_distribution = [
            (token, value, value / relative_total, self.baseline_counts[token] / self.baseline_denom,
            relative_freq := (value / relative_total) / (self.baseline_counts[token] / self.baseline_denom),
            (1 + math.log(value)) * (1 + math.log(relative_freq))
            )
            for token, value in relative_counts.items()
            if value >= frequency_cutoff and token != ""
        ]
        relative_distribution = pd.DataFrame(
            relative_distribution,
            columns=["token", "count", "freq", "baseline_freq", "relative_freq", "log_count_relfreq"]
        )
        
        if compute_mdistance:
            distances, outliers = calc_mahalanobis_distances(relative_distribution[["count", "relative_freq"]])
            relative_distribution["mdistance"] = distances
            pass
            
        return relative_distribution

    def get_filtered_relative_distribution(self, attribution_frame: pd.DataFrame,
                                  frequency_cutoff: int = 1,
                                  compute_mdistance: bool = True, *, full_distribution: bool = False) -> pd.DataFrame:
        """
        Produce token distribution for parameter attributed example 
        and relative frequencies to full training data.
        """
        relative_distribution = self.get_relative_distribution(attribution_frame, frequency_cutoff, compute_mdistance)
        if full_distribution:
            relative_distribution_copy = relative_distribution.copy()
        relative_distribution = relative_distribution[relative_distribution["count"] >= type(self).ABSOLUTE_COUNT_CUTOFF].sort_values(
            "relative_freq", ascending=False)
        relative_distribution = relative_distribution.head(type(self).RELATIVE_FREQ_SORT_CUTOFF).nlargest(type(self).TOP_N_CUTOFF, "count")

        if full_distribution:
            return relative_distribution, relative_distribution_copy
        else:
            return relative_distribution


### 
###  Ablation related logic
###

def get_change_region(df: pd.DataFrame, all_diffs: bool = False, n: int = 2) -> pd.DataFrame:
    """
    For rows where df.correct != df.correct_ablated, return the slice
    of `-n` and `n` rows around the appropriate mask.
    """
    if all_diffs:
        diff_mask = (df.pred != df.pred_ablated) 
    else:
        diff_mask = df.correct != df.correct_ablated
    diff_mask = functools.reduce(lambda x, y: x | y, [diff_mask.shift(i) for i in range(1, n + 1)], diff_mask)
    diff_mask = functools.reduce(lambda x, y: x | y, [diff_mask.shift(-i) for i in range(1, n + 1)], diff_mask)
    return df[diff_mask]

def fetch_model_unit(model: nn.Module, unit: str) -> nn.Module:
    unit, *rest = unit.split(".")
    if re.match("^[0-9]+$", unit):
        model = model[int(unit)]
    else:
        model = getattr(model, unit)
    
    if len(rest):
        return fetch_model_unit(model, ".".join(rest))
    else:
        return model

def generate_predict_batch(model: TransformerModel, data: List[str]) -> pd.DataFrame:
    # Ensure all datums have the same number of tokens.
    assert(len(set(len(datum.split(" ")) for datum in data)) == 1)
    datums = get_data_tensor(vocab, (" ".join(data)).split(" "))
    datums = datums.reshape((datums.shape[0] // CONTEXT_WINDOW_LENGTH, CONTEXT_WINDOW_LENGTH)).T

    targets = torch.cat((
        datums[1:, :],
        torch.full([1, datums.shape[1]], vocab(tokenizer("."))[0]).to(DEVICE),
    ), dim=0).reshape(-1)

    src_mask = generate_square_subsequent_mask(CONTEXT_WINDOW_LENGTH).to(DEVICE)

    model.eval()
    with torch.no_grad():
        output = model(datums, src_mask)
        output_flat = output.view(-1, num_tokens)

    num_examples = datums.shape[1]
    rows = []

    for i in range(num_examples):
        datum = vocab.lookup_tokens(datums[:, i].tolist())
        preds = vocab.lookup_tokens(output_flat[i::num_examples].argmax(axis=1).tolist())
        tgts = vocab.lookup_tokens(targets[i::num_examples].tolist())
        correct = [1 if x == y else 0 for x, y in zip(tgts, preds)]
        key = [f"{i}-{j}" for j in range(CONTEXT_WINDOW_LENGTH)]
        rows.append(pd.DataFrame({"key": key, "datum": datum, "pred": preds, "correct": correct}))

    outputs = pd.concat(rows)
    return outputs

class ModelEffectVerifier:

    def __init__(self):
        self.cache_model_state()

    def cache_model_state(self):
        self.base_model = TransformerModel(num_tokens, embedding_dimension, num_heads, hidden_dimension,
                                           num_layers, dropout).to(DEVICE)
        self.model_state = torch.load(config["trained_model_path"])
        self.base_model.load_state_dict(self.model_state)

    def check_replacement_rate(self, n: int, token: str, use_bias: bool = False, *,
                                data: Set[str] = None, # TODO: Make sure we are passing this
                                unit: str = "layers.2.linear1",
                                show_changes: bool = False,
                                change_window: int = 2,
                                relevant_datums: bool = False) -> float:
        """
        Check replacement rate when ablating neuron n in a given unit.
        """
        if data is None:
            raise ValueError("Must provide data to check_replacement_rate")

        model_ablated_bias = TransformerModel(num_tokens, embedding_dimension, num_heads, hidden_dimension, num_layers, dropout).to(DEVICE)
        model_ablated_bias.load_state_dict(self.model_state)

        model_ablated_bias_unit = fetch_model_unit(model_ablated_bias, unit)
        name1, pm1 = [p for p in model_ablated_bias_unit.named_parameters()][0]
        name2, pm2 = [p for p in model_ablated_bias_unit.named_parameters()][1]
            
        if use_bias:
            bias = pm1 if name1 == "bias" else pm2
            bias.requires_grad = False
            bias[n] = -999
        else:
            weights = pm1 if name1 == "weight" else pm2
            weights.requires_grad = False
            weights[:, n] = 0

        data = list(data)[:2000]
        diffs = generate_predict_batch(self.base_model, data).merge(
            generate_predict_batch(model_ablated_bias, data),
            suffixes=("", "_ablated"),
            on=["key"]
        )
            
        if relevant_datums:
            diffs["datum_ix"] = [key.split("-")[0] for key in diffs["key"].tolist()]
            diffs.drop(columns=["key"], inplace=True)
        else:
            diffs.drop(columns=["key"], inplace=True)

        all_changes = get_change_region(diffs, True, n = change_window)

        if relevant_datums:
            relevant_diffs = diffs[diffs.datum_ix.isin(set(all_changes.datum_ix.tolist()))]
            return relevant_diffs, all_changes
        
        changes = all_changes[
        (
            ((all_changes.pred == token) | (all_changes.pred_ablated == token)) &
            (all_changes.pred != all_changes.pred_ablated)
        )
        ]
        num_changes = changes.shape[0]
        
        denom = all_changes[
        (
            ((all_changes.pred == token) | (all_changes.pred_ablated == token))
        )
        ].shape[0]

        ratio = num_changes / denom if denom > 0 else -1
        
        if show_changes:
            return ratio, denom, all_changes
        else:
            return ratio, denom

### End model ablation related

if __name__ == "__main__":
    vocab = get_wikitext_vocab_iterator()
    num_tokens = len(vocab)
    embedding_dimension = 200
    hidden_dimension = 200
    num_layers = 3
    num_heads = 4
    dropout = 0.2
    model = TransformerModel(num_tokens, embedding_dimension, num_heads, hidden_dimension, num_layers, dropout).to(DEVICE)

    train_batch_size = 20
    eval_batch_size = 2

    tokenizer = get_tokenizer("basic_english")

    CONTEXT_WINDOW_LENGTH = 35
    src_mask = generate_square_subsequent_mask(CONTEXT_WINDOW_LENGTH).to(DEVICE)

    train_data, val_data, test_data = get_batched_data(vocab, train_batch_size, eval_batch_size)
    
    counter = get_most_common_parameters()
    distribution_shift_calc = DistributionShiftCalculator()
    verifier = ModelEffectVerifier()

    most_common = counter.most_common()

    params_groups = [
        list(g[1][0] for g in group[1])
        for group in groupby(enumerate(most_common), lambda g: int(g[0] / config["parameter_batch_size"]))
    ][:config["parameter_batches_limit"]]

    relative_distributions = None
    if not os.path.exists(config["relative_distributions_file"]):
        full_relative_distributions = None

        for group in params_groups:
            attribution_frames = extract_parameter_from_full_attribution_batch(group)
            new_relative_distributions = []
            new_full_distributions = []

            for unit, flat_index in group: 
                attribution_frame = attribution_frames[(attribution_frames.unit == unit) & (attribution_frames.flat_index == flat_index)]

                relative_distribution, full_distribution = distribution_shift_calc.get_filtered_relative_distribution(attribution_frame, full_distribution=True)
                relative_distribution["unit"] = unit
                relative_distribution["flat_index"] = flat_index
                full_distribution["unit"] = unit
                full_distribution["flat_index"] = flat_index

                new_relative_distributions.append(relative_distribution)
                new_full_distributions.append(full_distribution)
            
            relative_distributions = pd.concat([relative_distributions, *new_relative_distributions])
            full_relative_distributions = pd.concat([full_relative_distributions, *new_full_distributions])

            # Periodically checkpoint the distributions
            relative_distributions.to_csv(config["relative_distributions_file"])
            full_relative_distributions.to_csv(config["full_distributions_file"])

    # For each neuron, perform zero-ablation exercise.
    # - Determine proportion of prediction flips after ablation for identified *notable tokens*
    # - Determine proportion of prediction flips after ablation on random sample of training data
    # - Compare results

    if not relative_distributions:
        relative_distributions = pd.read_csv(config["relative_distributions_file"])
    
    # Focus on MLP neurons.
    all_mlp_params = [(unit, flat_index) for (unit, flat_index) in chain(*params_groups) if \
                      ".linear" in unit]
    
    neurons = defaultdict(list)
    for unit, flat_index in all_mlp_params:
        neurons[(unit, flat_index % config["hidden_dimension"])].append((unit, flat_index))

    neuron_analysis = {}

    print("Computing attribution data")
    attribution_frame = extract_parameter_from_full_attribution_batch(set(list(chain(*neurons.values()))))
    attribution_frame = distribution_shift_calc.replace_datum_ids_and_add_datum(attribution_frame)

    print(f"Generating predictions for {len(neurons)} MLP neurons")
    for neuron, group in tqdm.tqdm(neurons.items()):
        unit, index = neuron

        relative_distribution = relative_distributions[
            (relative_distributions.unit == unit) &
            (relative_distributions.flat_index.isin([flat_index for unit, flat_index in group]))]
        if relative_distribution.shape[0] == 0:
            continue

        flagged_tokens = set(relative_distribution.token.tolist())

        attributed_data = attribution_frame[
            (attribution_frame.unit == unit) &
            (attribution_frame.flat_index.isin([flat_index for unit, flat_index in group]))
        ].datum.tolist()
        attributed_data = list(random.sample(attributed_data, len(attributed_data)))[:2000]
        token = relative_distribution.token.iloc[0]
        clean_unit = unit.replace(".weight", "").replace(".bias", "")
        
        def compute_token_summaries(data):
            _, _, changes = verifier.check_replacement_rate(index, token, data=data,
                                                            unit=clean_unit, show_changes=True, change_window = CONTEXT_WINDOW_LENGTH)
            switched_preds_summary = {
                "unit": unit,
                "index": index,
                "parameters": group,
                "notable_tokens": list(flagged_tokens),
                "relative_distribution": relative_distribution.copy(),
                "token_summaries": {},
                "attributed_data": data,
            }

            switched_preds_mask = changes.pred != changes.pred_ablated

            switched_preds_summary["total_pred_tokens"] = len(set(changes.pred.tolist()) )
            switched_preds_summary["total_pred_ablated_tokens"] = len(set(changes.pred_ablated.tolist()) )

            switched_tokens = list(set(changes[switched_preds_mask].pred.tolist() + changes[switched_preds_mask].pred_ablated.tolist()))
            for token in tqdm.tqdm(switched_tokens):
                token_summary = {}
                from_mask = changes.pred == token
                to_mask = changes.pred_ablated == token
                any_mask = from_mask | to_mask

                flipped_proportion = sum(switched_preds_mask & any_mask) / sum(any_mask)
                if flipped_proportion == 0:
                    continue

                to_tokens = dict(Counter(changes[switched_preds_mask & from_mask].pred_ablated.tolist()))
                to_tokens_denom = sum(switched_preds_mask & from_mask)
                to_tokens = { token: {"count": value, "ratio": value / to_tokens_denom } for (token, value) in to_tokens.items() }

                from_tokens = dict(Counter(changes[switched_preds_mask & to_mask].pred.tolist()))
                from_tokens_denom = sum(switched_preds_mask & to_mask)
                from_tokens = { token: {"count": value, "ratio": value / from_tokens_denom } for (token, value) in from_tokens.items() }

                token_summary["to_tokens"] = to_tokens
                token_summary["from_tokens"] = from_tokens
                token_summary["flipped_proportion"] = flipped_proportion
                token_summary["count_flipped"] = sum(any_mask)
                token_summary["notable"] = token in flagged_tokens
                if token_summary["notable"]:
                    token_summary["token_stats"] = relative_distribution[relative_distribution.token == token].copy()

                switched_preds_summary["token_summaries"][token] = token_summary
            
            return switched_preds_summary

        neuron_analysis[neuron] = compute_token_summaries(attributed_data)

        # TODO: Add comparative analysis on random data.

        with open(config["ablation_analysis_file"], "wb") as f:
            pickle.dump(neuron_analysis, f)

