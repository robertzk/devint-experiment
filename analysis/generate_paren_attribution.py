import bisect
import functools
import json
import math
import os
import pickle
from collections import namedtuple
from typing import Dict

import numpy as np
import pandas as pd
import torch
import tqdm
import tqdm.notebook
from torch import Tensor

num_epochs = 5
num_batches = 2920
batch_size = 20
training_item = namedtuple("training_item", "id datum")
prefix = "single-step-parens/"

def read_metadata(epoch: int):
    with open(f"{prefix}grads_{epoch}.json", "r") as f:
        return json.load(f)

def add_training_identifier(metadata: dict) -> dict:
    metadata["data_batch"] = [training_item(hash(tuple(d)), d) for d in metadata["data_batch"]]
    return metadata

def collect_top_parameters(tensors: Dict[str, Tensor], n: int = 100) -> pd.DataFrame:
    """
    Given a dict of tensors, such as the gradient updates for each parameter
    in an architecture keyed by the unit (e.g. "encoder"), we identify
    the top `n` parameter values and produce a DataFrame with
    columns "unit", "flat_index" and "value", where "unit" originates
    from the `tensors` keys and "flat_index" indicates the parameter
    index in the corresponding tensors after a `.view(-1)` operation.
    """
    
    # Flatten the tensors so we can efficiently find their top `n` values.
    flattened_tensor = torch.cat(tuple(t.view(-1) for t in tensors.values()))
    
    # Record the shapes so we can attribute indices back to originating tensors.
    tensor_shapes = [(k, tuple(t.shape)) for (k, t) in tensors.items()]
    tensor_keys = [t[0] for t in tensor_shapes]
    tensor_offsets = [functools.reduce(lambda x, y: x * y, t[1]) for t in tensor_shapes]
    tensor_offsets = np.array(tensor_offsets).cumsum().tolist()
    tensor_offsets_shifted = [0] + tensor_offsets[:-1]

    # Get indices of top `n` values
    abs_indices_object = flattened_tensor.abs().topk(n)
    indices_object = flattened_tensor.topk(n)
    indices_metadata = [
        (bisect.bisect_right(tensor_offsets, int(abs_indices_object.indices[i])), 
         abs_indices_object.indices[i], abs_indices_object.values[i])
        for i in range(len(abs_indices_object.values))
    ]
    
    units = [tensor_keys[i] for (i, *_) in indices_metadata]
    flat_indices = [int(full_ix - tensor_offsets_shifted[ind_ix]) for (ind_ix, full_ix, _) in indices_metadata]
    
    return pd.DataFrame({"unit": units, "flat_index": flat_indices,
                         "value": flattened_tensor[abs_indices_object.indices].tolist(),
                         "abs_value": abs_indices_object.values.tolist()})

def read_grads(path: str) -> Tensor:
    with open(path, "rb") as f:
        return pickle.loads(f.read()).to("cuda")


if __name__ == "__main__":
    metadata_per_epoch = {
        epoch: [add_training_identifier(m) for m in read_metadata(epoch)] for epoch in range(1, num_epochs + 1)
    }

    training_ids = sorted(list(set(y.id for m in metadata_per_epoch.values() for x in m for y in x["data_batch"])))

    num_layers = 3

    full_data_attribution = []
    topk_cutoff = 1000
    output_prefix = f"{prefix}analysis/{topk_cutoff}"
    output_columns = []

    count = 0

    os.makedirs(output_prefix, exist_ok=True)

    for epoch in tqdm.tqdm(range(1, num_epochs + 1), total=num_epochs):
        for batch_num in tqdm.tqdm(range(0, num_batches), total=num_batches):

            if count > 0 and count % 100 == 0:
                data_attribution = pd.DataFrame(full_data_attribution, columns=output_columns)
                data_attribution.to_csv(os.path.join(output_prefix, str(count // 100)), index=False)
                full_data_attribution = []

            count += 1

            for i in range(0, batch_size):
                rows = []
                full_prefix = os.path.join(prefix, f"grads_{epoch}", str(batch_num), str(i))
                
                tensors = {}
                for dir, dirs, files in os.walk(full_prefix):
                    for file in files:
                        if "layers." in file:
                            layer = int(file.split("layers.")[1].split(".")[0])
                            # Ignore layers that were constructed from previous runs.
                            if layer >= num_layers:
                                continue
                        curpath = os.path.join(dir, file)
                        grads = read_grads(curpath)
                        tensors[curpath[len(full_prefix)+1:]] = grads

                if len(tensors) == 0:
                    # Guard against an error in empty directories
                    continue

                # Top 1000 most significant parameter updates
                top_params = collect_top_parameters(tensors, n=topk_cutoff).sort_values("value", ascending=False)
                columns = top_params.columns
                top_params["epoch"] = epoch
                top_params["batch_num"] = batch_num
                top_params["datum_ind"] = i
                datum_id = metadata_per_epoch[epoch][batch_num]["data_batch"][i].id
                top_params["datum_id"] = datum_id
                top_params = top_params[["epoch", "batch_num", "datum_ind", "datum_id", *columns]]
                output_columns = ["epoch", "batch_num", "datum_ind", "datum_id", *columns]

                full_data_attribution.extend(list(top_params.itertuples(index=False, name=None)))

    print("Writing metadata")
    with open(os.path.join(output_prefix, "metadata_per_epoch.pkl"), "wb") as f:
        pickle.dump(metadata_per_epoch, f)
