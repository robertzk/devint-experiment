from dataclasses import dataclass, field
from json import JSONEncoder
import os
import pickle
from torch import Tensor
from typing import List

@dataclass
class TrainLogItem:

    grads: dict = field(default_factory=dict)
    data_batch: List[str] = field(default_factory=list)
    targets: List[str] = field(default_factory=list)
    epoch: int
    batch_num: int
    train_loss: int

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


