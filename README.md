# Development interpretability experiment

This is the codebase accompaniment to the post [Training Process Transparency through Gradient Interpretability](https://www.lesswrong.com/posts/DtkA5jysFZGv7W4qP/training-process-transparency-through-gradient).

## Hardware

We recommend running this on a machine with an Ampere A4000 GPU or better and 2TB of local disk storage. Machines with these specs are available from [paperspace](https://paperspace.com/).

## Installing dependencies

To generate required dependencies first generate poetry lock file, and then install the packages:

```
poetry lock
poetry install
```

## Training a model

Train a model and generate the required gradient data. It will be saved to the directory specified by the `log_prefix` variable in `train_llm.py`.

```
poetry run devint_experiment/train_llm.py
```

Once this succeeds, all gradient data should be available in the `log_prefix`.

## Running analysis

There are several IPy notebooks that were infeasible to bundle with the project, but we make several analysis
scripts available:

```
poetry run analysis/generate_full_data_attribution.py
```

This will generate the top 1000 parameter differences per training step and save them in `f"{log_prefix}/analysis"`.
These are too large to be analyzed in-memory so this enables their analysis through sharded files.

```
poetry run analysis/analyze_full_attribution.py
```

This will generate an ablation analysis file into `f"{log_prefix}/analysis/ablation_analysis.pkl"`. The contents of this file will be
a Python object that has token-by-token zero-weight ablation analysis for training data attribution across the model infrastructure.


