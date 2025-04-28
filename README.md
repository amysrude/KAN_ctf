# KAN

This directory contains the KAN method for the CTF for Science framework. This model incorporates a network with learnable activation functions on edges (weights)
and weight parameters that are replaced by univariate functions parametrized as a spline. KAN2 with multiplication nodes will also be eventually implemented.

## Usage

To run a baseline, use the `run.py` script from the **project root** followed by the path to a configuration file. For example:

```bash
python models/KAN/run.py models/KAN/config/config_Lorenz.yaml
python models/KAN/run.py models/KAN/config/config_KS.yaml

```

**Important**: Always run the script from the project root to ensure relative paths (e.g., to datasets and results directories) work correctly.

This command executes the average baseline on the specified dataset (e.g., `ODE_Lorenz`) for the sub-datasets defined in the config file (e.g., sub-datasets 1 through 6). Results, including predictions, evaluation metrics, and visualizations, are saved in `results/<dataset_name>/KAN_<version>/<batch_id>/`.

### Batch Run Approach

The `run.py` script supports batch runs across multiple sub-datasets as specified in the configuration file. It:
- Parses the `pair_id` from the config to determine which sub-datasets to process.
- Generates a unique `batch_id` for the run (e.g., `batch__20250404_164642`).
- Processes each sub-dataset, saving results and visualizations in a structured directory.
- Aggregates metrics in a `batch_results.yaml` file.

## Configuration Files

Configuration files are located in the `models/KAN/config/` directory and specify the dataset, sub-datasets, and baseline method, along with method-specific parameters.

### Available Configuration Files

- `config_KS.yaml`: Runs the constant baseline on `PDE_KS` for all sub-datasets.
- `config_Lorenz.yaml`: Runs the random baseline on `ODE_Lorenz` for all sub-datasets.

### Configuration Structure

Each configuration file must include:
- **`dataset`** (required):
  - `name`: The dataset name (e.g., `ODE_Lorenz`, `PDE_KS`).
  - `pair_id`: Specifies sub-datasets to run on. Formats:
    - Single integer: `pair_id: 3`
    - List: `pair_id: [1, 2, 3, 4, 5, 6]`
    - Range string: `pair_id: '1-6'`
    - Omitted or `'all'`: Runs on all sub-datasets.
- **`model`**:
  - `name`: `KAN`.
  - `version`: 1


Example (`models/KAN/config/config_Lorenz.yaml`):
```yaml
dataset: 
  name: ODE_Lorenz
  pair_id: [1]


model:
  name: KAN
  version: 1 #Implement KAN2.0 with multiplication nodes later
  steps: 2000 
  pred_window: 1
  lag: 5
  train_ratio: 0.9
  batch: -1
  lr: 0.001
  optimizer: 'Adam' 
  base_fun: 'silu' 
  seed: 42
  width: [2,2,2]
  grid: 3
  update_grid: True
  k: 3
  lamb:  0.0001
  lamb_coef: 0.0001
```

## Examples

- **KAN on Lorenz all datasets**:
  ```bash
  python models/KAN/run.py models/KAN/config/config_Lorenz.yaml
  ```
## Pre-requisites
- Python 3.9.7 or higher


## Requirements

KAN relies on the following packages lists in `requirements.txt`:
- numpy
- pykan
- torch
- scikit-learn
- pandas 
- tqdm


## Notes from Author of KAN for how to hyperparameter tune

## Advice on hyperparameter tuning
Many intuition about MLPs and other networks may not directly transfer to KANs. So how can I tune the hyperparameters effectively? Here is my general advice based on my experience playing with the problems reported in the paper. Since these problems are relatively small-scale and science-oriented, it is likely that my advice is not suitable to your case. But I want to at least share my experience such that users can have better clues where to start and what to expect from tuning hyperparameters.

* Start from a simple setup (small KAN shape, small grid size, small data, no reguralization `lamb=0`). This is very different from MLP literature, where people by default use widths of order `O(10^2)` or higher. For example, if you have a task with 5 inputs and 1 outputs, I would try something as simple as `KAN(width=[5,1,1], grid=3, k=3)`. If it doesn't work, I would gradually first increase width. If that still doesn't work, I would consider increasing depth. You don't need to be this extreme, if you have better understanding about the complexity of your task.

* Once an acceptable performance is achieved, you could then try refining your KAN (more accurate or more interpretable).

* If you care about accuracy, try grid extention technique. An example is [here](https://kindxiaoming.github.io/pykan/Examples/Example_1_function_fitting.html). But watch out for overfitting, see below.

* If you care about interpretability, try sparsifying the network with, e.g., `model.train(lamb=0.01)`. It would also be advisable to try increasing lamb gradually. After training with sparsification, plot it, if you see some neurons that are obvious useless, you may call `pruned_model = model.prune()` to get the pruned model. You can then further train (either to encourage accuracy or encouarge sparsity), or do symbolic regression.

* I also want to emphasize that accuracy and interpretability (and also parameter efficiency) are not necessarily contradictory, e.g., Figure 2.3 in [our paper](https://arxiv.org/pdf/2404.19756). They can be positively correlated in some cases but in other cases may dispaly some tradeoff. So it would be good not to be greedy and aim for one goal at a time. However, if you have a strong reason why you believe pruning (interpretability) can also help accuracy, you may want to plan ahead, such that even if your end goal is accuracy, you want to push interpretability first. 

* Once you get a quite good result, try increasing data size and have a final run, which should give you even better results!

Disclaimer: Try the simplest thing first is the mindset of physicists, which could be personal/biased but I find this mindset quite effective and make things well-controlled for me. Also, The reason why I tend to choose a small dataset at first is to get faster feedback in the debugging stage (my initial implementation is slow, after all!). The hidden assumption is that a small dataset behaves qualitatively similar to a large dataset, which is not necessarily true in general, but usually true in small-scale problems that I have tried. To know if your data is sufficient, see the next paragraph.

Another thing that would be good to keep in mind is that please constantly checking if your model is in underfitting or overfitting regime. If there is a large gap between train/test losses, you probably want to increase data or reduce model (`grid` is more important than `width`, so first try decreasing `grid`, then `width`). This is also the reason why I'd love to start from simple models to make sure that the model is first in underfitting regime and then gradually expands to the "Goldilocks zone".
