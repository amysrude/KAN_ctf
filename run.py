## Importing all necessary dependencies
import argparse
import yaml
from pathlib import Path
from typing import List, Dict, Any
import datetime
import numpy as np
from ctf4science.data_module import load_dataset, get_prediction_timesteps, parse_pair_ids, get_applicable_plots
from ctf4science.eval_module import evaluate, save_results
from ctf4science.visualization_module import Visualization
from kan_ctf import KANctf

#added by me
import matplotlib.pyplot as plt
import torch
import kan
import torch.nn as nn

def main(config_path: str) -> None:
    """
    Main function to run KAN on specified sub-datasets.

    Loads configuration, parses pair_ids, initializes the model, generates predictions,
    evaluates them, and saves results for each sub-dataset under a batch identifier.

    Args:
        config_path (str): Path to the configuration file.
    """

    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Load dataset name and get list of sub-dataset train/test pair_ids
    dataset_name = config['dataset']['name']
    pair_ids = parse_pair_ids(config['dataset'])


    # Generate a unique batch_id for this run (encode both the model type and the learning rate used)
    model_name = f"{config['model']['name']}_{config['model']['version']}"
    batch_id = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

    # Initialize batch results dictionary for summary
    batch_results = {
        'batch_id': batch_id,
        'model': model_name,
        'dataset': dataset_name,
        'pairs': []
    }

    # Initialize Visualization object
    viz = Visualization()

    # Get applicable visualizations for the dataset
    applicable_plots = get_applicable_plots(dataset_name)

    # Process each sub-dataset
    for pair_id in pair_ids:
        print('pair_id ----------------------------', pair_id)
        # Load sub-dataset
        train_data, init_data = load_dataset(dataset_name, pair_id)
        train_data = np.concatenate(train_data, axis = 1)


        # Load metadata (to provide forecast length)
        prediction_timesteps = get_prediction_timesteps(dataset_name, pair_id)
        prediction_horizon_steps = prediction_timesteps.shape[0]
   
        # Initialize the model with the config and train_data
        model = KANctf(config, train_data, init_data, prediction_horizon_steps, pair_id) #arg: config, train_data, prediction_horizon_steps, pair_id
        
        # Generate predictions
        pred_data = model.predict()

        # Evaluate predictions using default metrics
        results = evaluate(dataset_name, pair_id, pred_data)

        # Save results for this sub-dataset and get the path to the results directory
        results_directory = save_results(dataset_name, model_name, batch_id, pair_id, config, pred_data, results)

        # Append metrics to batch results
        # Convert metric values to plain Python floats for YAML serialization
        batch_results['pairs'].append({
            'pair_id': pair_id,
            'metrics': results
        })

        # Generate and save visualizations that are applicable to this dataset
        for plot_type in applicable_plots:
            fig = viz.plot_from_batch(dataset_name, pair_id, results_directory, plot_type=plot_type)
            viz.save_figure_results(fig, dataset_name, model_name, batch_id, pair_id, plot_type, results_directory)

    # Save aggregated batch results
    with open(results_directory.parent / 'batch_results.yaml', 'w') as f:
        yaml.dump(batch_results, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help="Path to the configuration file.")
    args = parser.parse_args()
    main(args.config)


print('Success!')



