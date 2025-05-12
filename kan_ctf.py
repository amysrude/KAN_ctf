import kan
import numpy as np
import torch
import torch.nn as nn
from kan import KAN
from typing import Optional, Dict

class KANctf:
    """
    Kolmogorov-Arnold Network (KAN) Model.
    Network model with learnable activation functions on edges (weights)
    and weight parameters that are replaced by univariate functions
    parametrized as a spline.

    Attributes:
        pair_id (int): Identifier for the data pair to consider.

        train_data (np.ndarray): Training data.
        n (int): Number of spatial points.
        m (int): Number of time points.
        init_data (np.ndarray): Burn-in data for prediction.
        train_ratio (float): Train to test ratio (0 to 1) 
        
        steps (int): Number of training steps
        lag (int): Number of past timesteps to consider in input.
        batch (int): Batch size, if -1 then full.
        pred_window (int): Number of timesteps to predict as output.
        prediction_horizon_steps (int): Total number of timesteps to predict.

        optimizer (str): Optimizer to use for training
        learning_rate (float): Learning rate for optimizer
        base_func (str): residual function b(x). an activation function phi(x) = sb_scale * b(x) + sp_scale * spline(x)
        lamb (float): Overall penalty strength
        lamb_coef (float): Coefficient magnitude penalty strength

        width (list): Number of neurons in each layer
        grid (int): Number of grid intervals 
        update_grid (bool): If True, update grid regularly before stop_grid_update_step (default -1)
        k (int): The order of piecewise polynomial for spline
        seed (int): Random number generator seed
    """ 
    def __init__(self, config: Dict, train_data: Optional[np.ndarray] = None, init_data: Optional[np.ndarray] = None, prediction_horizon_steps: int = 0, pair_id: Optional[int] = None):
        """
        Initialize the KAN model with the provided configuration.

        Args:
            config (Dict): Configuration dictionary containing method and parameters.
            train_data (Optional[np.ndarray]): Training data for the model.
            init_data (Optional[np.ndarray]): Initialization data for prediction. 
            prediction_horizon_steps (int): Number of timesteps to predict
            pair_id (Optional[int]): Identifier for the data pair to consider
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dtype =  torch.float32

        self.pair_id = pair_id
        self.train_data = train_data
        self.n = train_data.shape[0] 
        self.m = train_data.shape[1] 

        if self.pair_id == 2 or self.pair_id == 4:
            print("Reconstruction task: 'lag' parameter set equal to 0 and 'prediction_window' set to 1")
            self.lag = 0
            self.prediction_window = 1
        else:
            self.lag = config['model']['lag']
            self.prediction_window = config['model']['pred_window']
        
        if self.lag > self.m: 
            raise ValueError(f"Select a 'lag' parameter smaller than the number of training timesteps ({self.m}).")

        
        self.init_data = init_data if init_data is not None else train_data[:,-self.lag:]
        self.prediction_horizon_steps = prediction_horizon_steps

        input_layer = [self.n * max(self.lag,1)]
        output_layer = [self.n * self.prediction_window]
        inner_layer = [
            config['model']['one_d'],
            config['model']['two_d'],
            config['model']['three_d'],
            config['model']['four_d'],
            config['model']['five_d'],
            ][:config['model']['num_neurons']]
        self.width = list(np.concatenate([input_layer, inner_layer, output_layer]))
        
        self.train_ratio = config['model']['train_ratio']
        self.grid = config['model']['grid']
        self.seed = config['model']['seed']
        self.lamb = config['model']['lamb']
        self.learning_rate = config['model']['lr']
        self.optimizer = config['model']['optimizer']
        self.update_grid = config['model']['update_grid']
        self.k = config['model']['k']
        self.steps = config['model']['steps'] 
        self.base_fun = config['model']['base_fun']
        self.lamb_coef = config['model']['lamb_coef']
        self.batch = config['model']['batch']


    def get_data(self):
        """
        Generate the data object for training by extracting input and output data.
        The input data is constructed by taking the past `lag` timesteps for each spatial point,
        and the output data is the corresponding future timesteps with a specified prediction window.
        """
        mean = np.mean(self.train_data, axis=1, keepdims=True)
        std = np.std(self.train_data, axis=1, keepdims=True)
        self.train_data = (self.train_data - mean) / std
        self.init_data = (self.init_data - mean) / std 
        self.normalizer  = [mean, std]

        input = np.zeros((self.m - self.lag - self.prediction_window, self.n, max(self.lag,1)))
        output = np.zeros((self.m - self.lag - self.prediction_window , self.n, self.prediction_window))

        for i in range(self.m - self.lag - self.prediction_window):
            input[i,:,:] = (self.train_data[:,i:i + max(self.lag,1)])          
            output[i,:,:] = (self.train_data[:,i + self.lag : i + self.lag + self.prediction_window])

        input = input.reshape(input.shape[0], -1)  
        output = output.reshape(output.shape[0], -1)  

        train_id = np.random.choice(self.m - self.lag - self.prediction_window, int((self.m - self.lag - self.prediction_window) * self.train_ratio), replace=False)
        test_id = np.array(list(set(range(self.m - self.lag - self.prediction_window)) - set(train_id)))
        dataset = {
            "train_input": torch.from_numpy(input[train_id]).type(self.dtype).to(self.device),
            "train_label": torch.from_numpy(output[train_id]).type(self.dtype).to(self.device),
            "test_input": torch.from_numpy(input[test_id]).type(self.dtype).to(self.device),
            "test_label": torch.from_numpy(output[test_id]).type(self.dtype).to(self.device)
        }
        return dataset


    def train(self):
        """ 
        Train the KAN model with the specified model structure and parameters
        """
        print('width', self.width) 
        data = self.get_data()
        model = KAN(width= self.width, grid=self.grid, k=self.k, seed=self.seed, device= self.device, base_fun = self.base_fun)
        model.fit(
            dataset = data,
            lamb_coef = self.lamb_coef,
            batch = self.batch,
            steps = self.steps, 
            lamb = self.lamb,
            opt = self.optimizer,
            lr = self.learning_rate,
            update_grid = self.update_grid,
            loss_fn = nn.MSELoss(),
            display_metrics=['train_loss', 'test_loss']
        )
        return model
    
    def predict(self):
        """
        Generate predictions based on the KAN model
        """
        model = self.train()
        prediction = np.zeros((self.prediction_horizon_steps, self.n))
        init_data = self.init_data.T
  
        if self.pair_id == 2 or self.pair_id ==4:
            input = torch.tensor(init_data).type(self.dtype)
            print(f'------------ Working on Prediction for Pair ID {self.pair_id}------------')
            with torch.no_grad():
                prediction = model(input).numpy()
        else:
            print(f'------------ Working on Prediction for Pair ID {self.pair_id}------------')    
            for i in range(self.prediction_horizon_steps):                
                input = torch.tensor(init_data.reshape(1,-1)).type(self.dtype)
                with torch.no_grad():
                    pred = model(input).numpy()[0]              
                prediction[i,:] = pred[:self.n]
                init_data = np.vstack([init_data[1:,:], pred[:self.n].reshape(1,self.n)])

        prediction = prediction.T * self.normalizer[1] + self.normalizer[0]
        return prediction
        
