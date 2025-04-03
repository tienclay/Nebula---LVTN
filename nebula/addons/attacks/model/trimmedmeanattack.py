import logging
from collections import OrderedDict

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple

from nebula.addons.attacks.model.modelattack import ModelAttack


class TrimmedMeanAttack(ModelAttack):
    """
    Implements a Trimmed Mean Attack by manipulating model parameters to bypass
    trimmed mean defense mechanisms.
    
    The attack works by estimating benign parameter distributions and carefully
    positioning malicious updates outside normal ranges while avoiding detection.

    Args:
        engine (object): The engine object that manages the aggregator.
        attack_params (dict): Parameters for the attack, including:
            - std_factor (float): Factor for standard deviation to determine attack range.
            - round_start_attack (int): Round to start the attack.
            - round_stop_attack (int): Round to stop the attack.
    """

    def __init__(self, engine, attack_params):
        """
        Initializes the TrimmedMeanAttack with the specified engine and parameters.

        Args:
            engine (object): The training engine object.
            attack_params (dict): Dictionary of attack parameters.
        """
        super().__init__(engine)
        self.std_factor = float(attack_params.get("std_factor", 3.5))
        self.round_start_attack = int(attack_params.get("round_start_attack", 0))
        self.round_stop_attack = int(attack_params.get("round_stop_attack", 10))
        self.rng = np.random.RandomState(attack_params.get("random_seed"))
        
        # Store model history to estimate benign parameter distributions
        self.model_history = []
        self.history_size = 3  # Number of rounds to keep in history
        
    def estimate_direction(self, model_params: OrderedDict) -> Dict[str, torch.Tensor]:
        """
        Estimate the attack direction for each parameter based on current model values.
        
        Args:
            model_params: Current model parameters

        Returns:
            Dictionary of direction tensors (-1 or 1) for each parameter
        """
        directions = {}
        for name, param in model_params.items():
            # Direction is opposite of parameter sign to maximize attack impact
            directions[name] = -torch.sign(param)
        return directions
    
    def calculate_statistics(self) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Calculate mean and standard deviation of parameters using model history.
        
        Returns:
            Tuple of (means, standard deviations) for each parameter
        """
        if not self.model_history:
            return {}, {}
            
        means = {}
        stds = {}
        
        # Get list of parameter names from first model in history
        param_names = self.model_history[0].keys()
        
        for name in param_names:
            # Stack corresponding parameters from all models in history
            try:
                stacked = torch.stack([model[name] for model in self.model_history])
                means[name] = torch.mean(stacked, dim=0)
                stds[name] = torch.std(stacked, dim=0) + 1e-8  # Prevent division by zero
            except Exception as e:
                logging.warning(f"Error calculating statistics for {name}: {e}")
                # Provide fallback values
                means[name] = torch.zeros_like(self.model_history[0][name])
                stds[name] = torch.ones_like(self.model_history[0][name])
                
        return means, stds
    
    def update_model_history(self, model_params: OrderedDict):
        """
        Update the history of observed models.
        
        Args:
            model_params: Current model parameters
        """
        # Make a deep copy of the model parameters
        model_copy = OrderedDict()
        for name, param in model_params.items():
            model_copy[name] = param.clone().detach()
            
        # Add to history and keep only the most recent entries
        self.model_history.append(model_copy)
        if len(self.model_history) > self.history_size:
            self.model_history.pop(0)
    
    def generate_attack_values(self, model_params: OrderedDict) -> OrderedDict:
        """
        Generate manipulated parameter values for the attack.
        
        Args:
            model_params: Current model parameters
            
        Returns:
            OrderedDict of attack parameter values
        """
        # Update history and calculate statistics
        self.update_model_history(model_params)
        means, stds = self.calculate_statistics()
        
        if not means:  # If we don't have statistics yet
            return model_params
            
        # Estimate directions
        directions = self.estimate_direction(model_params)
        
        # Generate attack values
        attack_model = OrderedDict()
        for name, param in model_params.items():
            if name not in means or name not in stds or name not in directions:
                attack_model[name] = param  # Keep original if missing stats
                continue
                
            direction = directions[name]
            mean = means[name]
            std = stds[name] * self.std_factor
            
            # Initialize attack values tensor
            attack_values = torch.zeros_like(mean)
            
            # Generate mask for parameters to increase (direction = -1)
            mask_increase = (direction == -1)
            # Generate mask for parameters to decrease (direction = 1)
            mask_decrease = (direction == 1)
            
            # When direction is -1, choose random values between μ+3σ and μ+4σ
            if mask_increase.any():
                lower_bound = mean + 3 * std
                upper_bound = mean + 4 * std
                random_factor = torch.rand_like(mean)
                values = lower_bound + random_factor * (upper_bound - lower_bound)
                attack_values[mask_increase] = values[mask_increase]
            
            # When direction is 1, choose random values between μ-4σ and μ-3σ
            if mask_decrease.any():
                lower_bound = mean - 4 * std
                upper_bound = mean - 3 * std
                random_factor = torch.rand_like(mean)
                values = lower_bound + random_factor * (upper_bound - lower_bound)
                attack_values[mask_decrease] = values[mask_decrease]
            
            attack_model[name] = attack_values
            
        return attack_model
    
    def model_attack(self, received_weights):
        """
        Applies the trimmed mean attack by modifying the received model weights.

        Args:
            received_weights (OrderedDict): The aggregated model weights to be modified.

        Returns:
            OrderedDict: The modified model weights after applying the attack.
        """
        logging.info("[TrimmedMeanAttack] Performing trimmed mean attack")
        
        # Generate and return the attack model
        attack_weights = self.generate_attack_values(received_weights)
        return attack_weights