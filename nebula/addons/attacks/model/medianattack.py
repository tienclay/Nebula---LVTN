import logging
from collections import OrderedDict
from typing import Dict, Tuple

import torch
import numpy as np

from nebula.addons.attacks.model.modelattack import ModelAttack


class MedianAttack(ModelAttack):
    def __init__(self, engine, attack_params):
        super().__init__(engine)
        self.round_start_attack = int(attack_params.get("round_start_attack", 0))
        self.round_stop_attack = int(attack_params.get("round_stop_attack", 10))
        self.default_sigma = float(attack_params.get("default_sigma", 0.1))
        self.history_size = int(attack_params.get("history_size", 3))
        self.b = float(attack_params.get("b", 2.0))  # median attack parameter
        self.rng = np.random.RandomState(attack_params.get("random_seed", 42))
        self.model_history = []

    def update_model_history(self, model_params: OrderedDict):
        model_copy = OrderedDict()
        for name, param in model_params.items():
            try:
                model_copy[name] = param.clone().detach()
            except Exception as e:
                logging.warning(f"[MedianAttack] Failed to clone param {name}: {e}")
        self.model_history.append(model_copy)
        if len(self.model_history) > self.history_size:
            self.model_history.pop(0)

    def calculate_statistics(self) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        if len(self.model_history) < self.history_size:
            return {}, {}

        means, stds = {}, {}
        param_names = self.model_history[0].keys()
        for name in param_names:
            try:
                stacked = torch.stack([model[name] for model in self.model_history if name in model])
                means[name] = stacked.mean(dim=0)
                stds[name] = stacked.std(dim=0) + 1e-8
            except Exception as e:
                logging.warning(f"[MedianAttack] Error in stats for {name}: {e}")
                fallback = self.model_history[0].get(name, torch.tensor(0.))
                means[name] = torch.zeros_like(fallback)
                stds[name] = torch.full_like(fallback, self.default_sigma)
        return means, stds

    def estimate_direction(self, model_params: OrderedDict) -> Dict[str, torch.Tensor]:
        directions = {}
        for name, param in model_params.items():
            try:
                directions[name] = -torch.sign(param)
            except Exception as e:
                logging.warning(f"[MedianAttack] Error estimating direction for {name}: {e}")
        return directions

    def generate_attack_model(self, model_params: OrderedDict) -> OrderedDict:
        self.update_model_history(model_params)
        means, stds = self.calculate_statistics()
        if not means:
            logging.info("[MedianAttack] Not enough history to perform attack. Returning original weights.")
            return model_params

        directions = self.estimate_direction(model_params)
        attack_model = OrderedDict()

        for name, param in model_params.items():
            try:
                if name not in means or name not in stds or name not in directions:
                    attack_model[name] = param
                    continue

                mu, sigma = means[name], stds[name]
                sigma[sigma < 1e-6] = self.default_sigma
                direction = directions[name]
                rand_tensor = torch.rand_like(param)

                attack_tensor = torch.empty_like(param)

                mask_up = (direction == -1)
                mask_down = (direction == 1)

                # [wmax, b·wmax] or [wmax, wmax/b]
                upper_up = torch.where(mu > 0, mu * self.b, mu / self.b)
                lower_up = mu

                # [b·wmin, wmin] or [wmin/b, wmin]
                lower_down = torch.where(mu < 0, mu * self.b, mu / self.b)
                upper_down = mu

                attack_tensor[mask_up] = lower_up[mask_up] + rand_tensor[mask_up] * (upper_up[mask_up] - lower_up[mask_up])
                attack_tensor[mask_down] = lower_down[mask_down] + rand_tensor[mask_down] * (upper_down[mask_down] - lower_down[mask_down])
                attack_tensor[direction == 0] = param[direction == 0]

                attack_model[name] = attack_tensor
            except Exception as e:
                logging.warning(f"[MedianAttack] Failed to craft param {name}: {e}")
                attack_model[name] = param
        return attack_model

    def model_attack(self, received_weights: OrderedDict) -> OrderedDict:
        try:
            logging.info("[MedianAttack] Performing partial knowledge median attack")
            return self.generate_attack_model(received_weights)
        except Exception as e:
            logging.exception("[MedianAttack] Attack failed. Returning original weights.")
            return received_weights
