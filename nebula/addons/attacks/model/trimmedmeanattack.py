import logging
from collections import OrderedDict

import numpy as np
import torch

from nebula.addons.attacks.model.modelattack import ModelAttack


class TrimmedMeanAttack(ModelAttack):
    def __init__(self, engine, attack_params):
        super().__init__(engine)
        self.round_start_attack = int(attack_params.get("round_start_attack", 1))
        self.round_stop_attack = int(attack_params.get("round_stop_attack", 100))
        self.default_sigma = float(attack_params.get("default_sigma", 0.2))
        self.history_size = int(attack_params.get("history_size", 20))
        self.rng = np.random.RandomState(attack_params.get("random_seed", 42))
        self.model_history = []

    def update_model_history(self, model_params: OrderedDict):
        model_copy = OrderedDict()
        for name, param in model_params.items():
            try:
                model_copy[name] = param.clone().detach()
            except Exception as e:
                logging.warning(f"[TrimmedMeanAttack] Failed to clone param {name}: {e}")
        self.model_history.append(model_copy)
        if len(self.model_history) > self.history_size:
            self.model_history.pop(0)

    def calculate_statistics(self) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        if len(self.model_history) < 2:
            return {}, {}

        means, stds = {}, {}
        param_names = self.model_history[0].keys()
        for name in param_names:
            try:
                stacked = torch.stack([model[name] for model in self.model_history if name in model])
                means[name] = stacked.mean(dim=0)
                stds[name] = stacked.std(dim=0) + 1e-8
            except Exception as e:
                logging.warning(f"[TrimmedMeanAttack] Error in stats for {name}: {e}")
                means[name] = torch.zeros_like(self.model_history[0].get(name, torch.tensor(0.0)))
                stds[name] = torch.full_like(self.model_history[0].get(name, torch.tensor(0.0)), self.default_sigma)
        return means, stds

    def estimate_direction(self, model_params: OrderedDict) -> dict[str, torch.Tensor]:
        directions = {}
        for name, param in model_params.items():
            try:
                directions[name] = -torch.sign(param)
            except Exception as e:
                logging.warning(f"[TrimmedMeanAttack] Error estimating direction for {name}: {e}")
        return directions

    def generate_attack_model(self, model_params: OrderedDict) -> OrderedDict:
        self.update_model_history(model_params)
        means, stds = self.calculate_statistics()
        if not means:
            logging.info("[TrimmedMeanAttack] Not enough history to perform attack. Returning original weights.")
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

                # Ranges
                upper_attack = mu + 4 * sigma
                lower_attack = mu + 3 * sigma
                upper_down = mu - 3 * sigma
                lower_down = mu - 4 * sigma

                attack_tensor = torch.empty_like(param)
                mask_up = direction == -1
                mask_down = direction == 1

                # Sample values
                attack_tensor[mask_up] = lower_attack[mask_up] + rand_tensor[mask_up] * (
                    upper_attack[mask_up] - lower_attack[mask_up]
                )
                attack_tensor[mask_down] = lower_down[mask_down] + rand_tensor[mask_down] * (
                    upper_down[mask_down] - lower_down[mask_down]
                )
                attack_tensor[direction == 0] = param[direction == 0]

                attack_model[name] = attack_tensor
            except Exception as e:
                logging.warning(f"[TrimmedMeanAttack] Failed to craft param {name}: {e}")
                attack_model[name] = param
        return attack_model

    def model_attack(self, received_weights: OrderedDict) -> OrderedDict:
        try:
            logging.info("[TrimmedMeanAttack] Performing partial knowledge trimmed mean attack")
            return self.generate_attack_model(received_weights)
        except Exception:
            logging.exception("[TrimmedMeanAttack] Attack failed. Returning original weights.")
            return received_weights
