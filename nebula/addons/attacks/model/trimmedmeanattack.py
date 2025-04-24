import logging
from collections import OrderedDict
import numpy as np
import torch
from nebula.addons.attacks.model.modelattack import ModelAttack

class TrimmedMeanAttack(ModelAttack):
    """
    Implements both full-knowledge and partial-knowledge attacks against TrimmedMean aggregation.
    Full-knowledge: uses true benign weights and history of global model.
    """
    def __init__(self, engine, attack_params):
        super().__init__(engine)
        self.round_start = int(attack_params.get("round_start_attack", 1))
        self.round_stop = int(attack_params.get("round_stop_attack", 10**9))
        self.b = float(attack_params.get("b", 16.0))
        self.use_full_knowledge = bool(attack_params.get("use_full_knowledge", True))
        # history for partial-knowledge
        self.history_size = int(attack_params.get("history_size", 20))
        self.default_sigma = float(attack_params.get("default_sigma", 0.2))
        self.k_low = float(attack_params.get("k1", 3))
        self.k_high = float(attack_params.get("k2", 4))
        self.model_history = []  # list of OrderedDict tensors
        self.current_round = 0

    def _add_to_history(self, weights: OrderedDict):
        try:
            snapshot = OrderedDict((k, v.clone().detach()) for k, v in weights.items())
            self.model_history.append(snapshot)
            if len(self.model_history) > self.history_size:
                self.model_history.pop(0)
        except Exception:
            logging.exception("[TrimmedMeanAttack] failed to add weights to history")

    def _stats(self):
        if len(self.model_history) < 2:
            return {}, {}
        means, stds = {}, {}
        try:
            for key in self.model_history[0]:
                stacked = torch.stack([m[key] for m in self.model_history])
                mu = stacked.mean(0)
                sigma = stacked.std(0).clamp_min(1e-6)
                sigma = torch.where(sigma < 1e-6,
                                     torch.full_like(sigma, self.default_sigma),
                                     sigma)
                means[key], stds[key] = mu, sigma
        except Exception:
            logging.exception("[TrimmedMeanAttack] error computing stats from history")
            return {}, {}
        return means, stds

    def _direction(self):
        if len(self.model_history) < 2:
            return {}
        try:
            prev, curr = self.model_history[-2], self.model_history[-1]
            return {k: torch.sign(curr[k] - prev[k]) for k in curr}
        except Exception:
            logging.exception("[TrimmedMeanAttack] error computing direction from history")
            return {}

    def _full_knowledge_attack(self, local_weights: OrderedDict) -> OrderedDict:
        try:
            benign_list = self.engine.get_benign_weights()
            if not benign_list:
                logging.warning("[TrimmedMeanAttack] no benign weights available for full-knowledge attack")
                return local_weights
            history = self.engine.get_global_weights_history()
            if len(history) < 2:
                logging.warning("[TrimmedMeanAttack] not enough global history for full-knowledge attack")
                return local_weights
            # determine true direction per parameter
            logging.warning(f"[TrimmedMeanAttack] benign_list size: {len(benign_list)}")
            logging.warning(f"[TrimmedMeanAttack] history size: {len(history)}")
            prev_g, curr_g = history[-2], history[-1]
            directions = {k: torch.sign(curr_g[k] - prev_g[k]) for k in curr_g}

            attacked = OrderedDict()
            for name, param in local_weights.items():
                dir_tensor = directions.get(name)
                if dir_tensor is None:
                    attacked[name] = param
                    continue
                # stack benign values and ensure shape matches
                try:
                    stacked = torch.stack([bw[name] for bw in benign_list])
                except Exception:
                    logging.exception(f"[TrimmedMeanAttack] error stacking benign for param {name}")
                    attacked[name] = param
                    continue
                wmax = stacked.max(0).values
                wmin = stacked.min(0).values
                logging.info(f"[TrimmedMeanAttack] benign max: {wmax}, min: {wmin}")
                # define safe sampling bounds
                upper_low = wmax
                upper_high = self.b * wmax
                lower_low = wmin / self.b
                lower_high = wmin
                logging.info(f"[TrimmedMeanAttack] upper_low: {upper_low}, upper_high: {upper_high}, lower_low: {lower_low}, lower_high: {lower_high}")
                # generate attack tensor
                rand = torch.rand_like(param)
                atk = torch.empty_like(param)
                mask_up = dir_tensor == -1
                mask_down = dir_tensor == 1
                atk[mask_up] = upper_low[mask_up] + rand[mask_up] * (upper_high[mask_up] - upper_low[mask_up])
                atk[mask_down] = lower_low[mask_down] + rand[mask_down] * (lower_high[mask_down] - lower_low[mask_down])
                atk[~(mask_up | mask_down)] = param[~(mask_up | mask_down)]
                attacked[name] = atk
            return attacked
        except Exception:
            logging.exception("[TrimmedMeanAttack] unexpected error in full-knowledge attack")
            return local_weights

    def _partial_attack(self, local_weights: OrderedDict,
                        means: dict, stds: dict, dirs: dict) -> OrderedDict:
        try:
            attacked = OrderedDict()
            for name, param in local_weights.items():
                mu = means.get(name)
                sigma = stds.get(name)
                dir_tensor = dirs.get(name)
                if mu is None or sigma is None or dir_tensor is None:
                    attacked[name] = param
                    continue
                # bounds for sampling
                ul = mu + self.k_low * sigma
                uh = mu + self.k_high * sigma
                ll = mu - self.k_high * sigma
                lh = mu - self.k_low * sigma
                rand = torch.rand_like(param)
                atk = torch.empty_like(param)
                mask_up = dir_tensor == -1
                mask_down = dir_tensor == 1
                atk[mask_up] = ul[mask_up] + rand[mask_up] * (uh[mask_up] - ul[mask_up])
                atk[mask_down] = ll[mask_down] + rand[mask_down] * (lh[mask_down] - ll[mask_down])
                atk[~(mask_up | mask_down)] = param[~(mask_up | mask_down)]
                attacked[name] = atk
            return attacked
        except Exception:
            logging.exception("[TrimmedMeanAttack] error in partial-knowledge attack")
            return local_weights

    def generate_attack_model(self, local_weights: OrderedDict) -> OrderedDict:
        self.current_round += 1
        if not (self.round_start <= self.current_round <= self.round_stop):
            return local_weights
        if self.use_full_knowledge:
            return self._full_knowledge_attack(local_weights)
        # else partial-knowledge
        self._add_to_history(local_weights)
        means, stds = self._stats()
        if not means:
            return local_weights
        dirs = self._direction()
        return self._partial_attack(local_weights, means, stds, dirs)

    def model_attack(self, received_weights: OrderedDict) -> OrderedDict:
        logging.info(f"[TrimmedMeanAttack] using {'full' if self.use_full_knowledge else 'partial'} knowledge at round {self.current_round}")
        try:
            return self.generate_attack_model(received_weights)
        except Exception:
            logging.exception("[TrimmedMeanAttack] error in model_attack, returning original weights")
            return received_weights
