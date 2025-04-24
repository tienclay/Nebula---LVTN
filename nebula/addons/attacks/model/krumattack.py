import logging
from collections import OrderedDict
import numpy as np
import torch
from nebula.addons.attacks.model.modelattack import ModelAttack

class KrumAttack(ModelAttack):
    def __init__(self, engine, attack_params):
        super().__init__(engine)
        self.round_start = int(attack_params.get("round_start_attack", 1))
        self.round_stop = int(attack_params.get("round_stop_attack", 10**9))
        self.epsilon = float(attack_params.get("epsilon", 1e-3))
        self.threshold = float(attack_params.get("lambda_threshold", 1e-5))
        self.max_iters = int(attack_params.get("max_binary_search", 20))
        self.use_full_knowledge = bool(attack_params.get("use_full_knowledge", True))
        self.c = int(attack_params.get("num_compromised", 1))

    def _get_global_direction(self):
        try:
            history = self.engine.get_global_weights_history()
            if len(history) < 2:
                raise ValueError("Not enough global history to compute direction")
            prev, curr = history[-2], history[-1]
            return {k: torch.sign(curr[k] - prev[k]) for k in curr}
        except Exception:
            logging.exception("[KrumAttack] Failed to get global direction")
            return {}

    def _compute_upper_bound_lambda(self, benign_list, wRe, directions):
        try:
            d = sum(v.numel() for v in wRe.values())
            wRe_tensor = wRe[next(iter(wRe))].view(-1)
            dist_wre = [
                torch.dist(bw[next(iter(wRe))].view(-1), wRe_tensor)
                for bw in benign_list
            ]
            max_dist = max(dist_wre)
            return max_dist / np.sqrt(d)
        except Exception:
            logging.exception("[KrumAttack] Failed to compute lambda upper bound")
            return 1.0

    def _binary_search_lambda(self, wRe, s, benign_list):
        try:
            lo, hi = 0.0, self._compute_upper_bound_lambda(benign_list, wRe, s)
            best_lambda = 0.0
            for _ in range(self.max_iters):
                mid = (lo + hi) / 2.0
                w1 = OrderedDict({k: wRe[k] - mid * s[k] for k in wRe})
                models = OrderedDict()
                models['adv0'] = (w1, 1)
                for i in range(1, self.c):
                    perturbed = OrderedDict({
                        k: w1[k] + (torch.randn_like(w1[k]) * self.epsilon)
                        for k in w1
                    })
                    models[f'adv{i}'] = (perturbed, 1)
                for idx, bw in enumerate(benign_list):
                    models[f'ben{idx}'] = (bw, 1)

                try:
                    agg = self.engine.aggregator.run_aggregation(models)
                except Exception:
                    logging.exception("[KrumAttack] Aggregator failed during binary search")
                    break

                if all(torch.allclose(agg[k], w1[k], atol=1e-6) for k in w1):
                    best_lambda = mid
                    lo = mid
                else:
                    hi = mid

                if hi - lo < self.threshold:
                    break
            return best_lambda
        except Exception:
            logging.exception("[KrumAttack] Binary search lambda failed")
            return 0.0

    def _full_knowledge_attack(self, local_weights):
        try:
            benign_list = self.engine.get_benign_weights()
            if not benign_list:
                logging.warning("[KrumAttack] No benign weights for full-knowledge attack")
                return local_weights
            history = self.engine.get_global_weights_history()
            if len(history) < 2:
                logging.warning("[KrumAttack] Insufficient global history")
                return local_weights
            wRe = history[-1]
            s = self._get_global_direction()
            if not s:
                return local_weights
            lam = self._binary_search_lambda(wRe, s, benign_list)
            w1 = OrderedDict({k: wRe[k] - lam * s[k] for k in wRe})
            attacked = OrderedDict()
            attacked['adv0'] = w1
            for i in range(1, self.c):
                attacked[f'adv{i}'] = OrderedDict({
                    k: w1[k] + (torch.randn_like(w1[k]) * self.epsilon)
                    for k in w1
                })
            return attacked['adv0']
        except Exception:
            logging.exception("[KrumAttack] Full-knowledge attack failed")
            return local_weights

    def _partial_knowledge_attack(self, local_weights):
        try:
            if not local_weights:
                return local_weights[0]
            mean_local = OrderedDict({
                k: sum(w[k] for w in local_weights) / len(local_weights)
                for k in local_weights[0]
            })
            history = self.engine.get_global_weights_history()
            if not history:
                return local_weights[0]
            wRe = history[-1]
            s = {k: torch.sign(mean_local[k] - wRe[k]) for k in wRe}
            lam = self._binary_search_lambda(wRe, s, local_weights)
            w1 = OrderedDict({k: wRe[k] - lam * s[k] for k in wRe})
            return w1
        except Exception:
            logging.exception("[KrumAttack] Partial-knowledge attack failed")
            return local_weights[0]

    def generate_attack_model(self, local_weights):
        try:
            self.current_round += 1
            if not (self.round_start <= self.current_round <= self.round_stop):
                return local_weights[0]
            if self.use_full_knowledge:
                return self._full_knowledge_attack(local_weights)
            return self._partial_knowledge_attack(local_weights)
        except Exception:
            logging.exception("[KrumAttack] generate_attack_model failed")
            return local_weights[0]

    def model_attack(self, received_weights):
        try:
            return self.generate_attack_model(received_weights)
        except Exception:
            logging.exception("[KrumAttack] model_attack failed")
            return received_weights[0]
