import gc
import logging
import math

import numpy
import torch

from nebula.core.aggregation.aggregator import Aggregator


class Balance(Aggregator):
    """
    Similarity-based aggregator that:
      1) Fetches the local model via self.config.participant["network_args"]["addr"]
      2) Loops through all models, skipping the local one
      3) Collects neighbors that satisfy the similarity threshold
      4) Returns alpha * local + (1-alpha) * average_of_similar
    """

    def __init__(
        self,
        config=None,
        gamma=0.3,  # gamma > 0
        kappa=1.0,  # kappa > 0
        R=2.0,  # R > 0
        alpha=0.5,  # blend factor
        **kwargs,
    ):
        super().__init__(config, **kwargs)
        self.gamma = gamma
        self.kappa = kappa
        self.R = R
        self.alpha = alpha
        self.lambda_func = lambda r: r / float(self.R)

        # If your aggregator tracks the current round, you can store it or fetch it from config
        # e.g. self.current_round = 0

    def run_aggregation(self, models):
        """
        Perform similarity-based filtering:
          - The 'models' arg is a dict: {addr: (param_dict, weight), ...}
          - We get the local model by using self.config.participant["network_args"]["addr"]
          - We compute a threshold = gamma * exp(-kappa*lambda(t)) * ||local_model||
          - We average all neighbors whose distance <= threshold
          - We blend the final average with the local model using 'alpha'
        """
        # 1) Call parent aggregator method (as in your Krum example)
        super().run_aggregation(models)

        # 2) Identify the local model via the node's address
        local_addr = self.config.participant["network_args"]["addr"]
        if local_addr not in models:
            raise ValueError(f"Local address {local_addr} not found in models.")

        local_model, local_weight = models[local_addr]

        # 3) Helper functions for distance and norm
        def model_norm(m):
            # L2 norm of the entire model
            total = 0.0
            for layer in m:
                total += torch.sum(m[layer] ** 2).item()
            return math.sqrt(total)

        def model_distance(m1, m2):
            # Euclidean distance between two model dicts
            dist = 0.0
            for layer in m1:
                l1 = m1[layer]
                l2 = m2[layer]
                dist += numpy.linalg.norm((l1 - l2).cpu().numpy())
            return dist

        # 4) Compute threshold = gamma * exp(-kappa * lambda(t)) * ||local_model||
        #    If you track the round, e.g. self.current_round, use that. Otherwise, default to 0.
        t = self.engine.round
        lam_t = self.lambda_func(t)
        decay_factor = math.exp(-self.kappa * lam_t)
        threshold = self.gamma * decay_factor * model_norm(local_model)

        logging.info(f"Node {local_addr} is running Balance for round {t}.")
        logging.info(f"Threshold: {threshold:.4f}")
        logging.info(f"Local model norm: {model_norm(local_model):.4f}")

        # 5) Collect all "similar" models
        #    We'll do a weighted average (using their second value).
        similar_accum = {layer: torch.zeros_like(param, dtype=torch.float32) for layer, param in local_model.items()}
        total_weight_similar = 0.0

        for addr, (nbr_model, nbr_weight) in models.items():
            # Skip comparing local model to itself
            if addr == local_addr:
                continue

            dist = model_distance(local_model, nbr_model)
            if dist <= threshold:
                # This neighbor is considered "similar"
                for layer in similar_accum:
                    similar_accum[layer] += nbr_model[layer].float() * nbr_weight
                total_weight_similar += nbr_weight

        # 6) If no neighbors pass the threshold, we can either:
        #    (a) return local model unchanged, or
        #    (b) proceed with no similar neighbors
        if total_weight_similar == 0:
            print("No neighbors passed similarity check. Returning local model.")
            return local_model

        logging.info(f"Total similar neighbors: {len(similar_accum)}")

        # 7) Compute the average of all similar neighbors
        for layer in similar_accum:
            similar_accum[layer] /= total_weight_similar
        # log information of similar_accum

        logging.info(f"Similar model norm: {model_norm(similar_accum):.4f}")

        # 8) Combine local model with the averaged neighbors
        #    final_model = alpha * local + (1 - alpha) * average_of_similar
        final_model = {}
        for layer in local_model:
            final_model[layer] = self.alpha * local_model[layer].float() + (1.0 - self.alpha) * similar_accum[layer]

        # 9) Cleanup
        gc.collect()

        # 10) Return the final blended model
        return final_model
