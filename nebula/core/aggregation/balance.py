import gc
import logging
import math

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
        **kwargs,
    ):
        super().__init__(config, **kwargs)
        self.gamma = float(self.config.participant["aggregator_args"]["balance_params"]["gamma"])
        self.kappa = float(self.config.participant["aggregator_args"]["balance_params"]["kappa"])
        self.alpha = float(self.config.participant["aggregator_args"]["balance_params"]["alpha"])
        self.R = int(self.config.participant["scenario_args"]["rounds"])
        self.lambda_func = lambda r: r / float(self.R)

    def run_aggregation(self, models):  # noqa: C901
        """
        Perform similarity-based filtering:
          - The 'models' arg is a dict: {addr: (param_dict, weight), ...}
          - We get the local model by using self.config.participant["network_args"]["addr"]
          - We compute a threshold = gamma * exp(-kappa*lambda(t)) * ||local_model||
          - We average all neighbors whose distance <= threshold
          - We blend the final average with the local model using 'alpha'

        Returns:
            tuple: (aggregated_params, agg_metrics) where agg_metrics is a dict of
                   aggregation-level metrics for TensorBoard logging.
        """
        # 1) Call parent aggregator method
        super().run_aggregation(models)
        # Loging balance params
        logging.debug("Running Balance Aggregator")
        logging.debug(f"Balance Params: gamma={self.gamma}, kappa={self.kappa}, alpha={self.alpha}, R={self.R}")

        # 2) Identify the local model via the node's address
        local_addr = self.config.participant["network_args"]["addr"]
        if local_addr not in models:
            raise ValueError(f"Local address {local_addr} not found in models.")  # noqa: TRY003

        local_model, local_weight = models[local_addr]

        # 3) Helper functions for distance and norm
        def model_norm(m):
            # L2 norm of the entire model
            total = 0.0
            for layer in m:
                total += torch.sum(m[layer] ** 2).item()
            return math.sqrt(total)

        def model_distance(m1, m2):
            dist_sq = 0.0
            for layer in m1:
                diff = (m1[layer] - m2[layer]).float()
                dist_sq += torch.sum(diff * diff).item()
            return math.sqrt(dist_sq)

        # 4) Compute threshold = gamma * exp(-kappa * lambda(t)) * ||local_model||
        t = self.engine.round
        lam_t = self.lambda_func(t)
        decay_factor = math.exp(-self.kappa * lam_t)
        local_norm = model_norm(local_model)
        threshold = self.gamma * decay_factor * local_norm

        logging.debug(f"Node {local_addr} is running Balance for round {t}.")
        logging.debug(f"Threshold: {threshold:.4f}")
        logging.debug(f"Local model norm: {local_norm:.4f}")

        # 5) Collect all "similar" models and track distances
        similar_accum = {layer: torch.zeros_like(param, dtype=torch.float32) for layer, param in local_model.items()}
        total_weight_similar = 0.0
        similar_models_count = 0
        total_neighbors = 0
        neighbor_distances = []

        for addr, (nbr_model, nbr_weight) in models.items():
            # Skip comparing local model to itself
            if addr == local_addr:
                continue

            total_neighbors += 1
            dist = model_distance(local_model, nbr_model)
            neighbor_distances.append(dist)
            logging.debug(f"Neighbor {addr} norm: {model_norm(nbr_model):.4f}")
            logging.debug(f"Distance to {addr}: {dist:.4f}")
            if dist <= threshold:
                # This neighbor is considered "similar"
                for layer in similar_accum:
                    similar_accum[layer] += nbr_model[layer].float() * nbr_weight
                total_weight_similar += nbr_weight
                similar_models_count += 1

        # Build aggregation metrics
        avg_distance = sum(neighbor_distances) / len(neighbor_distances) if neighbor_distances else 0.0
        agg_metrics = {
            "Aggregation/Threshold": threshold,
            "Aggregation/SimilarNeighborsCount": similar_models_count,
            "Aggregation/TotalNeighbors": total_neighbors,
            "Aggregation/AvgNeighborDistance": avg_distance,
            "Aggregation/LocalModelNorm": local_norm,
            "Aggregation/NeighborAcceptRate": similar_models_count / max(total_neighbors, 1),
        }

        # 6) If no neighbors pass the threshold, return local model unchanged
        if similar_models_count == 0:
            logging.debug("No neighbors passed similarity check. Returning local model.")
            return local_model, agg_metrics

        # Log number of similar models
        logging.info(f"Number of similar neighbors: {similar_models_count}")

        # 7) Compute the average of all similar neighbors
        for layer in similar_accum:
            similar_accum[layer] /= total_weight_similar

        # 8) Combine local model with the averaged neighbors
        #    final_model = alpha * local + (1 - alpha) * average_of_similar
        final_model = {}
        for layer in local_model:
            final_model[layer] = self.alpha * local_model[layer].float() + (1.0 - self.alpha) * similar_accum[layer]

        # Log the final model norm
        logging.info(f"Final model norm: {model_norm(final_model):.4f}")

        # 9) Cleanup
        gc.collect()

        # 10) Return the final blended model with metrics
        return final_model, agg_metrics
