import logging
from collections import OrderedDict
import numpy as np
import torch
from nebula.addons.attacks.model.modelattack import ModelAttack


class KrumAttack(ModelAttack):
    """
    Implements the optimization‐based attack on Krum (§3.2 USENIX 2020).
    Supports both full‐knowledge and partial‐knowledge modes.
    """

    def __init__(self, engine, attack_params):
        super().__init__(engine)

        # attack parameters
        self.full_knowledge   = bool(attack_params.get("full_knowledge", False))
        self.round_start      = int(attack_params.get("round_start", 1))
        self.round_end        = int(attack_params.get("round_end", 10**9))
        self.compromised_c    = int(attack_params["compromised_count"])
        self.total_workers    = int(attack_params["total_workers"])  # m
        self.epsilon          = float(attack_params.get("epsilon", 1e-3))
        self.threshold        = float(attack_params.get("lambda_threshold", 1e-5))
        self.max_binary_iters = int(attack_params.get("max_binary_iters", 50))

        # for partial‐knowledge
        self.history: list[OrderedDict[str, torch.Tensor]] = []

        # track rounds
        self.round = 0

    # ————————————— tiện ích chung —————————————

    @staticmethod
    def _flatten_params(model: OrderedDict) -> torch.Tensor:
        """Convert all tensors into one long vector."""
        return torch.cat([v.view(-1) for v in model.values()])

    @staticmethod
    def _euclidean(a: torch.Tensor, b: torch.Tensor) -> float:
        """Euclidean distance between two flattened parameter vectors."""
        return torch.norm(a - b).item()

    def _krum_select(self, candidates: list[OrderedDict]) -> int:
        """
        Krum selection: for each candidate i, compute squared distances
        to all others, take sum of the smallest (m − c − 2) distances,
        and pick the i with minimal score.
        Returns index of the chosen model.
        """
        m, c = self.total_workers, self.compromised_c
        k = m - c - 2
        flats = [self._flatten_params(w) for w in candidates]
        scores = []
        for i in range(len(flats)):
            dists = [(flats[i] - flats[j]).pow(2).sum().item()
                     for j in range(len(flats)) if j != i]
            dists.sort()
            scores.append(sum(dists[:k]))
        return int(np.argmin(scores))

    # ———————————— full‐knowledge λ solver ————————————

    def _upper_bound_lambda(self, wRe: OrderedDict, benigns: list[OrderedDict]) -> float:
        """
        Compute the upper bound in Theorem 1 (Eq.3):
          λ ≤ 1/(m−2c−1)/√d * ...
            + 1/√d * max_i D(w_i, wRe)
        We approximate it by taking the worst‐case terms.
        """
        m, c = self.total_workers, self.compromised_c
        d = self._flatten_params(wRe).numel()
        # distances from wRe
        DwRe = [self._euclidean(self._flatten_params(b), self._flatten_params(wRe))
                for b in benigns]
        max_DwRe = max(DwRe)

        # for each benign i, find sum of distances to its m−c−2 nearest other benigns
        sums = []
        for i, bi in enumerate(benigns):
            others = benigns[:i] + benigns[i+1:]
            dists = sorted(self._euclidean(self._flatten_params(bi), self._flatten_params(o))
                           for o in others)
            sums.append(sum(dists[: m - c - 2]))
        min_sum = min(sums)

        return (1.0 / ((m - 2*c - 1) * np.sqrt(d))) * min_sum + (1.0 / np.sqrt(d)) * max_DwRe

    def _binary_search_lambda(self, wRe: OrderedDict, s: torch.Tensor, benigns: list[OrderedDict]):
        """
        Binary‐search λ in [0, ub] so that when we craft
          w1 = wRe − λ·s  and replicate it c times,
        Krum will select w1 out of [w1,…,w1, benigns…].
        """
        ub = self._upper_bound_lambda(wRe, benigns)
        lb = 0.0

        for _ in range(self.max_binary_iters):
            mid = (lb + ub) / 2
            # craft c copies of w1
            flat = self._flatten_params(wRe) - mid * s
            # rebuild OrderedDict with same shapes
            w1 = OrderedDict()
            idx = 0
            for k, v in wRe.items():
                numel = v.numel()
                w1[k] = flat[idx:idx+numel].view_as(v).to(v.device)
                idx += numel
            # test Krum
            candidates = [w1] * self.compromised_c + benigns
            chosen = self._krum_select(candidates)
            if chosen == 0:
                return mid, w1
            ub = mid
            if ub - lb < self.threshold:
                break
        # fallback: tiny λ
        return None, wRe

    # —————————— partial‐knowledge helper ——————————

    def _add_history(self, w: OrderedDict):
        snap = OrderedDict((k, v.clone().detach()) for k, v in w.items())
        self.history.append(snap)
        if len(self.history) > self.compromised_c:
            self.history.pop(0)

    def _partial_stats(self):
        # mean of c before‐attack models
        flats = [self._flatten_params(w) for w in self.history]
        avg = torch.stack(flats, dim=0).mean(0)
        return avg

    # —————————————— attack entry point ——————————————

    def generate_attack_model(self, wRe: OrderedDict, benigns: list[OrderedDict]=None) -> OrderedDict:
        """
        If full_knowledge=True, require benigns list and solve Eq.2 via binary‐search.
        Else partial: solve Eq.4 using history and craft w1 = wRe − λ·ŝ.
        """
        self.round += 1
        # outside attack window?
        if not (self.round_start <= self.round <= self.round_end):
            return wRe

        # FULL KNOWLEDGE PATH
        if self.full_knowledge:
            if benigns is None:
                logging.warning("[KrumAttack] full_knowledge requires benigns list")
                return wRe

            # estimate s from true wRe vs. previous global?
            # here we reuse partial‐estimation on history:
            flat_prev = self._flatten_params(self.history[-1]) if self.history else torch.zeros_like(self._flatten_params(wRe))
            s = torch.sign(self._flatten_params(wRe) - flat_prev)

            λ, w1 = self._binary_search_lambda(wRe, s, benigns)
            return w1

        # PARTIAL KNOWLEDGE PATH
        # 1. update self.history with before‐attack local model
        self._add_history(wRe)

        # 2. compute ŝ = sign( mean(history) - wRe )
        avg = self._partial_stats()
        s_hat = torch.sign(avg - self._flatten_params(wRe))

        # 3. choose λ = ||avg - wRe||_∞  (max diff) as a simple bound
        diff = (avg - self._flatten_params(wRe)).abs()
        λ = diff.max().item()

        # 4. craft w1 = wRe − λ·ŝ
        flat_w1 = self._flatten_params(wRe) - λ * s_hat
        w1 = OrderedDict()
        idx = 0
        for k, v in wRe.items():
            numel = v.numel()
            w1[k] = flat_w1[idx:idx+numel].view_as(v).to(v.device)
            idx += numel

        return w1

    def model_attack(self, received_weights: OrderedDict, benigns: list[OrderedDict]=None) -> OrderedDict:
        try:
            logging.info("[KrumAttack] running attack (round %d)", self.round+1)
            return self.generate_attack_model(received_weights, benigns)
        except Exception:
            logging.exception("[KrumAttack] failed—return original")
            return received_weights
