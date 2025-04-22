import logging
from collections import OrderedDict

import numpy as np
import torch
from nebula.addons.attacks.model.modelattack import ModelAttack


class MedianAttack(ModelAttack):
    """
    Partial‑knowledge median attack dựa trên §3.4 “Attacking Median”.
    Attacker dùng history bản thân để ước lượng wmax,j ≈ μ + k·σ
    và wmin,j ≈ μ - k·σ, rồi sample:
      - nếu s_j = –1: [wmax,j, b·wmax,j]
      - nếu s_j = +1: [b·wmin,j, wmin,j]
    """

    def __init__(self, engine, attack_params):
        super().__init__(engine)
        # --- cấu hình chung ---
        self.round_start_attack = int(attack_params.get("round_start_attack", 1))
        self.round_stop_attack  = int(attack_params.get("round_stop_attack", 10**9))
        self.default_sigma      = float(attack_params.get("default_sigma", 0.2))
        self.history_size       = int(attack_params.get("history_size", 20))
        self.k1           = float(attack_params.get("median_k", 3.0))
        self.k2                  = float(attack_params.get("b_factor", 4.0))  

        self.rng = np.random.RandomState(attack_params.get("random_seed", 42))
        self.model_history: list[OrderedDict[str, torch.Tensor]] = []
        self.current_round = 0

    # ------------------ tiện ích nội bộ ------------------

    def _add_to_history(self, weights: OrderedDict) -> None:
        snapped = OrderedDict((k, v.clone().detach()) for k, v in weights.items())
        self.model_history.append(snapped)
        if len(self.model_history) > self.history_size:
            self.model_history.pop(0)

    def _stats(self) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """
        Tính μj, σj trên lịch sử.
        Trả về ({name: μ}, {name: σ}), hoặc ({}, {}) nếu chưa đủ 2 vòng.
        """
        if len(self.model_history) < 2:
            return {}, {}
        means, stds = {}, {}
        for name in self.model_history[0]:
            stack = torch.stack([m[name] for m in self.model_history])
            mu    = stack.mean(0)
            sigma = stack.std(0).clamp_min(1e-6)
            # phòng trường hợp sigma quá nhỏ
            sigma = torch.where(sigma < 1e-6,
                                torch.full_like(sigma, self.default_sigma),
                                sigma)
            means[name], stds[name] = mu, sigma
        return means, stds

    def _direction(self) -> dict[str, torch.Tensor]:
        """
        Ước lượng s_j = sign(w_t - w_{t-1}) cho mỗi tham số.
        """
        if len(self.model_history) < 2:
            return {}
        prev, curr = self.model_history[-2], self.model_history[-1]
        return {k: torch.sign(curr[k] - prev[k]) for k in curr}

    # ---------------------- attack -----------------------

    def generate_attack_model(self, local_weights: OrderedDict) -> OrderedDict:
        """Sinh cọng weights attack dựa trên median‑attack."""
        self.current_round += 1

        # nếu chưa tới vòng tấn công → trả về nguyên bản
        if not (self.round_start_attack <= self.current_round <= self.round_stop_attack):
            return local_weights

        # thêm vào history
        self._add_to_history(local_weights)

        # tính μ, σ
        means, stds = self._stats()
        if not means:
            return local_weights

        # tính hướng thay đổi
        directions = self._direction()
        crafted = OrderedDict()

        for name, w in local_weights.items():
            # chỉ craft khi đủ μ, σ, s
            if name not in means or name not in stds or name not in directions:
                crafted[name] = w
                continue

            mu    = means[name]
            sigma = stds[name]
            s     = directions[name]
            # estimate wmax_j, wmin_j
            wmax = mu + self.k_median * sigma
            wmin = mu - self.k_median * sigma

            # tạo tensor kết quả
            atk = torch.empty_like(w)
            r   = torch.rand_like(w)

            mask_up   = s == -1  # tham số giảm → push lên trên
            mask_down = s ==  1  # tham số tăng → push xuống dưới

            # sample uniform
            # s==-1: [wmax, b * wmax]
            atk[mask_up] = wmax[mask_up] + r[mask_up] * (self.b * wmax[mask_up] - wmax[mask_up])
            # s==1: [b * wmin, wmin]
            atk[mask_down] = self.b * wmin[mask_down] + r[mask_down] * (wmin[mask_down] - self.b * wmin[mask_down])

            # giữ nguyên với s==0
            zero_mask = ~(mask_up | mask_down)
            atk[zero_mask] = w[zero_mask]

            crafted[name] = atk

        return crafted

    def model_attack(self, received_weights: OrderedDict) -> OrderedDict:
        try:
            logging.info("[MedianAttack] running partial‑knowledge median attack")
            return self.generate_attack_model(received_weights)
        except Exception:
            logging.exception("[MedianAttack] failed, returning original")
            return received_weights
