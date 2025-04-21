import logging
from collections import OrderedDict

import numpy as np
import torch
from nebula.addons.attacks.model.modelattack import ModelAttack


class TrimmedMeanAttack(ModelAttack):
    """
    Partial‑knowledge trimmed‑mean attack (USENIX 2020, §3.4).
    Attacker chỉ biết lịch sử các mô hình đã chính mình gửi, 
    ước lượng hướng thay đổi và biên μ ± k·σ để đẩy tham số
    ra ngoài phạm vi benign.
    """

    def __init__(self, engine, attack_params):
        super().__init__(engine)

        # --- các tham số cấu hình ---
        self.round_start_attack = int(attack_params.get("round_start_attack", 1))
        self.round_stop_attack  = int(attack_params.get("round_stop_attack", 10 ** 9))
        self.default_sigma      = float(attack_params.get("default_sigma", 0.2))
        self.history_size       = int(attack_params.get("history_size", 20))

        self.rng = np.random.RandomState(attack_params.get("random_seed", 42))
        self.model_history: list[OrderedDict[str, torch.Tensor]] = []

        self.current_round = 0   # sẽ tăng ở mỗi lần gọi attack

    # ------------------------------------------------------------------ #
    #                       Công cụ tiện ích nội bộ                      #
    # ------------------------------------------------------------------ #

    def _add_to_history(self, weights: OrderedDict) -> None:
        """Lưu một bản sao models vào history (cắt còn history_size)."""
        copied = OrderedDict((k, v.clone().detach()) for k, v in weights.items())
        self.model_history.append(copied)
        if len(self.model_history) > self.history_size:
            self.model_history.pop(0)

    def _stats(self) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """
        Trả về (means, stds) của mỗi tham số, tính trên history.
        Nếu history < 2 vòng, trả về 2 dict rỗng.
        """
        if len(self.model_history) < 2:
            return {}, {}

        means, stds = {}, {}
        for k in self.model_history[0]:
            stacked = torch.stack([m[k] for m in self.model_history])
            mu      = stacked.mean(0)
            sigma   = stacked.std(0).clamp_min(1e-6)
            sigma   = torch.where(sigma < 1e-6,
                                  torch.full_like(sigma, self.default_sigma),
                                  sigma)
            means[k], stds[k] = mu, sigma
        return means, stds

    def _direction(self) -> dict[str, torch.Tensor]:
        """
        Ước lượng hướng thay đổi s_j = sign(w_t - w_{t-1})
        Trả về dict {param_name: tensor direction (‑1/0/+1)}.
        Nếu history < 2 vòng, trả về dict rỗng.
        """
        if len(self.model_history) < 2:
            return {}
        prev, curr = self.model_history[-2], self.model_history[-1]
        return {k: torch.sign(curr[k] - prev[k]) for k in curr}

    # ------------------------------------------------------------------ #
    #                               Attack                               #
    # ------------------------------------------------------------------ #

    def generate_attack_model(self, local_weights: OrderedDict) -> OrderedDict:
        """Sinh weights đã bị poison dựa trên thuật toán trimmed‑mean."""
        # cập nhật vòng
        self.current_round += 1

        # nếu chưa tới vòng tấn công thì pass‑through
        if not (self.round_start_attack <= self.current_round <= self.round_stop_attack):
            return local_weights

        # thêm mô hình hiện tại vào history
        self._add_to_history(local_weights)

        # cần ít nhất 2 snapshot để tính hướng
        means, stds = self._stats()
        if not means:
            return local_weights

        directions = self._direction()
        attack_model = OrderedDict()

        for name, param in local_weights.items():
            # bảo đảm param nằm trong means/stds/directions
            if name not in means or name not in stds or name not in directions:
                attack_model[name] = param
                continue

            mu, sigma   = means[name], stds[name]
            direction   = directions[name]
            rand_tensor = torch.rand_like(param)

            # dải lấy mẫu
            upper_high  = mu + 4 * sigma   # μ + 4σ
            upper_low   = mu + 3 * sigma   # μ + 3σ
            lower_high  = mu - 3 * sigma   # μ - 3σ
            lower_low   = mu - 4 * sigma   # μ - 4σ

            atk         = torch.empty_like(param)

            mask_up   = direction == -1   # tham số đang giảm ⇒ đẩy lên cao
            mask_down = direction ==  1   # tham số đang tăng ⇒ đẩy xuống thấp

            # Uniform sample trên từng đoạn
            atk[mask_up]   = upper_low[mask_up]   + rand_tensor[mask_up] * (upper_high[mask_up]  - upper_low[mask_up])
            atk[mask_down] = lower_low[mask_down] + rand_tensor[mask_down] * (lower_high[mask_down] - lower_low[mask_down])

            # giữ nguyên nếu direction == 0
            zero_mask = ~(mask_up | mask_down)
            atk[zero_mask] = param[zero_mask]

            attack_model[name] = atk

        return attack_model

    # hàm chính được framework gọi
    def model_attack(self, received_weights: OrderedDict) -> OrderedDict:
        try:
            logging.info("[TrimmedMeanAttack] running partial‑knowledge trimmed‑mean attack")
            return self.generate_attack_model(received_weights)
        except Exception:
            logging.exception("[TrimmedMeanAttack] failed, return original weights")
            return received_weights
