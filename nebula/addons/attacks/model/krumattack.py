import logging
from collections import OrderedDict
import torch
from nebula.addons.attacks.model.modelattack import ModelAttack

class KrumAttack(ModelAttack):
    """
    Implements a single-malicious-client Krum poisoning attack.
    Crafts one malicious update (index 0) based solely on the
    benign clients' updates from self.engine.get_benign_weights().
    """

    def __init__(self, engine, attack_params):
        super().__init__(engine)
        self.lr = float(attack_params.get("lr", 1.0))
        self.threshold = float(attack_params.get("threshold", 1e-5))
        self.round_start = int(attack_params.get("round_start_attack", 1))
        self.round_stop = int(attack_params.get("round_stop_attack", 10**9))
        self.current_round = 0
        self.last_model = None

    def model_attack(self, received_weights: OrderedDict) -> OrderedDict:
        """
        Overrides ModelAttack.model_attack.
        Crafts a malicious global model (as a single OrderedDict)
        so that Krum would have selected it.
        """
        self.current_round += 1
        if not (self.round_start <= self.current_round <= self.round_stop):
            logging.info(f"[KrumAttack] Round {self.current_round} outside attack window, skipping")
            self.last_model = received_weights
            return received_weights

        try:
            benign_list = self.engine.get_benign_weights()
            if not benign_list:
                logging.warning("[KrumAttack] No benign weights; skipping attack")
                self.last_model = received_weights
                return received_weights

            
            return attacked

        except Exception as e:
            logging.exception(f"[KrumAttack] Error during model_attack: {e}")
            self.last_model = received_weights
            return received_weights
