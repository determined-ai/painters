import pedl
import logging
from train_cnn import _train_model

if __name__ == "__main__":
    logging.info("Training Painter by Number in PEDL, experiment {}, trial {}".format(
        pedl.get_experiment_id(), pedl.get_trial_id()))
    experiment_cfg = pedl.get_experiment_config()
    logging.info("Experiment configuration: ", experiment_cfg)

    _train_model()