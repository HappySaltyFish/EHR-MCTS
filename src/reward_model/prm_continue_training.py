"""Compatibility entry point for PRM weight-initialized training.

Only model/projector/tokenizer weights are restored; optimizer/scheduler start fresh.
"""

from reward_model.prm_training import main


if __name__ == "__main__":
    main(default_filtering="yes", continuation_defaults=True)
