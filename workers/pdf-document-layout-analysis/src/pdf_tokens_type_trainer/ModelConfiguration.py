from dataclasses import dataclass, asdict

from pdf_token_type_labels.TokenType import TokenType


@dataclass
class ModelConfiguration:
    context_size: int = 4
    num_boost_round: int = 700
    num_leaves: int = 127
    bagging_fraction: float = 0.6810645192499981
    lambda_l1: float = 1.1533558410486358e-08
    lambda_l2: float = 4.91211684620458
    feature_fraction: float = 0.7087268965467017
    bagging_freq: int = 10
    min_data_in_leaf: int = 47
    feature_pre_filter: bool = False
    boosting_type: str = "gbdt"
    objective: str = "multiclass"
    metric: str = "multi_logloss"
    learning_rate: float = 0.1
    seed: int = 22
    num_class: int = len(TokenType)
    verbose: int = -1
    deterministic: bool = True
    resume_training: bool = False
    early_stopping_rounds: int = None

    def dict(self):
        return asdict(self)
