from pdf_tokens_type_trainer.ModelConfiguration import ModelConfiguration

config_json = {
    "boosting_type": "gbdt",
    "verbose": -1,
    "learning_rate": 0.1,
    "num_class": 2,
    "context_size": 1,
    "num_boost_round": 400,
    "num_leaves": 191,
    "bagging_fraction": 0.9166599392739231,
    "bagging_freq": 7,
    "feature_fraction": 0.3116707710163228,
    "lambda_l1": 0.0006901861637621734,
    "lambda_l2": 1.1886914989632197e-05,
    "min_data_in_leaf": 50,
    "feature_pre_filter": True,
    "seed": 22,
    "deterministic": True,
}

MODEL_CONFIGURATION = ModelConfiguration(**config_json)

if __name__ == "__main__":
    print(MODEL_CONFIGURATION)
