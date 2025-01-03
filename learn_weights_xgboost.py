import logging
import numpy as np
import pandas as pd

from xgboost import XGBClassifier
from sklearn.metrics import log_loss


class LearnProbabilitiesXgboost:
    def __init__(self, args, df, x_name, cat_name):
        self.x_name = x_name
        self.cat_name = cat_name
        self.args = args

        params = {'objective': 'multi:softprob',
                  'num_class': self.args.num_class,
                  'subsample': self.args.subsample,
                  'n_estimators': self.args.n_estimators,
                  'min_child_weight': self.args.min_child_weight,
                  'max_depth': self.args.max_depth,
                  'learning_rate': self.args.learning_rate,
                  'colsample_bytree': self.args.colsample_bytree,
                  'colsample_bylevel': self.args.colsample_bylevel,
                  'gamma': self.args.gamma,
                  'eval_metric': log_loss}

        self.model_sampling = self.train_sampling_model(df, params)

    def get_p_cat_x(self, x, cat):
        proba = self.model_sampling.predict_proba(x)
        p_cat_x = proba[np.arange(0, len(x)), cat]
        return p_cat_x

    def train_sampling_model(self, df, params):
        logging.debug("Training model in LearnProbabilitiesXgboost")
        x = df[self.x_name].to_numpy()
        cat = df[self.cat_name]

        clf = XGBClassifier()
        clf.set_params(**params)
        clf.fit(x, cat)

        return clf


class LearnSampleWeightsXgboost(LearnProbabilitiesXgboost):
    def __init__(self, args, df_rct, df_obs, get_weight_decision_rct, x_name, cat_name):
        super().__init__(args, pd.concat([df_rct, df_obs]), x_name, cat_name)
        self.get_weight_decision_rct = get_weight_decision_rct

    def get_weight_sampling(self, x):
        p_s1_x = self.get_p_cat_x(x, 1)
        p_s0_x = self.get_p_cat_x(x, 0)
        weight = p_s1_x / p_s0_x
        return weight

    def get_weight(self, x, a, a_i):
        return self.get_weight_sampling(x) * self.get_weight_decision_rct(x=x, a=a, a_i=a_i)


def add_model_specific_args(parent_parser):
    parser = parent_parser.add_argument_group("Logistic regression")

    parser.add_argument(
        "--num_class",
        default=5,
        help="Number of classes",
    )

    parser.add_argument(
        "--gamma",
        default=0.2,
        help="Gamma",
    )

    parser.add_argument(
        "--subsample",
        default=0.6,
        help="Subsample",
    )

    parser.add_argument(
        "--n_estimators",
        default=194,
        help="Number of estimators",
    )

    parser.add_argument(
        "--min_child_weight",
        default=4,
        help="Minimum child weight",
    )

    parser.add_argument(
        "--max_depth",
        default=4,
        help="Maximum depth",
    )

    parser.add_argument(
        "--learning_rate",
        default=0.05,
        help="Learning rate",
    )

    parser.add_argument(
        "--colsample_bytree",
        default=0.9,
        help="Colsample by tree",
    )

    parser.add_argument(
        "--colsample_bylevel",
        default=0.4,
        help="Colsample by level",
    )

    return parent_parser
