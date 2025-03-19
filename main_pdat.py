import logging
import numpy as np
import pandas as pd
import argparse

import save_utils
import os
import learn_weights_xgboost

from numpy.random import default_rng
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

from calibration import save_calibration_odds
from robust_policy import ObservationalPolicyCurve


def create_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--test_size",
        type=float,
        default=0.05,
        help="proportion of samples for loss curve (default: %(default)s)",
    )

    parser.add_argument(
        "--file",
        type=str,
        help="name of the file containing the data",
    )

    parser.add_argument(
        "--n_bins",
        type=int,
        default=20,
        help="number of bins for calibration curves (default: %(default)s)",
    )

    parser.add_argument(
        "--save",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="save results and logging from experiment",
    )

    parser.add_argument("--y_max", type=float, default=1.0, help="maximum y value")

    parser.add_argument(
        "--gamma_list",
        type=float,
        default=[1.0, 2.0],
        nargs="+",
        help="list of gamma (default: %(default)s)",
    )

    parser.add_argument(
        "--decisions",
        nargs="+",
        type=int,
        default=[0, 1, 2, 3, 4],
        help="decision of evaluated policy (default: %(default)s)",
    )

    # add model specific args
    parser = save_utils.add_specific_args(parser)
    parser = learn_weights_xgboost.add_model_specific_args(parser)

    return parser


def prepare_original_df(df):
    df = df.drop(["EDATUM_index"], axis=1)

    df = pd.get_dummies(
        df,
        columns=[
            "birth_country",
            "marital_status",
            "highest_education",
        ],
    )

    df["total_income_p50imp"] = df["total_income_p50imp"].fillna(df["total_income_p50imp"].mean())
    df["total_income_p50imp"] = np.log(df["total_income_p50imp"] + 1)

    mapping = {
        "acei": 0,
        "arb": 1,
        "ccb": 2,
        "spc": 3,
        "tzd": 4,
    }
    df["drug_class_5cat_index"] = df["drug_class_5cat_index"].map(mapping)
    df["pdc"] = 1 - df["pdc"]

    df = df.rename({"drug_class_5cat_index": "a", "pdc": "y"}, axis=1)

    x_name_cont = ["age_index", "total_income_p50imp"]

    logging.debug("x_name_cont before scaler: %s", df[x_name_cont].head())
    standard_scaler = preprocessing.StandardScaler()
    df[x_name_cont] = standard_scaler.fit_transform(df[x_name_cont])
    logging.debug("x_name_cont after scaler: %s", df[x_name_cont].head())

    x_names = df.columns.to_list()
    x_names.remove("a")
    x_names.remove("y")
    x_names.remove("sum_pills")
    x_names.remove("LopNr")
    x_names.remove("birth_country_father")
    x_names.remove("birth_country_mother")
    if "time_init" in x_names:
        x_names.remove("time_init")

    logging.info("x_names: %s", x_names)
    logging.debug("Mean y for each a: %s", df.groupby("a")["y"].mean())

    return df, x_names


def main():
    global weight_func
    parser = create_parser()
    args = parser.parse_args()

    out_dir = ""
    if args.save:
        out_dir = save_utils.save_logging_and_setup(args)

    quant_arr = np.linspace(0.0001, 0.9999, 51).tolist()
    rng = default_rng(1057)

    # Read data
    logging.info("Read data")
    df = pd.read_csv(
        os.path.join(os.getcwd(), "data", args.file),
        sep="\t",
        header=0,
    )
    df = df.rename({"# drug_class_5cat_index": "drug_class_5cat_index"}, axis=1)

    df, x_name = prepare_original_df(df)
    logging.info("Prepared")
    logging.info("Number of samples: {}".format(len(df)))
    logging.info("Number of samples with a=0: {}".format((df["a"] == 0).sum()))
    logging.info("Number of samples with a=1: {}".format((df["a"] == 1).sum()))
    logging.info("Number of samples with a=2: {}".format((df["a"] == 2).sum()))
    logging.info("Number of samples with a=3: {}".format((df["a"] == 3).sum()))
    logging.info("Number of samples with a=4: {}".format((df["a"] == 4).sum()))

    df_learn, df_cal = train_test_split(
        df, test_size=0.3, random_state=rng.integers(10000)
    )

    weight_func = learn_weights_xgboost.LearnProbabilitiesXgboost(args, df_learn, x_name, 'a')
    save_calibration_odds(out_dir, df_cal, x_name, 'a', weight_func, 5, args.n_bins)
    n_df_cal = len(df_cal)
    np.save(os.path.join(out_dir, "calibration_scale"), [1, 1, 1, 1, 1])

    df_train, df_beta = train_test_split(
        df, test_size=args.test_size, random_state=rng.integers(10000)
    )
    save_loss(df_train, df_beta, x_name, args, quant_arr, weight_func, out_dir)

    if args.save:
        np.save(os.path.join(out_dir, "quant_arr"), quant_arr)


def treat_all_policy(x, d_new):
    return np.ones(len(x))


def save_loss(df_train, df_beta, x_name, args, quant_arr, weight_func, out_dir):
    robust_alpha = ObservationalPolicyCurve(
        args.y_max, treat_all_policy, weight_func.get_p_cat_x, x_name
    )

    loss_list_orig = []
    alpha_list_orig = []

    loss_orig, alpha_orig, __ = robust_alpha.get_quantiles_rct(
        quant_arr, df_train, 0, is_same_policy=True
    )
    loss_list_orig += [loss_orig]
    alpha_list_orig += [alpha_orig]

    for ii, d_i in enumerate(args.decisions):
        loss_list = []
        alpha_list = []

        for i, gamma_i in enumerate(args.gamma_list):
            logging.info("d_i: {}, gamma_i: {}".format(d_i, gamma_i))
            loss_n1, alpha_n1, __ = robust_alpha.get_robust_quantiles_policy(
                quant_arr, df_train, df_beta, gamma_i, d_i
            )
            loss_list += [loss_n1]
            alpha_list += [alpha_n1]

        loss_orig, alpha_orig, __ = robust_alpha.get_quantiles_rct(
            quant_arr, df_train, d_i
        )
        loss_list_orig += [loss_orig]
        alpha_list_orig += [alpha_orig]

        if args.save:
            np.savez(os.path.join(out_dir, "loss_list_{}".format(d_i)), *loss_list)
            np.savez(os.path.join(out_dir, "alpha_list_{}".format(d_i)), *alpha_list)

    if args.save:
        np.savez(os.path.join(out_dir, "loss_list_orig"), *loss_list_orig)
        np.savez(os.path.join(out_dir, "alpha_list_orig"), *alpha_list_orig)


if __name__ == "__main__":
    main()
