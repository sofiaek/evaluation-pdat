import os

import numpy as np
import pandas as pd


def save_calibration_odds(out_dir: str, df: pd.DataFrame, x_name: str, cat_name: str, weight_func, n_cat, n_bins=10):
    x = df[x_name]

    odds_true_list = []
    odds_pred_list = []
    for cat in range(n_cat):
        cat_prob = weight_func.get_p_cat_x(x, cat)
        odds = cat_prob / (1 - cat_prob)
        sort_id = np.argsort(odds)

        odds = odds[sort_id]
        cat_np = df[cat_name].to_numpy()
        cat_np = np.where(cat_np == cat, 1, 0)
        cat_np = cat_np[sort_id]

        # Calculate the mean value of each bin
        bins = len(df) // n_bins

        bin_means_p_true = np.array([np.mean(cat_np[i:i + bins]) for i in range(0, bins * n_bins, bins)])
        odds_true = bin_means_p_true / (1 - bin_means_p_true)

        odds_predicted = np.array([np.mean(odds[i:i + bins]) for i in range(0, bins * n_bins, bins)])

        odds_true_list += [odds_true]
        odds_pred_list += [odds_predicted]

    np.save(os.path.join(out_dir, 'calibration_odds_true'), odds_true_list)
    np.save(os.path.join(out_dir, 'calibration_odds_pred'), odds_pred_list)
