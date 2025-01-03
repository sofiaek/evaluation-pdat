import numpy as np


class RobustPolicyCurve:
    def __init__(self, loss_max, x_name=["x"], y_name="y"):
        self.loss_max = loss_max
        self.x_name = x_name
        self.y_name = y_name

    def get_score_one_sided(self, df):
        y = df[self.y_name].to_numpy()
        score = y
        return score

    def get_robust_quantiles_policy(self, quant_list, df, df_beta, gamma, a_i):
        loss = self.get_score_one_sided(df)
        x = df[self.x_name]
        a = df["a"].to_numpy()
        weight_low, weight_high = self.get_weight_bounds(x, a, a_i, gamma)

        x_beta = df_beta[self.x_name]
        __, weight_high_beta = self.get_weight_bounds(x_beta, a_i, a_i, gamma)
        loss = loss[weight_low != 0]
        weight_high = weight_high[weight_low != 0]
        weight_low = weight_low[weight_low != 0]

        sort_idx = np.argsort(loss)
        sort_idx = np.append(sort_idx, 0)  # Dummy, to avoid append later

        loss_n1 = loss[sort_idx]
        weight_low = weight_low[sort_idx]
        weight_high = weight_high[sort_idx]
        weight_high[-1] = 0
        weight_low[-1] = 0

        temp = np.partition(-weight_high, 10)
        print("Gamma: {}, a_i: {}".format(gamma, a_i))
        print(-temp[:10])
        print("99 percentile: {}".format(np.percentile(weight_high, 99)))
        print("99.9 percentile: {}".format(np.percentile(weight_high, 99.9)))

        weight_cum_low = np.cumsum(weight_low)
        weight_cum_high = np.cumsum(weight_high[::-1])[::-1]

        loss_n1[-1] = self.loss_max
        beta_n1, weight_beta_n1 = self.get_weight_beta(weight_high_beta)
        idx_beta = 0
        alpha_n1 = np.ones(len(loss_n1))

        weight_cum_0 = weight_cum_low + np.append(weight_cum_high[1:], 0)
        weight_cum_low_end = weight_cum_low[-1]
        while (
            idx_beta < 10
        ):

            weight_inf = weight_beta_n1[idx_beta]

            weight_cum_n1 = weight_cum_0 + weight_inf
            weight_cum_low[-1] = weight_cum_low_end + weight_inf

            F_hat_n1 = weight_cum_low / weight_cum_n1
            beta = beta_n1[idx_beta]
            alpha_temp = 1 - (1 - beta) * F_hat_n1
            alpha_temp = np.where(beta < alpha_temp, alpha_temp, 1)
            alpha_n1 = np.where(alpha_temp < alpha_n1, alpha_temp, alpha_n1)
            idx_beta += 1
        alpha_n1 = np.where(alpha_n1 == 1, 0, alpha_n1)
        loss_n1 = np.where(alpha_n1 == 0, self.loss_max, loss_n1)

        loss_array = self.loss_max * np.ones(len(quant_list))
        for m, quant_alpha in enumerate(quant_list):
            idx = np.argwhere(alpha_n1 < quant_alpha)
            if len(idx) == 0:
                loss_array[m] = self.loss_max
            else:
                loss_array[m] = loss_n1[idx[0]]
        return loss_n1, alpha_n1, loss_array

    def get_quantiles_rct(self, quant_list, df_all, a_i, is_same_policy=False):
        loss_rct = np.sort(df_all[df_all["a"] == a_i]["y"].to_numpy())
        if is_same_policy:
            loss_rct = np.sort(df_all["y"].to_numpy())
        alpha_rct = np.cumsum(1 / (len(loss_rct)) * np.ones(len(loss_rct)))
        loss_array = self.loss_max * np.ones(len(quant_list))
        alpha_rct = alpha_rct[::-1]
        for m, quant_alpha in enumerate(quant_list):
            idx = np.argwhere(alpha_rct < quant_alpha)
            if len(idx) == 0:
                loss_array[m] = self.loss_max
            else:
                loss_array[m] = loss_rct[idx[0]]
        return loss_rct, alpha_rct, loss_array

    @staticmethod
    def get_weight_beta(weight_all):
        n = len(weight_all)
        weight_n1 = 1 / (n + 1) * np.ones(n + 1)
        alpha_n1 = np.cumsum(weight_n1)
        weight_n1 = np.append(weight_all, 1000)
        weight_n1 = np.sort(weight_n1)
        weight_n1 = weight_n1[::-1]
        return alpha_n1, weight_n1


class ObservationalPolicyCurve(RobustPolicyCurve):
    def __init__(self, loss_max, get_p_ax_policy, get_p_ax, x_name=["x"], y_name="y"):
        super().__init__(loss_max, x_name, y_name)
        self.func_get_p_ax_policy = get_p_ax_policy
        self.func_get_p_ax = get_p_ax

    def get_weight_bounds(self, x, a, a_policy, gamma):
        p_ax = self.func_get_p_ax(x, a_policy)

        weight_low = 1 + 1 / gamma * (1 / p_ax - 1)
        weight_high = 1 + gamma * (1 / p_ax - 1)

        p_ax_policy = self.func_get_p_ax_policy(x, a_policy)
        weight_low = np.where(a == a_policy, p_ax_policy * weight_low, 0)
        weight_high = np.where(a == a_policy, p_ax_policy * weight_high, 0)

        return weight_low, weight_high


class RctPolicyCurve(RobustPolicyCurve):
    def __init__(self, loss_max, get_weight, x_name=["x"], y_name="y"):
        super().__init__(loss_max, x_name, y_name)
        self.func_get_weight = get_weight

    def get_weight_bounds(self, x, a, a_policy, gamma):
        weight = self.func_get_weight(x, a, a_policy)
        weight_low = 1 / gamma * weight
        weight_high = gamma * weight
        return weight_low, weight_high
