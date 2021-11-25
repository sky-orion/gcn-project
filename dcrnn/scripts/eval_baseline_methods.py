import argparse
import numpy as np
import pandas as pd

from statsmodels.tsa.vector_ar.var_model import VAR

from libs import utils
from libs.utils import StandardScaler
from libs.metrics import masked_rmse_np, masked_mape_np, masked_mae_np


def var_predict(df, n_forwards=(1, 5, 10), n_lags=10, test_ratio=0.2):
    """
    Multivariate time series forecasting using Vector Auto-Regressive Model.
    :param df: pandas.DataFrame, index: time, columns: sensor id, content: data.
    :param n_forwards: a tuple of horizons.
    :param n_lags: the order of the VAR model.
    :param test_ratio:
    :return: [list of prediction in different horizon], dt_test
    """
    n_sample, n_output = df.shape  # 1440*126
    n_test = int(round(n_sample * test_ratio))
    n_train = n_sample - n_test
    df_train, df_test = df[:n_train], df[n_train:]

    scaler = StandardScaler(mean=df_train.values.mean(), std=df_train.values.std())
    data = scaler.transform(df_train.values)
    var_model = VAR(data)
    var_result = var_model.fit(n_lags)
    max_n_forwards = np.max(n_forwards)
    # Do forecasting.
    result = np.zeros(shape=(len(n_forwards), n_test, n_output))
    start = n_train - n_lags - max_n_forwards + 1
    for input_ind in range(start, n_sample - n_lags):
        prediction = var_result.forecast(scaler.transform(df.values[input_ind: input_ind + n_lags]), max_n_forwards)
        for i, n_forward in enumerate(n_forwards):
            result_ind = input_ind - n_train + n_lags + n_forward - 1
            if 0 <= result_ind < n_test:
                result[i, result_ind, :] = prediction[n_forward - 1, :]

    df_predicts = []
    for i, n_forward in enumerate(n_forwards):
        df_predict = pd.DataFrame(scaler.inverse_transform(result[i]), index=df_test.index, columns=df_test.columns)
        df_predicts.append(df_predict.values)
    return np.squeeze(np.array(df_predicts)), np.array(df_test)


def eval_var(traffic_reading_df, n_forwards, n_lags):
    y_predicts, y_test = var_predict(traffic_reading_df, n_forwards=n_forwards, n_lags=n_lags, test_ratio=0.2)
    logger.info('VAR (lag=%d)' % n_lags)
    logger.info('Model\tHorizon\tRMSE\tMAPE\tMAE')
    # for i, horizon in  enumerate(n_forwards):
    #     rmse = masked_rmse_np(preds=y_predicts[i].values, labels=y_test.values, null_val=0)
    #     mape = masked_mape_np(preds=y_predicts[i].values, labels=y_test.values, null_val=0)
    #     mae = masked_mae_np(preds=y_predicts[i].values, labels=y_test.values, null_val=0)
    #     line = 'VAR\t%d\t%.2f\t%.2f\t%.2f' % (horizon, rmse, mape * 100, mae)
    #     logger.info(line)
    print(y_predicts.shape, y_test.shape)
    get_acc(pred=y_predicts, true=y_test, logger=logger)
    # output = {'prediction': y_predicts[0], 'truth': y_test}
    # np.savez_compressed(f'./var_predicted_results_{n_lags}.npz', **output)


def main(args):
    traffic_reading_df = pd.read_csv(args.traffic_reading_filename, header=None)
    n_forwards = [args.n_lags]
    eval_var(traffic_reading_df, n_forwards, args.n_lags)


def get_acc(pred, true, logger):
    acc_all, acc_all_H, acc_all_N, acc_all_L = [], [], [], []
    for step in range(pred.shape[0]):
        acc_step, acc_step_H, acc_step_N, acc_step_L = [], [], [], []
        # for idx in range(pred.shape[-1]):
        pred_node, true_node = pred[step, :].reshape(-1), true[step, :].reshape(-1)  # 变成向量 n*1
        pred_cls, true_cls = np.zeros(shape=pred_node.size), np.zeros(shape=pred_node.size)
        high_idx, normal_idx, low_idx = [], [], []
        for i in range(pred_node.size):
            if pred_node[i] < 2 / 3:
                pred_cls[i] = 0
                low_idx.append(i)
            elif pred_node[i] >= 4 / 3:
                pred_cls[i] = 2
                high_idx.append(i)
            else:
                pred_cls[i] = 1
                normal_idx.append(i)
        for i in range(true_node.size):
            if true_node[i] < 2 / 3:
                true_cls[i] = 0
            elif true_node[i] >= 4 / 3:
                true_cls[i] = 2
            else:
                true_cls[i] = 1
        # print(len(true_cls), len(true_cls[high_idx]), len(true_cls[normal_idx]), len(true_cls[low_idx]))
        acc = sum(pred_cls == true_cls) / len(true_cls)
        accH = sum(true_cls[high_idx] == pred_cls[high_idx]) / len(true_cls[high_idx]) if high_idx else 0  # 高中 低
        accN = sum(true_cls[normal_idx] == pred_cls[normal_idx]) / len(true_cls[normal_idx]) if normal_idx else 0
        accL = sum(true_cls[low_idx] == pred_cls[low_idx]) / len(true_cls[low_idx]) if low_idx else 0

        acc_step.append(acc)
        acc_step_H.append(accH)
        acc_step_N.append(accN)
        acc_step_L.append(accL)
        acc_all.append(sum(acc_step) / len(acc_step))
        acc_step_H, acc_step_N, acc_step_L = \
            list(filter(lambda x: x != 0, acc_step_H)), list(filter(lambda x: x != 0, acc_step_N)), list(
                filter(lambda x: x != 0, acc_step_L))
        acc_step_H_, acc_step_N_, acc_step_L_ = \
            sum(acc_step_H) / len(acc_step_H) if acc_step_H else 0, \
            sum(acc_step_N) / len(acc_step_N) if acc_step_N else 0, \
            sum(acc_step_L) / len(acc_step_L) if acc_step_L else 0
        acc_all_H.append(acc_step_H_)
        acc_all_N.append(acc_step_N_)
        acc_all_L.append(acc_step_L_)

        logger.info(
            f'Horizon {step + 1:02d} {sum(acc_step) / len(acc_step):.4f}|{acc_step_H_:.4f}|{acc_step_N_:.4f}|{acc_step_L_:.4f}')

    acc_all_H, acc_all_N, acc_all_L = list(filter(lambda x: x != 0, acc_all_H)), list(
        filter(lambda x: x != 0, acc_all_N)), list(filter(lambda x: x != 0, acc_all_L))
    acc_all_H_, acc_all_N_, acc_all_L_ = \
        sum(acc_all_H) / len(acc_all_H) if acc_all_H else 0, \
        sum(acc_all_N) / len(acc_all_N) if acc_all_N else 0, \
        sum(acc_all_L) / len(acc_all_L) if acc_all_L else 0
    logger.info(f'=Average=  {sum(acc_all) / len(acc_all):.4f} {acc_all_H_:.4f} {acc_all_N_:.4f} {acc_all_L_:.4f}')


if __name__ == '__main__':
    logger = utils.get_logger('./data/model', 'Baseline')
    parser = argparse.ArgumentParser()
    parser.add_argument('--traffic_reading_filename', default="/devdata/zhaohaoran/airspace_data.csv", type=str,
                        help='Path to the traffic Dataframe.')
    parser.add_argument('--n_lags', default=1, type=int)
    args = parser.parse_args()
    main(args)
