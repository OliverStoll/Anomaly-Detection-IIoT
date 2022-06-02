import pandas as pd
import numpy as np
from datetime import datetime


def get_timestamp_percentiles(path, timestamps):
    """
    Function that returns the index of the timestamp in the given file.
    """

    list_percentiles = []
    for timestamp in timestamps:
        # convert timestamp to datetime object if needed
        try:  # normal timestamp
            timestamp = datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S')
        except:
            timestamp = datetime.strptime(timestamp, '%d/%m/%Y  %H:%M:%S')

        df = pd.read_csv(path)
        time_series = df['time_sec']

        # find first index where timestamp is greater than the time_series
        index = time_series.searchsorted(str(timestamp))
        max = len(time_series)
        list_percentiles.append(index / max)
    return list_percentiles


def find_timetuples_from_indexes(indexes, max_index=None):
    """
    Function that checks all indexes if they are adjacent and returns a list of timespan tuples.

    :param indexes: list of single indexes
    """

    timespans_list = []
    old_index = indexes[0]
    current_list = [old_index]
    for i in range(1, len(indexes) - 1):
        if indexes[i] - old_index == 1:
            current_list.append(indexes[i])
        else:
            timespans_list.append(current_list)
            current_list = [indexes[i]]
        old_index = indexes[i]
    timespans_list.append(current_list)

    # get the start and end of the timespans
    tuples_list = [[timespan[0], timespan[-1] + 1] for timespan in timespans_list]

    # divide all tuples by the max index to get the percentage
    if max_index:
        tuples_list = [[tuple[0] / max_index, tuple[1] / max_index] for tuple in tuples_list]

    return tuples_list


def get_tp_fp_fn_tn_from_indexes(pred_indexes, max_index, labels):

    # prepare lists of all possible outcomes for an index
    true_positives = []
    true_negatives = []
    false_positive = []
    false_negative = []

    # get all indexes that are labeled as anomaly
    label_indexes = []
    for index in range(max_index):
        for label in labels:
            index_percentage = index / max_index
            if label[0] <= index_percentage <= label[1]:
                label_indexes.append(index)
                break

    # check the predicted indexes against the labels
    for index in pred_indexes:
        if index in label_indexes:
            true_positives.append(index)
        else:
            false_positive.append(index)

    # check the labels against the predicted indexes
    for index in label_indexes:
        if index not in pred_indexes:
            false_negative.append(index)

    # check all indexes against labels and predicted indexes
    for index in range(max_index):
        if index not in label_indexes and index not in pred_indexes:
            true_negatives.append(index)

    tp = len(true_positives)
    fp = len(false_positive)
    fn = len(false_negative)
    tn = len(true_negatives)

    return tp, fp, fn, tn


def calculate_precision_recall_f1(tp, fp, fn):
    """
    Function that calculates the precision, recall and f1 score.
    """

    precision = tp / (tp + fp) if tp > 0 else 0
    recall = tp / (tp + fn) if tp > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision > 0 or recall > 0 else 0

    return precision, recall, f1


def calculate_auc(trainer, mse):

    all_anomaly_indexes = []
    fps = []
    tps = []
    f1_max = (0, 0)

    mse_sorted = np.sort(mse)

    for threshold in mse_sorted:
        anomaly_indexes = np.where(mse > threshold)[0]
        all_anomaly_indexes.append((anomaly_indexes, threshold))

    for i, (indexes, threshold) in enumerate(all_anomaly_indexes):
        tp, fp, fn, tn = get_tp_fp_fn_tn_from_indexes(pred_indexes=indexes,
                                                      max_index=trainer.data_3d.shape[0],
                                                      labels=trainer.labels)
        precision, recall, f1 = calculate_precision_recall_f1(tp=tp, fp=fp, fn=fn)
        if f1 > f1_max[0]:
            f1_max = (f1, i, threshold)
        fps.append(fp)
        tps.append(tp)
    p = tp + fn
    n = fp + tn

    # divide all the tps and fps by p and n
    tps = np.array(tps) / p
    fps = np.array(fps) / n

    # plot the ROC curve from the coordinates
    auc = np.trapz(y=tps, x=fps)

    return fps, tps, auc, f1_max


def calculate_fft_from_data(data_3d):
    """
    Calculate the FFT of the data. The Data needs to be a 3D array.

    :param data_3d: the 3d array of the data
    :return: the 3d array with the fft of the data
    """

    # calculate the FFT by iterating over the features and the samples
    fft = np.ndarray((data_3d.shape[0], data_3d.shape[1], data_3d.shape[2]))
    for j in range(data_3d.shape[2]):  # iterate over each feature
        for i in range(data_3d.shape[0]):  # iterate over each sample
            fft_origin = data_3d[i, :, j]
            fft_single = abs(np.fft.fft(fft_origin, axis=0))

            # append the single fft computation to the fft array
            fft[i, :, j] = fft_single

    return fft