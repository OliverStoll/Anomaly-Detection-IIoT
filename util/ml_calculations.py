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


def get_timetuples_from_indexes(indexes, max_index=None):
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


def get_confusion_matrix_from_anom_indexes(pred_indexes, max_index, labels):
    """ Function that calculates the confusion matrix from the given indexes """

    # get indexes of the labels
    label_indexes = []
    for label in labels:
        label_index_range = range(int(label[0] * max_index), (label[1] * max_index))
        label_indexes += list(label_index_range)

    # check the predicted indexes against the labels
    tps = list(set(pred_indexes) & set(label_indexes))
    fps = list(set(pred_indexes) - set(label_indexes))
    fns = list(set(label_indexes) - set(pred_indexes))
    tns = list(set(range(max_index)) - set(pred_indexes) - set(label_indexes))

    return len(tps), len(fps), len(fns), len(tns)


def get_precision_recall_f1(tp, fp, fn):
    """ Function that calculates the precision, recall and f1 score """

    precision = tp / (tp + fp) if tp > 0 else 0
    recall = tp / (tp + fn) if tp > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision > 0 or recall > 0 else 0

    return precision, recall, f1


def get_anomaly_indexes(threshold, mse, is_until_failure):
    if is_until_failure:
        first_anomaly_index = np.argmax(mse > threshold)
        if first_anomaly_index == 0:  # no anomaly found
            anomaly_indexes = []
        else:
            anomaly_indexes = list(range(first_anomaly_index, mse.size))
    else:
        anomaly_indexes = np.where(mse > threshold)[0]
    return anomaly_indexes


def calculate_f1(mse, threshold, labels, is_until_failure):
    """ Function that calculates the f1 score for a given threshold. """

    # get the anomaly indexes
    pred_indexes = get_anomaly_indexes(threshold=threshold, mse=mse, is_until_failure=is_until_failure)

    # calculate the f1 score
    tp, fp, fn, tn = get_confusion_matrix_from_anom_indexes(pred_indexes, mse.size, labels)
    _, _, f1 = get_precision_recall_f1(tp, fp, fn)

    return f1


def calculate_auc(trainer, mse, is_until_failure, curve_type="roc"):
    """ Calculates the AUC of the given trainer. Also return two lists of false positives and true positives amounts """

    all_anomaly_indexes = []
    fps = []
    tps = []
    f1s = []
    precisions = []
    recalls = []
    f1_max = (0, 0)

    mse_sorted = np.sort(mse)

    for threshold in mse_sorted:
        anomaly_indexes = get_anomaly_indexes(threshold=threshold, mse=mse, is_until_failure=is_until_failure)
        all_anomaly_indexes.append((anomaly_indexes, threshold))

    for i, (indexes, threshold) in enumerate(all_anomaly_indexes):
        tp, fp, fn, tn = get_confusion_matrix_from_anom_indexes(pred_indexes=indexes,
                                                                max_index=trainer.data_3d.shape[0],
                                                                labels=trainer.labels)
        precision, recall, f1 = get_precision_recall_f1(tp=tp, fp=fp, fn=fn)
        if f1 > f1_max[0]:
            f1_max = (f1, i, threshold)
        f1s.append((f1, i, threshold))
        fps.append(fp)
        tps.append(tp)
        precisions.append(precision)
        recalls.append(recall)
    p = tp + fn
    n = fp + tn

    if curve_type == "roc":
        # divide all the tps and fps by p and n
        tps = np.array(tps) / p if p > 0 else np.array(tps) * 0
        fps = np.array(fps) / n if n > 0 else np.array(fps) * 0
        auc = np.trapz(x=fps, y=tps)

        x_vals = fps
        y_vals = tps

    else:  # TODO
        x_vals = precisions
        y_vals = recalls
        auc = np.trapz(x=precisions, y=recalls)

    return x_vals, y_vals, auc, f1_max


def fft_from_data(data_3d):
    """ Calculate the FFT of the data. The Data needs to be a 3D array. """

    # calculate the FFT by iterating over the features and the samples
    fft = np.ndarray((data_3d.shape[0], data_3d.shape[1], data_3d.shape[2]))
    for j in range(data_3d.shape[2]):  # iterate over each feature
        for i in range(data_3d.shape[0]):  # iterate over each sample
            fft_origin = data_3d[i, :, j]
            fft_single = abs(np.fft.fft(fft_origin, axis=0))

            # append the single fft computation to the fft array
            fft[i, :, j] = fft_single

    return fft