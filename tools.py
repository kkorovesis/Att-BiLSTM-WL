import os
import pickle
import torch
import math
import sys
import csv
import pandas as pd
import numpy

from matplotlib import pyplot as plt
from torch.autograd import Variable
from sklearn.utils import compute_class_weight


#############################################
# Utils
#############################################

def sort_batch(lengths, others):
    """
    Sort batch data and labels by length
    Args:
        lengths (nn.Tensor): tensor containing the lengths for the data

    Returns:

    """
    batch_size = lengths.size(0)

    sorted_lengths, sorted_idx = lengths.sort()
    reverse_idx = torch.linspace(batch_size - 1, 0, batch_size).long()
    sorted_lengths = sorted_lengths[reverse_idx]

    return sorted_lengths, (lst[sorted_idx][reverse_idx] for lst in others)


def progress(loss, epoch, batch, batch_size, dataset_size):
    batches = math.ceil(float(dataset_size) / batch_size)
    count = batch * batch_size
    bar_len = 20
    filled_len = int(round(bar_len * count / float(dataset_size)))

    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    status = 'Epoch {}, Loss: {:.4f}'.format(epoch, loss)
    _progress_str = "\r \r [{}] ...{}".format(bar, status)
    sys.stdout.write(_progress_str)
    sys.stdout.flush()

    if batch == batches:
        print()


def get_class_weights(y):
    """
    Returns the normalized weights for each class
    based on the frequencies of the samples
    :param y: list of true labels (the labels must be hashable)
    :return: dictionary with the weight for each class
    """

    weights = compute_class_weight('balanced', numpy.unique(y), y)

    d = {c: w for c, w in zip(numpy.unique(y), weights)}

    return d


def class_weigths(targets):
    w = get_class_weights(targets)
    labels = get_class_labels(targets)
    return torch.FloatTensor([w[l] for l in sorted(labels)])


def get_class_labels(y):
    """
    Get the class labels
    :param y: list of labels, ex. ['positive', 'negative', 'positive', 'neutral', 'positive', ...]
    :return: sorted unique class labels
    """
    return numpy.unique(y)


def get_labels_to_categories_map(y):
    """
    Get the mapping of class labels to numerical categories
    :param y: list of labels, ex. ['positive', 'negative', 'positive', 'neutral', 'positive', ...]
    :return: dictionary with the mapping
    """
    labels = get_class_labels(y)
    return {l: i for i, l in enumerate(labels)}


def df_to_csv(df,file):
    df.to_csv(file, sep=',', index=True)


def csv_to_df(file):
    df = pd.read_csv(file, sep=',')
    return df


def write_to_csv(file, data):
    with open(file_cache_name(file), 'wb') as csv_file:
        spamwriter = csv.writer(csv_file, delimiter=' ')
        for line in data:
            spamwriter.writerow(line)


def read_from_csv(file, data):
    with open(file_cache_name(file), 'wb') as csv_file:
        spamwriter = csv.reader(csv_file, delimiter=' ')
        for line in data:
            data.add(line)


def file_cache_name(file):
    head, tail = os.path.split(file)
    filename, ext = os.path.splitext(tail)
    return os.path.join(head, filename + ".p")


def write_cache_word_vectors(file, data):
    with open(file_cache_name(file), 'wb') as pickle_file:
        pickle.dump(data, pickle_file)


def load_cache_word_vectors(file):
    with open(file_cache_name(file), 'rb') as f:
        return pickle.load(f)

#############################################
# Ploters
#############################################

def loss_curve(df,EPOCHS):
    plt.figure()
    plt.plot(range(1, EPOCHS + 1), df["Train_Loss"], 'b', label='training loss')
    plt.plot(range(1, EPOCHS + 1), df["Val_Loss"], 'r', label='validation loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.title('Train/Val Loss')
    plt.legend(loc='best')
    plt.show()

def f1_curve(df,EPOCHS):
    plt.figure()
    plt.plot(range(1, EPOCHS + 1), df["Train_F1"], 'b', label='training F1')
    plt.plot(range(1, EPOCHS + 1), df["Val_F1"], 'r', label='validation F1')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.title('Train/Val F1')
    plt.legend(loc='best')
    plt.grid()
    plt.show()

def acc_curve(df,EPOCHS):
    plt.figure()
    plt.plot(range(1, EPOCHS + 1), df["Train_Acc"], 'b', label='training acc')
    plt.plot(range(1, EPOCHS + 1), df["Val_Acc"], 'r', label='validation acc')
    plt.ylabel('acc')
    plt.xlabel('epoch')
    plt.title('Train/Val Acc')
    plt.legend(loc='best')
    plt.grid()
    plt.show()

def mae_curve(df, EPOCHS):
    plt.figure()
    plt.plot(range(1, EPOCHS + 1), df["Macro_MAE"], 'b', label='macro mean ab error')
    plt.plot(range(1, EPOCHS + 1), df["Micro_MAE"], 'r', label='micro mean ab error')
    plt.ylabel('mae')
    plt.xlabel('epoch')
    plt.title('Mean Absolute Error')
    plt.legend(loc='best')
    plt.grid()
    plt.show()

#############################################
# Model Evaluator
#############################################

def eval_dataset(dataloader, model, loss_function):
    # switch to eval mode -> disable regularization layers, such as Dropout
    model.eval()

    y_pred = []
    y = []

    total_loss = 0
    for i_batch, sample_batched in enumerate(dataloader, 1):
        # get the inputs (batch)
        # inputs, topics, labels, lengths, indices = sample_batched
        inputs, topics, labels, lengths, topic_lengths, weights, indices= sample_batched

        # sort batch (for handling inputs of variable length)
        lengths, (inputs, labels, topics) = sort_batch(lengths, (inputs, labels, topics))

        # convert to CUDA Variables
        if torch.cuda.is_available():
            inputs = Variable(inputs.cuda())
            topics = Variable(topics.cuda())
            labels = Variable(labels.cuda())
            lengths = Variable(lengths.cuda())
            topic_lengths = Variable(topic_lengths.cuda())
        else:
            inputs = Variable(inputs)
            topics = Variable(topics)
            labels = Variable(labels)
            topic_lengths = Variable(topic_lengths)

        outputs = model(inputs, topics, lengths, topic_lengths)

        loss = loss_function(outputs, labels)
        total_loss += loss.data[0]

        _, predicted = torch.max(outputs.data, 1)

        y.extend(list(labels.data.cpu().numpy().squeeze()))
        y_pred.extend(list(predicted.squeeze()))

    avg_loss = total_loss / i_batch

    return avg_loss, (y, y_pred)