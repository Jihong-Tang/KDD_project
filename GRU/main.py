import pandas as pd
import numpy as np
import os
import tqdm
import random
import argparse
from torchnet import meter
from tqdm import tqdm_notebook
from pathlib import Path
from sklearn.metrics import roc_auc_score
import math
import torch
from torch import nn
import model
import util

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_labels(result, data_id):
    labels = np.zeros(len(data_id), dtype=np.int)
    for index, line in enumerate(result):
        if line[1] > line[0]:
            labels[index] = 1
    data_id['label'] = labels
    return data_id


def train_model(train, batch_size, epochs):
    gru = model.GRUModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(gru.parameters(), lr=0.001)
    gru.train()
    for _ in tqdm.tqdm(range(epochs), desc='Epochs'):
        for i, data in tqdm.tqdm(train.items(), desc='Train'):
            batch = []
            id = list(data.keys())[0]
            label = list(data.values())[0]['status'][0]
            dataset = list(data.values())[0].drop('status', axis=1)
            # print(len(data.values()))
            # dataset = dataset.values.tolist()
            for index in range(96, len(dataset), 96):
                data_input = dataset[(index - 96):index]
                if len(batch) == batch_size or len(dataset) - i <= batch_size:
                    labels = torch.tensor([label] * len(batch)).to(device)
                    batch_tensor = torch.tensor(batch, dtype=torch.float)
                    # print(batch_tensor.type())
                    # os.system('clear')
                    if len(list(batch_tensor.size())) == 1:
                        continue
                    # print('Tensor size: {}'.format(batch_tensor.size()))
                    out_put = gru(batch_tensor.to(device))
                    loss = criterion(torch.squeeze(out_put).cpu(), labels)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                else:
                    batch.append(data_input.values.tolist())
    return gru


def test_model(model, data):
    confusion_matrix = meter.ConfusionMeter(2)
    model.eval()
    res, y_true = [], []
    for item in data.values():
        labels = list(item.values())[0]['status'][0]
        list(item.values())[0].drop('status', axis=1, inplace=True)
        dataset = list(item.values())[0]
        for index in tqdm.tqdm(range(96, len(dataset), 96)):
            data_input = torch.tensor(dataset[(index - 96):index].values, dtype=torch.float)
            label = torch.tensor([labels])
            data_input = torch.unsqueeze(data_input, 0).to(device)
            out_put = torch.squeeze(model(data_input), 0).cpu()
            res.append(max(torch.squeeze(out_put).detach().numpy().tolist()))
            y_true.append(label.numpy().tolist()[0])
            _, out_put = torch.max(out_put, 1)
            confusion_matrix.add(out_put.data, label.data)
    cm_value = confusion_matrix.value()
    model.train()
    roc_auc = roc_auc_score(y_true=y_true, y_score=res, average='weighted')
    accuracy = (cm_value[0][0] + cm_value[1][1]) / cm_value.sum()
    precision = cm_value[0][0] / (cm_value[0][0] + cm_value[0][1])
    recall = cm_value[0][0] / (cm_value[0][0] + cm_value[1][0])
    f1 = 2 * precision * recall / (precision + recall)
    # print('Accuracy: {}, Precision: {}, Recall: {}, F1: {}'.format(accuracy, precision, recall, f1))
    return accuracy, precision, recall, f1, roc_auc


def setup_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default='distribution/')
    parser.add_argument("--label", type=str, default='combined_data_clean2.csv')
    args = parser.parse_args()
    util.load_data(args.data_dir)
    setup_seed(0)
    data_label = util.label_preprocessing(args.label)
    dict_data = util.preprocess_table('dataset/', data_label)
    data_label.sort_values(by=['id'], inplace=True)
    train = util.dict_slice(dict_data, 0, math.floor(0.8 * len(dict_data)))
    test_temp = util.dict_slice(dict_data, math.floor(0.8 * len(dict_data)), len(dict_data))
    test = {}
    for index, key in enumerate(test_temp.keys()):
        test[index] = test_temp[key]
    print('Preprocessing finished')
    gru = train_model(train, batch_size=10, epochs=5)
    accuracy, precision, recall, f1, roc_auc = test_model(gru, test)
    print(gru)
    print(
        'Accuracy: {}, Precision: {}, Recall: {}, F1: {}, Roc_Auc: {}'.format(accuracy, precision, recall, f1, roc_auc))
