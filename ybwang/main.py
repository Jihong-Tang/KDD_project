import pandas as pd
import numpy as np
import os
import tqdm
from torchnet import meter
from tqdm import tqdm_notebook
from pathlib import Path
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


def train_model(train, batch_size):
    gru = model.GRUModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(gru.parameters(), lr=0.001)
    gru.train()
    for i, data in tqdm.tqdm(train.items(), desc='Train'):
        batch = []
        id = list(data.keys())[0]
        label = list(data.values())[0]['status'][0]
        list(data.values())[0].drop('status', axis=1, inplace=True)
        print(len(data.values()))
        dataset = list(data.values())[0]
        for index in range(96, len(dataset), 96):
            data_input = dataset[(index - 96):index]
            if len(batch) == batch_size or len(dataset) - i <= batch_size:
                labels = [label] * len(batch)
                batch_tensor = torch.tensor(batch, dtype=torch.float)
                print(batch_tensor.type())
                print('Tensor size: {}'.format(batch_tensor.size()))
                out_put = gru(batch_tensor)
                loss = criterion(out_put, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            else:
                batch.append(data_input.values.tolist())
    return gru


def val(model, data, labels):
    model.eval()
    for index in tqdm.tqdm(range(96, len(data), 96)):
        data_input = data[(index - 96):index]
        out_put, data_id = model(data_input)
        result = get_labels(out_put, data_id)


def test_model(model, data, labels):
    confusion_matrix = meter.ConfusionMeter(2)
    model.eval()
    for index in tqdm.tqdm(range(96, len(data), 96)):
        data_input = data[(index - 96):index]
        out_put, _ = model(data_input)
        confusion_matrix.add(out_put.detach(), labels[(index / 96) - 1])
    cm_value = confusion_matrix.value()
    model.train()
    accuracy = (cm_value[0][0] + cm_value[1][1]) / cm_value.sum()
    precision = cm_value[0][0] / (cm_value[0][0] + cm_value[0][1])
    recall = cm_value[0][0] / (cm_value[0][0] + cm_value[1][0])
    f1 = 2 * precision * recall / (precision + recall)
    return accuracy, precision, recall, f1


if __name__ == '__main__':
    # util.load_data('distribution\\')
    data_label = util.label_preprocessing('combined_data_clean2.csv')
    dict_data = util.preprocess_table('dataset/', data_label)
    data_label.sort_values(by=['id'], inplace=True)
    train = util.dict_slice(dict_data, 0, math.floor(0.7 * len(dict_data)))
    test_temp = util.dict_slice(dict_data, math.floor(0.7 * len(dict_data)), len(dict_data))
    test = {}
    for index, key in enumerate(test_temp.keys()):
        test[index] = test_temp[key]
    print('Preprocessing finished')
    train_model(train, batch_size=10)

    '''
    different_key = [x for x in data_label['id'].drop_duplicates() if x not in data_label['id']]
    print('Different key: {}'.format(different_key))
    for key in different_key:
        dataset = dataset[~dataset['id'].isin([key])]
    labels = np.zeros(len(data_label), dtype=np.int)
    for i, value in enumerate(data_label['covid_status']):
        if 'positive' in value:
            labels[i] = 1
    
    print('Preprocessing finished')
    print(len(dataset))
    print(len(data_label))
    dataset_train_index = 96 * math.floor(0.7 * len(data_label))
    dataset_val_index = 96 * (math.floor(0.7 * len(data_label)) + math.floor(0.1 * len(data_label)))
    gru = train(dataset[:dataset_train_index], labels[:(dataset_train_index / 96)])
    # val(gru, dataset[dataset_train_index:dataset_val_index],
    # labels[(dataset_train_index / 96):(dataset_val_index / 96)])
    accuracy, precision, recall, f1 = test(gru, dataset[dataset_train_index:], labels[(dataset_train_index / 96):])
    print('Result accuracy = {:.4f}, precision = {:.4f}, recall = {:.4f}, f1 = {:.4f} .'.format(accuracy, precision,
                                                                                      recall, f1))
                                                                                      '''
    # print(df)
