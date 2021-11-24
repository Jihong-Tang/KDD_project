import argparse
import random

import numpy as np
# from matplotlib import pyplot as plt

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score

from data import CoswaraDataset
from models import LSTM

parser = argparse.ArgumentParser()
parser.add_argument('--base_dir', dest='base_dir', type=str, default='/Users/mental/datasets/coswara-data/',
                    help='the base directory of the coswara dataset')
parser.add_argument('--epochs', '-e', type=int, default=40)
parser.add_argument('--batch_size', '-bs', type=int, default=32)
parser.add_argument('--seq_len', type=int, default=96)
parser.add_argument('--model', '-m', type=str, default='lstm',
                    choices=['lstm'])
parser.add_argument('--hidden_dim', type=int, default=64)
parser.add_argument('--num_layers', type=int, default=4)
parser.add_argument('--dropout', type=float, default=0)
parser.add_argument('--optimizer', '-o', default='adam',
                    choices=['adam', 'sgd'])
parser.add_argument('--lr', type=float, default=0.001)

labels_legend = {
    "id": "User ID",
    "a": "Age (number)",
    "covid_status": "Health status (e.g., positive_mild, healthy, etc.)",
    "record_date": "Date when the user recorded and submitted the samples",
    "ep": "Proficient in English (y/n)",
    "g": "Gender (male/female/other)",
    "l_c": "Country",
    "l_l": "Locality",
    "l_s": "State",
    "rU": "Returning User (y/n)",
    "asthma": "Asthma (True/False)",
    "cough": "Cough (True/False)",
    "smoker": "Smoker (True/False)",
    "test_status": "Status of COVID Test (p->Positive, n->Negative, na->Not taken Test)",
    "ht": "Hypertension  (True/False)",
    "cold": "Cold (True/False)",
    "diabetes": "Diabetes (True/False)",
    "diarrhoea": "Diarrheoa (True/False)",
    "um": "Using Mask (y/n)",
    "ihd": "Ischemic Heart Disease (True/False)",
    "bd": "Breathing Difficulties (True/False)",
    "st": "Sore Throat (True/False)",
    "fever": "Fever (True/False)",
    "ftg": "Fatigue (True/False)",
    "mp": "Muscle Pain (True/False)",
    "loss_of_smell": "Loss of Smell & Taste (True/False)",
    "cld": "Chronic Lung Disease (True/False)",
    "pneumonia": "Pneumonia (True/False)",
    "ctScan": "CT-Scan (y/n if the user has taken a test)",
    "testType": "Type of test (RAT/RT-PCR)",
    "test_date": "Date of COVID Test (if taken)",
    "vacc": "Vaccination status (y->both doses, p->one dose(partially vaccinated), n->no doses)",
    "ctDate": "Date of CT-Scan",
    "ctScore": "CT-Score",
    "others_resp": "Respiratory illnesses other than the listed ones (True/False)",
    "others_preexist": "Pre-existing conditions other than the listed ones (True/False)"
}

# Health Status
# 1. Do you have any of these symptoms?
#   a. Cold
#   b. Cough
#   c. Fever
#   d. Diarrhoea
#   e. Sore Throat
#   f. None of the above
# 2. Do you have any of these conditions?
#   a. Loss of Smell or Taste
#   b. Muscle Pain
#   c. Fatigue
#   d. Breathing Difficulties
#   e. None of the above
# 3. Do you have any of these respiratory ailments?
#   a. Pneumonia
#   b. Asthma
#   c. Chronic Lung Disease
#   d. Others
#   e. None of the above
# 4. Do you have any of these pre-existing conditions?
#   a. Hypertension
#   b. Ischemic Heart Disease
#   c. Diabetes
#   d. Others
#   e. None of the above

# Possible COVID Test Status
# 1. Tested positive in my last test
#   a. Mild symptoms
#   b. Moderate symptoms
#   c. No symptoms (asymptomatic)
#   d. Recovered
# 2. Tested negative in my last test
#   a. May have been exposed to the virus through contact
#   b. Not been exposed to the virus through contact
# 3. Never taken a test
#   a. May have been exposed to the virus through contact
#   b. Not been exposed to the virus through contact

# covid_status = {
#     'positive_moderate': 72,
#     'recovered_full': 94,
#     'no_resp_illness_exposed': 162,
#     'positive_mild': 229,
#     'healthy': 1330,
#     'positive_asymp': 41,
#     'resp_illness_not_identified': 135
# }


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    setup_seed(233)

    args = parser.parse_args()
    base_dir = args.base_dir
    epochs = args.epochs
    bs = args.batch_size

    print('Base directory of coswara dataset: {}'.format(base_dir))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device: {}'.format(device))

    dataset = CoswaraDataset(base_dir=base_dir, seq_len=args.seq_len)
    print('The size of dataset: {}'.format(len(dataset)))

    train_idx, test_idx = train_test_split(list(range(len(dataset))), train_size=0.7)
    train_dataset = Subset(dataset, train_idx)
    test_dataset = Subset(dataset, test_idx)

    print('The size of training dataset: {}'.format(len(train_dataset)))
    print('The size of test dataset: {}'.format(len(test_dataset)))

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=bs,
        shuffle=True,
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=128,
        shuffle=False,
    )

    if args.model == 'lstm':
        model = LSTM(input_dim=26, hidden_dim=args.hidden_dim, num_layers=args.num_layers,
                     dropout=args.dropout, num_classes=2, device=device).to(device).float()
    else:
        raise NotImplementedError

    criterion = nn.CrossEntropyLoss()
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    else:
        raise NotImplementedError
    total_step = len(train_loader)
    for epoch in range(epochs):
        model.train()
        for idx, (audios, labels) in enumerate(train_loader):
            # print(idx, audios.shape, labels.shape)
            audios = audios.to(device)
            labels = labels.to(device)

            outputs = model(audios.float())
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if idx % 10 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch + 1, epochs, idx + 1, total_step, loss.item()))

        model.eval()
        with torch.no_grad():
            TP, TN, FN, FP = 0, 0, 0, 0
            score_list = []
            label_list = []
            for audios, labels in test_loader:
                audios = audios.to(device)
                labels = labels.to(device)

                outputs = model(audios.float())

                _, predicted = torch.max(outputs.data, 1)

                TP += ((predicted == 1) & (labels.data == 1)).cpu().sum()
                TN += ((predicted == 0) & (labels.data == 0)).cpu().sum()
                FP += ((predicted == 1) & (labels.data == 0)).cpu().sum()
                FN += ((predicted == 0) & (labels.data == 1)).cpu().sum()

                score_list.extend(predicted.detach().cpu().numpy())
                label_list.extend(labels.cpu().numpy())

            precision = TP / (TP + FP)
            recall = TP / (TP + FN)
            F1 = 2 * recall * precision / (recall + precision)
            accuracy = (TP + TN) / (TP + TN + FP + FN)

            roc_auc = roc_auc_score(np.array(label_list), np.array(score_list))
            # fpr, tpr, _ = roc_curve(np.array(label_list), np.array(score_list))

            print('Epoch [{}/{}], ROC-AUC: {}, Precision: {}, Recall: {}, F1 score: {}, Accuracy: {}'.format(
                epoch + 1, epochs, roc_auc, precision, recall, F1, accuracy)
            )

            # plt.plot(fpr, tpr)
            # plt.show()
