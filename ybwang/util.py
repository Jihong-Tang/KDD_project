import pandas as pd
import tqdm
from pathlib import Path
import math
from sklearn.preprocessing import LabelEncoder


def load_data(data_path, slide_window=96, bin=26):  # 如果出现小于-20情况跳过当前音频文件
    # total_data = pd.DataFrame(columns=[str(i) for i in range(bin)].append('id'))
    for file_path in tqdm.tqdm(list(Path(data_path).iterdir()), desc='Date'):
        file_path = file_path.absolute()
        path1 = Path.joinpath(file_path, file_path.stem)
        for path2 in Path(path1).iterdir():
            skip = False
            total_data = pd.DataFrame(columns=[str(i) for i in range(bin)].append('id'))
            # id = os.path.basename(path2.name)
            id = path2.stem
            for path3 in Path(path2).glob('*.dat'):
                df = pd.read_csv(path3, header=None)
                for item in df.min():
                    if item < -20.0:
                        print('Drop id: {}'.format(id))
                        skip = True
                        break
                if skip:
                    break
                index_cut = math.floor(len(df) / slide_window) * slide_window
                df_temp = df[:index_cut]
                df_temp['id'] = id
                total_data = total_data.append(df_temp, ignore_index=True)
            if not skip:
                total_data.to_csv('dataset/{}.csv'.format(id), index=False)


def label_preprocessing(data_path):  # 预处理label
    data_label = pd.read_csv(data_path, index_col='id')
    data_label.dropna(axis=1, inplace=True)
    data_label.drop('record_date', axis=1, inplace=True)
    for key, item in data_label.iterrows():
        if item['covid_status'] == 'healthy':
            data_label.loc[key, 'covid_status'] = 0
        else:
            data_label.loc[key, 'covid_status'] = 1
    data_label['ep'] = LabelEncoder().fit(data_label['ep']).transform(data_label['ep'])
    data_label['g'] = LabelEncoder().fit(data_label['g']).transform(data_label['g'])
    data_label['l_c'] = LabelEncoder().fit(data_label['l_c']).transform(data_label['l_c'])
    data_label['l_s'] = LabelEncoder().fit(data_label['l_s']).transform(data_label['l_s'])
    return data_label


def dict_slice(adict, start, end):
    keys = list(adict.keys())
    dict_slice = {}
    for k in keys[int(start):int(end)]:
        dict_slice[k] = adict[k]
    return dict_slice


def preprocess_table(path, data_label):  # 预处理table，将label整合
    dict_data = {}
    for i, data_path in enumerate(Path(path).glob('*.csv')):
        #print('Read data path {}'.format(data_path))
        dataset = pd.read_csv(data_path)
        if not dataset.empty:
            dict_data_temp = {}
            id = dataset['id'].iloc[0]
            dataset.drop('id', axis=1, inplace=True)
            dataset['ep'] = data_label['ep'].loc[id]
            dataset['g'] = data_label['g'].loc[id]
            dataset['l_c'] = data_label['l_c'].loc[id]
            dataset['l_s'] = data_label['l_s'].loc[id]
            dataset['a'] = data_label['a'].loc[id]
            dataset['status'] = data_label['covid_status'].loc[id]
            # print(dataset.shape)
            dict_data_temp[id] = dataset
            dict_data[i] = dict_data_temp
    return dict_data
