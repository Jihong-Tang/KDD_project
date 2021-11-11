import pandas as pd
import tqdm
from pathlib import Path
import math

def load_data(data_path, slide_window=96, bin=26):
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
                        skip = True
                        break
                if skip:
                    break
                index_cut = math.floor(len(df) / slide_window) * slide_window
                df_temp = df[:index_cut]
                df_temp['id'] = id
                total_data = total_data.append(df_temp, ignore_index=True)
            if not skip:
                total_data.to_csv('dataset\\{}.csv'.format(id), index=False)
