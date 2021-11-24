# KDD_project
This is the repository to store the codes and materials in the KDD group project for Group (No. 2)

# Project information

## Basic information

- Project topic: classification
- Group number: N0.2

## Topic and Title

```
Topic: classification.
Title: COVID-19 Classification Based on Cough Sound
```

## People
```
Chung-chi,
Jiabao,
Zongchao,
Jihong,
Yubo,
Lingyun
```

<!-- **Say hello here to your group members**


> Hello~ from Jihong
> Hello~ from Jiabao
> Hello~ from
> Hello~ from
> Hello~ from
> Hello~ from -->

# Data

The data we are using is collected from [Coswara](https://github.com/iiscleap/Coswara-Data).

# Model

The models applied in this project are as follows:

- VAE
- GRU
- LSTM
- VGGish
- transformer-AST

# Results

The results are shown in the [final paper](https://github.com/Jihong-Tang/KDD_project/blob/main/paperwork/final_report/final_report.pdf).

# template

how to compile and execute
the description of each source file
an example to show how to run the program
the operating system you tested your program (e.g., linux and Windows)
anything you want to include

---

# Example to run

## data preprecess
### Environment
Linux
### Run the code

```
python clean2_readTheFile.py # to clean the audio files with less audio files
python clean2_readFileOverviewCsv.py # to clean the audio files with damaged audio files
python mfcc.py # to extract the MFCC features

```

## VAE
### Environment
Linux
### Run the code
```
python load_npy.py # to use the saved data
python main.py # to run the VAE model
```

## GRU
Environment:

Python 3.6, torch, torchnet, pandas, sklearn, tqdm

### Run the code
```
python main.py --data_dir={data_dir} --label={label}
```

--data_dir: Dir that store the MFCC preprocessed data, 'distribution/' by defult

--label: The csv file that store the preprocessed labels and other informations, 'combined_data_clean2.csv' by defult

After running the code, the processed dataset which used for input of the GRU model would be stored in the dataset folder.



---

# References

References are listed [here](https://github.com/Jihong-Tang/KDD_project/blob/main/paperwork/final_report/final_report.pdf).
