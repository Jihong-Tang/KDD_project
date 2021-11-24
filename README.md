# KDD_project
This is the repository to store the codes and materials in the KDD group project for Group (No. 2)


# People

**Say hello here to your group members**


> Hello~ from Jihong
> Hello~ from Jiabao
> Hello~ from
> Hello~ from
> Hello~ from
> Hello~ from


# Topic and Title

Topic: classification.

Title: COVID-19 Classification Based on Cough Sound

# Data

The data we are using is collected from Coswara (citation)...

# Model

The models are as follows:

- VAE
- GRU
- LSTM
- VGGish
- transformer

# Result

# Example to run

Some descriptions: >>>

```
python runTest.py
```

# References



# GRU
Environment:

Python 3.6, torch, torchnet, pandas, sklearn, tqdm

# Run the code
```
python main.py --data_dir={data_dir} --label={label}
```

--data_dir: Dir that store the MFCC preprocessed data, 'distribution/' by defult

--label: The csv file that store the preprocessed labels and other informations, 'combined_data_clean2.csv' by defult

After running the code, the processed dataset which used for input of the GRU model would be stored in the dataset folder.
