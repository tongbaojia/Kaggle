import re
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import pandas as pd

def get_title(name):
    '''For feature engineering: 
    return a title string'''
    title_search = re.search(' ([A-Za-z]+)\.', name)
    # If the title exists, extract and return it.
    if title_search:
        return title_search.group(1)
    return "Empty"

def get_number(name):
    '''For feature engineering: 
    return a number string'''
    if " " in name:
        s = name.split(" ")[1]
    else:
        s = name
    num_search = re.findall('\d+', s)
    # If the title exists, extract and return it.
    if num_search:
        return int(num_search[0])
    return 0

##check if there is nan
def check_null(list_df):
    '''For training preparation:
    Check if in a df is null, and print them
    takes a list of dataframes'''
    for dataset in list_df:
        null_columns=dataset.columns[dataset.isnull().any()]
        if dataset.isnull().values.any():
            print(dataset[dataset.isnull().any(axis=1)][null_columns].head())
    return None

def pack_data(df, keyword="Survived", split=0.05, shuffle=True, SEED = 24):
    '''For training preparation:
    split data, and return a dictionary, 
    containing training, testing and validation
    seed for reproducibility
    '''
    data = {}
    train_t, test_t = train_test_split(df, train_size=1 - split, test_size=split, random_state=SEED, shuffle=shuffle)
    train_t, val_t  = train_test_split(train_t, train_size=1 - split, test_size=split, shuffle=False)
    data["X_train"] = train_t.drop(keyword, axis=1)
    data["y_train"] = train_t[keyword]
    #print(data["y_train"])
    data["X_val"]   = val_t.drop(keyword, axis=1)
    data["y_val"]   = val_t[keyword]
    data["X_test"]  = test_t.drop(keyword, axis=1)
    data["y_test"]  = test_t[keyword]
    return data

def upsample_data(df, keyword="Survived"):
    '''For training preparation:
    Check if there is null, and print them
    takes a list of dataframes'''
    # Separate majority and minority classes
    index = df[keyword]
    df_dead = df[index==0]
    df_live = df[index==1] 
    #print(df_dead.head())
    #print(df_live.head())
    
    n_dead = df_dead.shape[0]
    n_live = df_live.shape[0] 
    diff_ratio = n_dead // n_live if n_dead > n_live else  n_live // n_dead 
    #print(diff_ratio)
    if n_dead > n_live and diff_ratio > 2:
        # Upsample minority class
        df_temp = resample(df_live, replace=True, n_samples=n_dead//2, random_state=123) # reproducible results
        return pd.concat([df_dead, df_temp])
    elif n_live > n_dead and diff_ratio > 2:
        # Upsample minority class
        df_temp = resample(df_dead, replace=True, n_samples=n_live//2, random_state=123) # reproducible results
        return pd.concat([df_live, df_temp])
    else:
        return pd.concat([df_live, df_dead])
