import pandas as pd
import numpy as np
import env
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import QuantileTransformer
 
def new_zillow_data(SQL_query, url):
    '''
    this function will:
    - take in a SQL_query 
    -create a connection url to mySQL
    -return a df of the given query from the zillow database
    
    '''
    
    url= f'mysql+pymysql://{env.username}:{env.password}@{env.hostname}/zillow'
    return pd.read_sql(SQL_query,url)    
    
    
    
def get_zillow_data(filename = "zillow_data.csv"):
    '''
    this function will:
    -check local directory for csv file
        return if exists
    if csv doesn't exist
    if csv doesnt exist:
        - create a df of the SQL_query
        write df to csv
    output zillow df
    
    '''
    directory = os.getcwd()
    
    SQL_query = '''select bedroomcnt, bathroomcnt, calculatedfinishedsquarefeet, taxvaluedollarcnt, 
                    yearbuilt, taxamount, fips, propertylandusedesc from properties_2017
                    join propertylandusetype using(propertylandusetypeid)
                    where propertylandusedesc like 'Single Family Residential';'''
    
    filename = "zillow_data.csv"
    
    url= f'mysql+pymysql://{env.username}:{env.password}@{env.hostname}/zillow'

    if os.path.exists(directory + filename):
        df = pd.read_csv(filename)
        return df
    else:
        df= new_zillow_data(SQL_query, url)
        df.to_csv(filename)
        return df
    
    
def preparing_data_zillow(df):
    '''
    droping unwanted rows for zillow first exercise
    converting float columns to integers
    '''
    df = df.dropna()
    df.calculatedfinishedsquarefeet = df.calculatedfinishedsquarefeet.astype(int).copy()
    df.fips = df.fips.astype(int).copy()
    df.yearbuilt = df.yearbuilt.astype(int).copy()

    return df


def wrangle_zillow():
    '''
    gets and prepares zillow data
    '''
    df = get_zillow_data()
    df = preparing_data_zillow(df)
    return df



def split_data(df, target):
    '''
    Takes in a dataframe and returns train, validate, test subset dataframes
    '''
    
    
    train, test = train_test_split(df,
                                   test_size=.2, 
                                   random_state=123, 
                                   
                                   )
    train, validate = train_test_split(train, 
                                       test_size=.25, 
                                       random_state=123, 
                                       
                                       )
    
    return train, validate, test


def scaler_quantile_normal(X_train, X_validate, X_test):
    '''
    takes in data and uses a QuantileTransformer on it
    with the hyperperameter output_distribution == 'normal'
    '''
    scaler = QuantileTransformer(output_distribution='normal')
    return scaler.fit_transform(X_train), scaler.fit_transform(X_validate), scaler.fit_transform(X_test)


def scaler_quantile_default(X_train, X_validate, X_test):
    '''
    takes in data and uses a QuantileTransformer on it
    '''
    scaler = QuantileTransformer()
    return scaler.fit_transform(X_train), scaler.fit_transform(X_validate), scaler.fit_transform(X_test)


def scaler_min_max(X_train, X_validate, X_test):
    '''
    takes train, test, and validate data and uses the min max scaler on it
    '''
    scaler = MinMaxScaler()
    return scaler.fit_transform(X_train), scaler.fit_transform(X_validate), scaler.fit_transform(X_test)

def scaler_robust(X_train, X_validate, X_test):
    '''
    takes train, test, and validate data and uses the RobustScaler on it
    '''
    scaler = RobustScaler()
    return scaler.fit_transform(X_train), scaler.fit_transform(X_validate), scaler.fit_transform(X_test)


def standard_scaler(X_train, X_validate, X_test):
    '''
    takes train, test, and validate data and uses the standard_scaler on it
    '''
    scaler = StandardScaler()
    return scaler.fit_transform(X_train), scaler.fit_transform(X_validate), scaler.fit_transform(X_test)

    