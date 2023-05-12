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
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import SelectKBest, RFE, f_regression, SequentialFeatureSelector
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.linear_model import LassoLars
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import TweedieRegressor

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
    df = df.drop(columns = ['Unnamed: 0', 'propertylandusedesc'])
    df.calculatedfinishedsquarefeet = df.calculatedfinishedsquarefeet[df.calculatedfinishedsquarefeet < 25000]
    
    df = df[df.taxvaluedollarcnt < df.taxvaluedollarcnt.quantile(.95)].copy()
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

def rfe(X_train, y_train, the_k):
    model = LinearRegression()
    rfe = RFE(model, n_features_to_select=the_k)
    rfe.fit(X_train, y_train)
    the_df = pd.DataFrame(
    {'rfe_ranking':rfe.ranking_},
    index=X_train.columns)
    return the_df[the_df['rfe_ranking'] == 1]

def select_kbest(X_train, y_train, the_k):
    kbest = SelectKBest(f_regression, k=the_k)
    kbest.fit(X_train, y_train)
    return X_train.columns[kbest.get_support()]

def metrics_reg(y, yhat):
    '''
    send in y_true, y_pred and returns rmse, r2
    '''
    rmse = mean_squared_error(y, yhat, squared=False)
    r2 = r2_score(y, yhat)
    return rmse, r2

def get_X_train_val_test(train,validate, test, x_target, y_target):
    '''
    geting the X's and y's and returns them
    '''
    X_train = train.drop(columns = x_target)
    X_validate = validate.drop(columns = x_target)
    X_test = test.drop(columns = x_target)
    y_train = train[y_target]
    y_validate = validate[y_target]
    y_test = test[y_target]

    return X_train, X_validate, X_test, y_train, y_validate, y_test


def get_model_numbers(X_train, X_validate, X_test, y_train, y_validate, y_test):
    baseline = y_train.mean()
    baseline_array = np.repeat(baseline, len(X_train))
    rmse, r2 = metrics_reg(y_train, baseline_array)

    metrics_train_df = pd.DataFrame(data=[
    {
        'model_train':'baseline',
        'rmse':rmse,
        'r2':r2
    }
    ])
    metrics_validate_df = pd.DataFrame(data=[
    {
        'model_validate':'baseline',
        'rmse':rmse,
        'r2':r2
    }
    ])


    Linear_regression1 = LinearRegression()
    Linear_regression1.fit(X_train,y_train)
    predict_linear = Linear_regression1.predict(X_train)
    rmse, r2 = metrics_reg(y_train, predict_linear)
    metrics_train_df.loc[1] = ['ordinary least squared(OLS)', rmse, r2]

    predict_linear = Linear_regression1.predict(X_validate)
    rmse, r2 = metrics_reg(y_validate, predict_linear)
    metrics_validate_df.loc[1] = ['ordinary least squared(OLS)', rmse, r2]


    lars = LassoLars()
    lars.fit(X_train, y_train)
    pred_lars = lars.predict(X_train)
    rmse, r2 = metrics_reg(y_train, pred_lars)
    metrics_train_df.loc[2] = ['lasso lars(lars)', rmse, r2]

    pred_lars = lars.predict(X_validate)
    rmse, r2 = metrics_reg(y_validate, pred_lars)
    metrics_validate_df.loc[2] = ['lasso lars(lars)', rmse, r2]


    pf = PolynomialFeatures(degree=2)
    X_train_degree2 = pf.fit_transform(X_train)
   

    pr = LinearRegression()
    pr.fit(X_train_degree2, y_train)
    pred_pr = pr.predict(X_train_degree2)
    rmse, r2 = metrics_reg(y_train, pred_pr)
    metrics_train_df.loc[3] = ['Polynomial Regression(poly2)', rmse, r2]

    X_validate_degree2 = pf.transform(X_validate)
    pred_pr = pr.predict(X_validate_degree2)
    rmse, r2 = metrics_reg(y_validate, pred_pr)
    metrics_validate_df.loc[3] = ['Polynomial Regression(poly2)', rmse, r2]


    glm = TweedieRegressor(power=1, alpha=0)
    glm.fit(X_train, y_train)
    pred_glm = glm.predict(X_train)
    rmse, r2 = metrics_reg(y_train, pred_glm)
    metrics_train_df.loc[4] = ['Generalized Linear Model (GLM)', rmse, r2]

    pred_glm = glm.predict(X_validate)
    rmse, r2 = metrics_reg(y_validate, pred_glm)
    metrics_validate_df.loc[4] = ['Generalized Linear Model (GLM)', rmse, r2]

    return metrics_train_df, metrics_validate_df
