import pandas as pd
import numpy as np
import env
import matplotlib.pyplot as plt
import seaborn as sns 
import os
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import wrangle
import warnings
warnings.filterwarnings('ignore')


def plot_variable_pairs(train):
    corr_train = train.corr()
    plt.figure(figsize=(8,6))
    sns.heatmap(corr_train, cmap='Purples', annot=True, linewidth = .5)
    plt.ylim(0,4)
    plt.show()
    for row, i in enumerate(train.columns):
        for count, j in enumerate(train.columns):
            
            sns.jointplot(x=train[i],y=train[j],data = train, kind='scatter')
            plt.show()
    


the_dict = {'continuous':['calculatedfinishedsquarefeet','taxvaluedollarcnt','taxamount','yearbuilt'],
           'catagorical':['bedroomcnt','bathroomcnt','fips']}
def plot_categorical_and_continuous_vars(train_subset,the_dict):
   
    for i in range(len(the_dict['catagorical'])):
        for j in range(len(the_dict['continuous'])):
            sns.swarmplot( y=train_subset[the_dict['continuous'][j]], x = train_subset[the_dict['catagorical'][i]], data=train_subset)
            plt.show()
            sns.stripplot(y=train_subset[the_dict['continuous'][j]], x = train_subset[the_dict['catagorical'][i]], data=train_subset)
            plt.show()
            sns.violinplot(y=train_subset[the_dict['continuous'][j]], x = train_subset[the_dict['catagorical'][i]], data=train_subset)
            plt.show()