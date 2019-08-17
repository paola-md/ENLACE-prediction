#####################################################
# Data preparation
# Last edition: 08/16/2019
#
######################################################

import os
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
% matplotlib inline
import seaborn as sns
import sklearn

#Feature extraction
from sklearn import preprocessing
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.linear_model import RidgeCV, LassoCV, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor

from sklearn.decomposition import PCA

#Data cleaning
from sklearn.ensemble import ExtraTreesRegressor as ETRg

#Modeling 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


#=========================================
#Directories
#=========================================
source = os.path.expanduser('C:\\Users\\pmeji\\Dropbox\\ITAM\\OctavoSemestre\\CuestionarioContexto\\build\\source\\bases')


#=========================================
#Methods
#=========================================
#Feature selection
def selectFeatures(X, y, thresh, top, num, obj, alfaR):
    X_ex = exclude(X, thresh)
    X_select = include(X_ex, y, top, num, obj, alfaR)
    return X[X_select.columns].join(y).join(X.cct)

#Feature extraction: Exclude
def exclude(X, var_tresh):
    #Exclude string variables which must be analyzed separetely
    X_num = X.select_dtypes(include = ['float', 'int64']) #Exclude string variables. 
    #Removes variables with low variance
    #Normalizes variables
    X_norm = normalize(X_num.fillna(0)).dropna(axis=1)  
    constant_filter = VarianceThreshold(threshold=var_tresh)
    constant_filter.fit(X_norm)
    constant_columns = [column for column in X_norm.columns
                    if column not in X_norm.columns[constant_filter.get_support()]]
    X_filtered = X_norm[X_norm.columns[constant_filter.get_support()]]
    return X_filtered

#Feature extraction: Include
def include(X, y, top, num, obj, alphaR):
    #filter method
    Xcorr = include_filter(X,y,top, obj)
    
    #Wrapper methods
    kbest_feat = include_kbest(Xcorr, y, num)
    forest_feat = include_forest(Xcorr, y, num)
    ols_feat = include_ols(Xcorr, y, num)
    rfe_feat = include_rfe(Xcorr, y, num)
    
    #Embedded method
    resultList= list(set(rfe_feat) | set(ols_feat)| set(forest_feat) | set(kbest_feat))
    if 'const' in resultList:
        resultList.remove('const')
    Xwrap = Xcorr[resultList]
    lasso_feat = select_lasso(Xwrap, y, alphaR, num)
    ridge_feat = select_ridge(Xwrap, y, alphaR, num)
    
    #Result
    finalList= list(set(ridge_feat) | set(lasso_feat))
    return Xwrap[finalList]
    
    
def include_filter(X, y, top, obj):
    df_xy = X.join(y) #Ya eliminamos las que tienen varianza menor a 0.01
    df_corr = df_xy.corr()[obj].sort_values(ascending=False)
    top_vals = abs(df_corr).sort_values(ascending=False)[1:1+top].index.tolist()
    return X[top_vals].fillna(0)

def include_kbest(X, y, num):
    #apply SelectKBest class to extract top 10 best features
    bestfeatures = SelectKBest(score_func=sklearn.feature_selection.f_regression, k=num)
    fit = bestfeatures.fit(X,y)
    dfscores = pd.DataFrame(fit.scores_)
    dfcolumns = pd.DataFrame(X.columns)
    #concat two dataframes for better visualization 
    featureScores = pd.concat([dfcolumns,dfscores],axis=1)
    featureScores.columns = ['Specs','Score']  #naming the dataframe columns
    kbest_feat = featureScores.sort_values('Score',ascending=False)[:num]['Specs'].unique().tolist()
    return kbest_feat

def include_forest(X, y, num):
    rf = RandomForestRegressor(n_estimators = 100,
                               n_jobs = -1,
                               oob_score = True,
                               bootstrap = True,
                               random_state = 42)
    rf.fit(X, y)
    feature_importances = pd.DataFrame(rf.feature_importances_, index = X.columns,columns=['importance']).sort_values('importance',ascending=False)
    forest_feat = list(feature_importances.index[:num])
    return forest_feat 

def include_ols(X, y, num):
    X_1 = sm.add_constant(X)
    #Fitting sm.OLS model
    model = sm.OLS(y,X_1).fit()
    ols_feat = model.pvalues.sort_values(ascending=False)[:num].index.tolist()
    return ols_feat

def include_rfe(X, y, num):
    #Initializing RFE model
    model = LinearRegression()
    rfe = RFE(model , num)
    #Transforming data using RFE
    X_rfe = rfe.fit_transform(X,y)  
    #Fitting the data to model
    model.fit(X_rfe,y)
    rfe_feat = X.columns[list(rfe.ranking_==1)].tolist()
    return rfe_feat

def select_lasso(X, y, a,num):
    reg = Lasso(alpha=a)
    reg.fit(X, y)
    coef = pd.Series(reg.coef_, index = X.columns)
    imp_coef = coef.sort_values()
    lasso_feat = abs(imp_coef[imp_coef>0]).sort_values(ascending=False).index.tolist()[:num]
    return lasso_feat

def select_ridge(X, y, a,num):
    reg = Ridge(alpha=a)
    reg.fit(X, y)
    coef = pd.Series(reg.coef_[0], index = X.columns)
    imp_coef = coef.sort_values()
    ridge_feat = abs(imp_coef[imp_coef>0]).sort_values(ascending=False).index.tolist()[:num]
    return ridge_feat

def normalize(dataset):
    dataNorm=((dataset-dataset.min())/(dataset.max()-dataset.min()))
    return dataNorm


#Miss Forest
def fill_missings(df):
    print(df.isnull().sum().sum())
    features = df.columns[df.isnull().sum()==0].tolist()
    if 'cct' in features:
        features.remove('cct')
    if 'Unnamed' in features:
        features.remove('Unnamed')

    missingVars = df.columns[df.isnull().sum()>0].tolist()

    for obj in missingVars:
        Etr = ETRg()
        XTrain = df[features][df[obj].notnull()] 
        YTrain = df[obj][df[obj].notnull()]
        XTest = df[features][df[obj].isnull()] 
        Etr.fit(XTrain,np.ravel(YTrain)) 
        Pred = Etr.predict(XTest)
        df.loc[df[obj].isnull(), obj] = Pred
        print(df.isnull().sum().sum())

    print(df.isnull().sum().sum())
    return df

def gen_PCA(df, num_comp):
    X = df.drop(['p_mat_std','cct'], axis=1)
    Xn = normalize(X)
    pca = PCA(n_components=num_comp)
    pca_result = pca.fit_transform(Xn.values)
    df['pca-one'] = pca_result[:,0]
    df['pca-two'] = pca_result[:,1] 
    df['pca-three'] = pca_result[:,2]
    df['pca-four'] = pca_result[:,3]
    print(sum(pca.explained_variance_ratio_))
    df_pca = df[['pca-one', 'pca-two', 'pca-three', 'pca-four','p_mat_std', 'cct']]
    return df_pca 

#Modelling
def calculate_AIC(y_test, y_pred, k):
    dif = np.array(y_test).ravel()- y_pred.ravel()
    emv = sum(dif ** 2)
    AIC = 2*k - 2*np.log(emv)
    return AIC
    
    
#===============================================
# Code 
#===============================================
filename = 'general_final.csv'
addr= os.path.join(source, filename)
df_g = pd.read_csv(addr, encoding = 'latin-1')

X = df_g.drop(['p_mat_std'], axis=1)
y = df_g[['p_mat_std']].astype('float64')

#Feature selection
thresh = 0.001
numCorr = ceil(len(df_g.columns)/4)
numFeat = ceil(len(df_g)/2000)
alphaR = 0.0001
varObj =  'p_mat_std'
df_general =selectFeatures(X, y, thresh, numCorr, numFeat, varObj,alphaR)

#Cleaning
df_general_clean = fill_missings(df_general)
df_general_clean.to_csv('select_general_clean.csv')

df_PCA = gen_PCA(df_general_clean)

#Modelling
X = df_PCA.drop(['p_mat_std', 'Unnamed: 0','cct'], axis=1)
y = df_PCA[['p_mat_std']].astype('float64')

X_trainp, X_testp, y_trainp, y_testp = train_test_split(X , y, 
                                                    test_size=0.2, 
                                                    random_state=15)

lm = LinearRegression()
lm.fit(X_train,y_train)
y_pred = lm.predict(X_test)
print(calculate_AIC(y_test, y_pred, len(X_train.columns)))




