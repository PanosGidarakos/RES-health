import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.pipeline import Pipeline


def percentage_of_misses(true, predicted):
    ratio = np.sum(true < predicted) / len(true)
    return ratio

def naive_baseline(df_test):
       
    preds = []
    for i in range(df_test.shape[0]):
        t = [x for x in df_test.columns if 'Grd_Prod_Pwr_min_(t+' in x]
        pred = np.zeros(len(t))
        pred =  pred + df_test.iloc[i, :]['Grd_Prod_Pwr_min']
        preds.append(pred)
    preds = np.vstack(preds)
    return preds

def mape1(y_true, y_pred):

    """
    Computes the Mean Absolute Percentage Error between the 2 given time series

    Args:
        y_true: A numpy array that contains the actual values of the time series.
        y_pred: A numpy array that contains the predicted values of the time series.

    Return:
        Mean Absolute Percentage Error value.


    """
    ola=[]
    if y_true.ndim >= 2 and y_pred.ndim >= 2:
        mapes = []

        nom = np.sum(np.abs(y_true - y_pred), axis=1)
        denom = np.sum(np.abs(y_true + y_pred), axis=1)
        
#         denom = np.sum(np.abs(y_true), axis=1)

        mapes = nom/denom
        mape1 = np.mean(mapes)
        return mape1
    else:
        y_true = y_true.ravel()
        y_pred = y_pred.ravel()
        mape1 = (np.mean(np.abs(y_true-y_pred)/np.mean(y_true)))
        mape2 = (np.sum(np.abs(y_true-y_pred)/np.sum(y_true+y_pred)))
        mape3 = (np.sum(np.abs(y_true-y_pred)/np.abs(np.sum(y_true+y_pred))))
        mape4 = (np.sum(np.abs(y_true-y_pred)/np.sum(np.abs(y_true+y_pred))))
        mape5 = (np.mean(np.abs(y_true-y_pred)/np.abs(np.mean(y_true))))
        mape6 = (np.mean(np.abs(y_true-y_pred)/(np.mean(np.abs(y_true)))))
                 

        ola.append([mape1, mape2, mape3, mape4, mape5, mape6])

        return mape4

def mpe1(y_true, y_pred):

    """
    Computes the Mean Percentage Error between the 2 given time series.

    Args:
        y_true: A numpy array that contains the actual values of the time series.
        y_pred: A numpy array that contains the predicted values of the time series.

    Return:
        Mean Absolute Error value.


    """
    mpe1 = (np.mean(y_true-y_pred)/np.mean(y_true))
    return mpe1


def score(y_true, y_pred):

    """
    Computes a set of values that measure how well a predicted time series matches the actual time series.

    Args:
        y_true: A numpy array that contains the actual values of the time series.
        y_pred: A numpy array that contains the predicted values of the time series.

    Return:
        Returns a value for each of the following measures:
        r-squared, mean absolute error, mean error, mean absolute percentage error, mean percentage error, median



    """
    r_sq = metrics.r2_score(y_true, y_pred)
    mae = metrics.mean_absolute_error(y_true, y_pred)
    me = np.mean(y_true-y_pred)
    mape = mape1(y_true, y_pred)
    mpe = mpe1(y_true, y_pred)
    med = np.median(y_true-y_pred)

    return r_sq, mae, me, mape, mpe, med


def filter_dates(df, start, end):
    """ 
    Remove rows of the dataframe that are not in the [start, end] interval.
    
    Args:
        df:DataFrame that has a datetime index.
        start: Date that signifies the start of the interval.
        end: Date that signifies the end of the interval.
   
   Returns:
        The Filtrared TimeSeries/DataFrame
    """
    date_range = (df.index >= start) & (df.index <= end)
    df = df[date_range]
    return df

def outliers_IQR(df):
    Q1 = df.quantile(0.10)
    Q3 = df.quantile(0.90)
    IQR = Q3 - Q1
    df_iqr = df[~((df < (Q1 - 1.5 * IQR)) | (df >(Q3 + 1.5 * IQR))).any(axis=1)]
    return df_iqr



def container_c(df,test_index,future_steps,pipeline,df_test,fit_features,target_features):
    p_list_pred = []
    p_list = []
    all_recalls = []

    for l in np.unique(df.label):
        try:
            unq_idx = test_index.drop_duplicates()
            temp = np.empty((unq_idx.shape[0], future_steps))
            temp[:] = 1e-6
            result_container = pd.DataFrame(temp.copy(), index=unq_idx)
            y_pred_test = pipeline.predict(df_test.loc[df_test.label==l][fit_features].values)
            result_temp = pd.DataFrame(y_pred_test, index=df_test.loc[df_test.label==l].index)
            result_container.loc[result_temp.index] = result_temp
            gt_container = pd.DataFrame(temp.copy(), index=unq_idx)
            y_test = df_test.loc[df_test.label==l][target_features]
            gt_temp = pd.DataFrame(y_test, index=df_test.loc[df_test.label==l].index)
            gt_container.loc[gt_temp.index] = gt_temp
            p_list_pred.append(result_container.copy())
            p_list.append(gt_container.copy())
            all_recalls.append(recall_score(y_test,y_pred_test,average='macro',zero_division=0))
        except:
            print(l)
    return p_list_pred,p_list,all_recalls

# def reg_container(df,test_index,future_steps,pipeline,df_test,fit_features,pipeline_lower,target_features):
#     all_mapes = []
#     p_list_pred = []
#     p_list_pred_lower = []
#     p_list = []
#     for l in np.unique(df.label):
#         try:
#             unq_idx = test_index.drop_duplicates()
#             temp = np.empty((unq_idx.shape[0], future_steps))
#             temp[:] = 1e-6
#             result_container = pd.DataFrame(temp.copy(), index=unq_idx)

#             y_pred_test = pipeline.predict(df_test.loc[df_test.label==l][fit_features].values)
#             result_temp = pd.DataFrame(y_pred_test, index=df_test.loc[df_test.label==l].index)
#             result_container.loc[result_temp.index] = result_temp

#             result_container_lower = pd.DataFrame(temp.copy(), index=unq_idx)
#             y_pred_test_lower = pipeline_lower.predict(df_test.loc[df_test.label==l][fit_features].values)
#             result_temp_lower = pd.DataFrame(y_pred_test_lower, index=df_test.loc[df_test.label==l].index)
#             result_container_lower.loc[result_temp_lower.index] = result_temp_lower



#             gt_container = pd.DataFrame(temp.copy(), index=unq_idx)
#             y_test = df_test.loc[df_test.label==l][target_features]
#             gt_temp = pd.DataFrame(y_test, index=df_test.loc[df_test.label==l].index)
#             gt_container.loc[gt_temp.index] = gt_temp
#             p_list_pred.append(result_container.copy())
#             p_list_pred_lower.append(result_container_lower.copy())
#             p_list.append(gt_container.copy())
#             all_mapes.append(mape1(y_test, y_pred_test))
#         except:
#             print(l)
#     return p_list_pred,p_list_pred_lower,p_list,all_mapes


def reg_container(df,test_index,future_steps,pipeline,df_test,fit_features,pipeline_lower,target_features):
    all_mapes = []
    p_list_pred = []
    p_list_pred_lower = []
    p_list = []
    for l in np.unique(df.label):
        try:
            unq_idx = test_index.drop_duplicates()
            temp = np.empty((unq_idx.shape[0], future_steps))
            temp[:] = 1e-6
            result_container = pd.DataFrame(temp.copy(), index=unq_idx)

            y_pred_test = pipeline.predict(df_test.loc[df_test.label==l][fit_features].values)
            result_temp = pd.DataFrame(y_pred_test, index=df_test.loc[df_test.label==l].index)
            result_container.loc[result_temp.index] = result_temp

            result_container_lower = pd.DataFrame(temp.copy(), index=unq_idx)
            y_pred_test_lower =pipeline_lower.predict(df_test.loc[df_test.label==l][fit_features].values)
            result_temp_lower = pd.DataFrame(y_pred_test_lower, index=df_test.loc[df_test.label==l].index)
            result_container_lower.loc[result_temp_lower.index] = result_temp_lower



            gt_container = pd.DataFrame(temp.copy(), index=unq_idx)
            y_test = df_test.loc[df_test.label==l][target_features]
            gt_temp = pd.DataFrame(y_test, index=df_test.loc[df_test.label==l].index)
            gt_container.loc[gt_temp.index] = gt_temp
            p_list_pred.append(result_container.copy())
            p_list_pred_lower.append(result_container_lower.copy())
            p_list.append(gt_container.copy())
            all_mapes.append(mape1(y_test, y_pred_test))
        except:
            print(l)
    return p_list_pred,p_list_pred_lower,p_list,all_mapes


def container_c(df,test_index,future_steps,pipeline,df_test,fit_features,target_features):
    p_list_pred = []
    p_list = []
    all_recalls = []

    for l in np.unique(df.label):
        
        unq_idx = test_index.drop_duplicates()
        temp = np.empty((unq_idx.shape[0], future_steps))
        temp[:] = 1e-6
        result_container = pd.DataFrame(temp.copy(), index=unq_idx)
        y_pred_test = pipeline.predict(df_test.loc[df_test.label==l][fit_features].values)
        result_temp = pd.DataFrame(y_pred_test, index=df_test.loc[df_test.label==l].index)
        result_container.loc[result_temp.index] = result_temp
        gt_container = pd.DataFrame(temp.copy(), index=unq_idx)
        y_test = df_test.loc[df_test.label==l][target_features]
        gt_temp = pd.DataFrame(y_test, index=df_test.loc[df_test.label==l].index)
        gt_container.loc[gt_temp.index] = gt_temp
        p_list_pred.append(result_container.copy())
        p_list.append(gt_container.copy())
        all_recalls.append(recall_score(y_test,y_pred_test,average='macro',zero_division=0))
    return p_list_pred,p_list,all_recalls

def percentage_of_misses(true, predicted):
    ratio = np.sum(true < predicted) / len(true)
    return ratio

def predict(df_test, model, feats, target):
    """
    Applies a regression model to predict values of a dependent variable for a given dataframe and 
    given features.

    Args:
        df_test: The input dataframe.
        model: The regression model. Instance of Pipeline.
        feats: List of strings: each string is the name of a column of df_test.
        target: The name of the column of df corresponding to the dependent variable.
    Returns:
        y_pred: Array of predicted values. 
    """
    
    df_x = df_test[feats]
    df_y = df_test[target]
    X = df_x.values
    y_true = df_y#.values
    y_pred = model.predict(X)
    return y_pred


def fit_pipeline(df, feats, target, pipeline, params):
    """
    Fits a regression pipeline on a given dataframe, and returns the fitted pipline,
    the predicted values and the associated scores.

    Args:
        df: The input dataframe.
        feats: List of names of columns of df. These are the feature variables.
        target: The name of a column of df corresponding to the dependent variable.
        pipeline: A pipeline instance, a scikit-learn object that sequentially applies a list of given 
                  preprocessing steps and fits a selected machine learning model.
        params: A dictionary that contains all the parameters that will be used by `pipeline`.
                The dictionary keys must follow the scikit-learn naming conventions.

    Returns:    
        pipeline: The fitted model. This is an instance of Pipeline.
        y_pred: An array with the predicted values.
        r_sq: The coefficient of determination “R squared”.
        mae: The mean absolute error.
        me: The mean error.
        mape: The mean absolute percentage error.
        mpe: The mean percentage error.
    """
    
    df_x = df[feats]
    df_y = df[target]
    X = df_x.values
    y = df_y.values

    pipeline.set_params(**params)
    pipeline.fit(X, y)
    y_pred = pipeline.predict(X)
    r_sq, mae, me, mape, mpe, _ = score(y, y_pred)
    return pipeline, y_pred, r_sq, mae, me, mape, mpe


# def naive_baseline(df_test):

#     preds = []
#     for i in range(df_test.shape[0]):
#         t = [x for x in df_test.columns if 'possible_stop_(t+' in x]
#         pred = np.zeros(len(t))
#         pred =  pred + df_test.iloc[i, :]['possible_stop']
#         preds.append(pred)
#     preds = np.vstack(preds)
#     return preds
from sklearn.metrics import accuracy_score,f1_score,recall_score,precision_score
def score_clas(y,y_pred):
    return accuracy_score(y,y_pred),f1_score(y,y_pred,average='macro'),recall_score(y,y_pred,average='macro'),precision_score(y,y_pred,average='macro')
# def score_clas(y,y_pred):
#     return accuracy_score(y,y_pred),f1_score(y,y_pred,average='macro')
def fit_pipeline_classifier(df, feats, target, pipeline, params):
    df_x = df[feats]
    df_y = df[target]
    X = df_x.values
    y = df_y.values

    pipeline.set_params(**params)
    pipeline.fit(X, y)
    y_pred = pipeline.predict(X)
    acc,f1,recall,precision = score_clas(y,y_pred)
    return pipeline, y_pred, acc,f1,recall,precision



def pred_gt_list(df,test_index,df_test,fit_features,target_features,pipeline):
    p_list_pred=[]
    p_list=[]
    all_mapes=[]
    for l in np.unique(df.label):
        try:
            unq_idx = test_index.drop_duplicates()
            temp = np.empty((unq_idx.shape[0], future_steps))
            temp[:] = 1e-1
            result_container = pd.DataFrame(temp.copy(), index=unq_idx)
            y_pred_test =pipeline.predict(df_test.loc[df_test.label==l][fit_features].values)
            result_temp = pd.DataFrame(y_pred_test, index=df_test.loc[df_test.label==l].index)
            result_container.loc[result_temp.index] = result_temp
            gt_container = pd.DataFrame(temp.copy(), index=unq_idx)
            y_test = df_test.loc[df_test.label==l][target_features]
            gt_temp = pd.DataFrame(y_test, index=df_test.loc[df_test.label==l].index)
            gt_container.loc[gt_temp.index] = gt_temp
            p_list_pred.append(result_container.copy())
            p_list.append(gt_container.copy())
            all_mapes.append(mape1(y_test, y_pred_test))

        except:
            print(l)
    return p_list_pred,p_list,all_mapes
