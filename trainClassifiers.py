import math

from scipy.stats import randint

from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import RepeatedKFold, GridSearchCV, train_test_split
from sklearn.metrics import mean_absolute_error,root_mean_squared_error,r2_score, balanced_accuracy_score
import pandas as pd
import joblib
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def toBeta(ms):
    return 2**ms/(1+2**ms)

def makeUniqueColumns(df):
    new_columns = []
    seen_columns = {}
    for col in df.columns:
        new_col = col
        count = seen_columns.get(col, 0)
        if count > 0:
            new_col = f"{col}_{count}"
        seen_columns[col] = count + 1
        new_columns.append(new_col)
    return df


def trainSmoking():
    ms = pd.read_csv("H:/My Drive/Data/classifier/ms8k_smoke.csv", encoding='cp1252',index_col=0)
    meta = pd.read_csv("H:/My Drive/Data/classifier/meta.csv", encoding='cp1252',index_col=0)

    #betas = toBeta(ms)
    #ms = pd.concat([betas, betas ** 0.5, betas ** 2], axis=1)
    #ms = makeUniqueColumns(ms)
    #betas2 = pd.concat([betas.iloc[:,0:3333],betas.iloc[:,0:3333]**0.5,betas.iloc[:,0:3333]**2],axis=1)
    #cors = ms.apply(lambda col: np.corrcoef(col, meta['smoke'])[0, 1])
    #ms = ms.iloc[:,np.argsort(abs(cors))[::-1]].iloc[:, 0:10000]

    print(f'Ms: {ms.shape}; meta: {meta.shape}')

    print("Loaded smoking dataset. Starting training...")
    rkf=RepeatedKFold(n_repeats=1,n_splits=5,random_state=123)
    param_grid = {'n_estimators': [20,50,100,200,500,1000,2000],
                  'max_depth': [2,3,4,5,6,7]}
    fit_dict={}
    all_y_pred=np.ones(meta.shape[0])
    for i, (train_index, test_index) in enumerate(rkf.split(ms)):
        print(f"Fold {i}:")
        X_train=ms.iloc[train_index,:]
        X_test = ms.iloc[test_index,:]
        y_train=meta['smoke'].iloc[train_index]
        y_test = meta['smoke'].iloc[test_index]
        for n_estimators in param_grid['n_estimators']:
            for max_depth in param_grid['max_depth']:
                rf = RandomForestClassifier(verbose=False,class_weight='balanced',n_jobs=8,n_estimators=n_estimators,max_depth=max_depth,random_state=123)
                rf.fit(X_train,y_train)
                y_pred=rf.predict(X_test)
                ba = balanced_accuracy_score(y_true=y_test, y_pred=y_pred,adjusted=True)
                print(f'Result for n_estimators:{n_estimators} and max_depth:{max_depth} was {ba:.5f}')
                if (n_estimators,max_depth) in fit_dict:
                    fit_dict[(n_estimators,max_depth)].append(ba)
                else:
                    fit_dict[(n_estimators, max_depth)]=[ba]

    best_val=0
    best_nd=None
    for (n, d), vals in fit_dict.items():
        if np.mean(vals) > best_val:
            best_nd = (n, d)
            best_val=np.mean(vals)

    for i, (train_index, test_index) in enumerate(rkf.split(ms)):
        X_train=ms.iloc[train_index,:]
        X_test = ms.iloc[test_index,:]
        y_train=meta['smoke'].iloc[train_index]
        y_test = meta['smoke'].iloc[test_index]
        rf = RandomForestClassifier(verbose=False, class_weight='balanced', n_jobs=8, n_estimators=best_nd[0],
                                    max_depth=best_nd[1], random_state=123)
        rf.fit(X_train, y_train)
        all_y_pred[test_index.astype(int)] = rf.predict(X_test)
        print(f"Fold {i} done!")

    rf = RandomForestClassifier(verbose=False, class_weight='balanced', n_jobs=8, n_estimators=best_nd[0],
                                max_depth=best_nd[1], random_state=123)
    rf.fit(ms, meta['smoke'])

    # save
    result=pd.DataFrame({'pred':all_y_pred,'smoke':meta['smoke']})
    result.to_csv('H:/My Drive/Data/classifier/smoking_classifier.csv')
    joblib.dump(rf,"H:/My Drive/Data/classifier/smoking_classifier.pkl")

    cm = confusion_matrix(meta['smoke'], all_y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Smoker','Non smoker'])
    fig, ax = plt.subplots(figsize=(8, 8))
    disp.plot(ax=ax)

    # Save the figure
    plt.savefig('H:/My Drive/Data/classifier/SmokingConfusion_matrix.png')
    plt.close()

def trainEthnicity():
    ms = pd.read_csv("H:/My Drive/Data/classifier/ms8k_ethnicity.csv", encoding='cp1252',index_col=0)
    meta = pd.read_csv("H:/My Drive/Data/classifier/meta.csv", encoding='cp1252',index_col=0)
    meta['ethnicity']=meta.iloc[:,45]
    meta['ethnicity'].replace({'Native American or Alaskan Native':'Other','Black or African American':'Other','A race/ethnicity not listed here':'Other','Multiracial or Biracial':np.nan},inplace=True)
    meta=meta.reset_index(drop=True)
    good=meta[meta['ethnicity'].notna()].index.tolist()
    meta=meta.iloc[good,:]
    ms=ms.iloc[good,:]
    #betas = toBeta(ms)
    #ms = pd.concat([betas, betas ** 0.5, betas ** 2], axis=1)
    #ms = makeUniqueColumns(ms)
    #betas2 = pd.concat([betas.iloc[:,0:3333],betas.iloc[:,0:3333]**0.5,betas.iloc[:,0:3333]**2],axis=1)
    #cors = ms.apply(lambda col: np.corrcoef(col, meta['smoke'])[0, 1])
    #ms = ms.iloc[:,np.argsort(abs(cors))[::-1]].iloc[:, 0:10000]

    print(f'Ms: {ms.shape}; meta: {meta.shape}')

    print("Loaded ethnicity dataset. Starting training...")
    rkf=RepeatedKFold(n_repeats=1,n_splits=5,random_state=123)
    param_grid = {'n_estimators': [20,50,100,200,500,1000,2000],
                  'max_depth': [2,3,4,5,6,7]}
    fit_dict={}
    all_y_pred=pd.Series(['']*ms.shape[0])
    for i, (train_index, test_index) in enumerate(rkf.split(ms)):
        print(f"Fold {i}:")
        X_train=ms.iloc[train_index,:]
        X_test = ms.iloc[test_index,:]
        y_train=meta['ethnicity'].iloc[train_index]
        y_test = meta['ethnicity'].iloc[test_index]
        for n_estimators in param_grid['n_estimators']:
            for max_depth in param_grid['max_depth']:
                rf = RandomForestClassifier(verbose=False,class_weight='balanced',n_jobs=8,n_estimators=n_estimators,max_depth=max_depth,random_state=123)
                rf.fit(X_train,y_train)
                y_pred=rf.predict(X_test)
                ba = balanced_accuracy_score(y_true=y_test, y_pred=y_pred,adjusted=True)
                print(f'Result for n_estimators:{n_estimators} and max_depth:{max_depth} was {ba:.5f}')
                if (n_estimators,max_depth) in fit_dict:
                    fit_dict[(n_estimators,max_depth)].append(ba)
                else:
                    fit_dict[(n_estimators, max_depth)]=[ba]

    best_val=0
    best_nd=None
    for (n, d), vals in fit_dict.items():
        if np.mean(vals) > best_val:
            best_nd = (n, d)
            best_val=np.mean(vals)

    for i, (train_index, test_index) in enumerate(rkf.split(ms)):
        X_train=ms.iloc[train_index,:]
        X_test = ms.iloc[test_index,:]
        y_train=meta['ethnicity'].iloc[train_index]
        y_test = meta['ethnicity'].iloc[test_index]
        rf = RandomForestClassifier(verbose=False, class_weight='balanced', n_jobs=8, n_estimators=best_nd[0],
                                    max_depth=best_nd[1], random_state=123)
        rf.fit(X_train, y_train)
        all_y_pred[test_index] = rf.predict(X_test)
        print(f"Fold {i} done!")

    rf = RandomForestClassifier(verbose=False, class_weight='balanced', n_jobs=8, n_estimators=best_nd[0],
                                max_depth=best_nd[1], random_state=123)
    rf.fit(ms, meta['ethnicity'])

    # save
    result=pd.DataFrame({'pred':all_y_pred,'ethnicity':meta['ethnicity']})
    result.to_csv('H:/My Drive/Data/classifier/ethnicity_classifier.csv')
    joblib.dump(rf,"H:/My Drive/Data/classifier/ethnicity_classifier.pkl")

    cm = confusion_matrix(meta['ethnicity'], all_y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Asian/Pacific Islander','Hispanic/Latino','Middle Easter/North African','Other','White/Caucasian'])
    fig, ax = plt.subplots(figsize=(16, 16))
    disp.plot(ax=ax)

    # Save the figure
    plt.savefig('H:/My Drive/Data/classifier/EthnicityConfusion_matrix.png')

    plt.close()





if __name__ == "__main__":
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    trainEthnicity()
    trainSmoking()
