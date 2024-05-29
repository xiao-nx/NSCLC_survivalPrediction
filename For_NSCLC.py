#!/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2/26/2024 12:13 AM
# @Author : Yi Chen
# @File :For_NSCLC.py

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler, StandardScaler

def load_patient_list(excel):
    df = pd.read_excel(excel)
    patient_list = [i[4:-4] for i in df['NO']]
    return patient_list

def map_rename(df, suffix):
    columns_to_suffix = df.columns[1:]
    rename_mapping = {col: col + suffix for col in columns_to_suffix if col in df.columns[1:]}
    df.rename(columns=rename_mapping, inplace=True)
    return df

def plot_nomogram(coef_df):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown']
    baseline = 0  # 你可以根据需要设置基线
    for idx, row in coef_df.iterrows():
        points = [baseline, baseline + row['Scores']]
        ax.plot(points, [idx, idx], marker='o', color=colors[idx % len(colors)])
        ax.text(points[1], idx, f"{row['Features']} ({row['Scores']:.2f} points)",
                va='center', ha='right' if row['Scores'] < 0 else 'left')

    ax.set_yticks([])
    ax.set_title('Nomogram for Survival Prediction')
    ax.set_xlabel('Points')
    plt.gca().invert_yaxis()  # 使得分最高的特征在顶部
    plt.show()

# -------------------------------------Feature selection stage---------------------------------------------
def remove_high_corr_no_variance(df):
    df_ori = df.copy()
    df = df[df.columns[1:]]
    select_feature_list = []
    var_features = df.var().sort_values()
    df = df[var_features[var_features > 0.01].index]
    for var in df.columns:
        sigle_rate = (df[var].value_counts().max() / df.shape[0])
        if sigle_rate < 0.9:
            select_feature_list.append(var)
    df = df[select_feature_list]

    cor = df.corr('spearman').abs()
    upper_tri = cor.where(np.triu(np.ones(cor.shape), k=1).astype(bool))
    to_drop = []
    for column in upper_tri.columns:
        for row in upper_tri.columns:
            if upper_tri[column][row] > 0.85:
                if np.sum(upper_tri[column]) > np.sum(
                        upper_tri[row]):
                    to_drop.append(column)
                else:
                    to_drop.append(row)
    to_drop = np.unique(to_drop)
    df_ori.drop(to_drop, axis=1, inplace=True)
    constant_columns = [col for col in df_ori.columns if df_ori[col].nunique() == 1]
    df_ori.drop(columns=constant_columns, inplace=True)
    return df_ori

#-------------------------------------------Lasso feature selection----------------------------------
def Lasso(df, outcome):
    from sklearn.feature_selection import RFE
    from sklearn.linear_model import LassoCV
    from sklearn.preprocessing import StandardScaler

    y = outcome['statusipsiKLINISCH']
    scale = StandardScaler()
    feature = scale.fit_transform(df.iloc[:, 1:])
    lasso = LassoCV(cv=5, random_state=42, max_iter=10000)
    lasso.fit(feature, y)
    lasso_coefs = lasso.coef_

    featuer_name = df.columns[1:]
    important_features = featuer_name[lasso_coefs != 0]

    print("Important features selected by LASSO:")
    print(important_features)
    return important_features


# ---------------------------------------model selected features-------------------------------------

def fold_Cox_selected(df, outcome):
    from sksurv.linear_model import CoxnetSurvivalAnalysis
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    from collections import Counter

    feature_candidate = []
    outcome_input = list(zip(outcome['PFS_boolean'].astype(bool), outcome['PFS_time']))
    outcome_structured = np.array(outcome_input, dtype=[('Event', 'bool'), ('Time', 'float')])
    feature = df.iloc[:, 1:]

    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for train_idx, test_idx in kf.split(feature, outcome['PFS_boolean'].astype(bool)):
        scale = StandardScaler()
        feature_train = scale.fit_transform(feature.iloc[train_idx])
        outcome_train = outcome_structured[train_idx]

        model = CoxnetSurvivalAnalysis(l1_ratio=1.0, alpha_min_ratio=0.1, n_alphas=100)
        model.fit(feature_train, outcome_train)
        coefficients = abs(np.sum(model.coef_, axis=1))
        sorted_indices = np.argsort(coefficients)[::-1]
        coefficients_new = coefficients[sorted_indices]
        indict = np.argmax(np.diff(coefficients_new))
        select_featuer = feature.columns[sorted_indices[:indict]]
        feature_candidate.extend(select_featuer)
    count = Counter(feature_candidate)
    keys_with_count_five = [key for key, value in count.items() if value >= 4]
    print(f"Common top features across all folds: {keys_with_count_five}")
    return keys_with_count_five


def test_predict_calculate_cox(test_feature, outcome_test, train_feature, outcome_train, columns):
    # import shap
    from sksurv.linear_model import CoxnetSurvivalAnalysis
    from sksurv.metrics import concordance_index_censored


    outcome_train = list(zip(outcome_train['PFS_boolean'].astype(bool), outcome_train['PFS_time']))
    outcome_structured_train = np.array(outcome_train, dtype=[('Event', 'bool'), ('Time', 'float')])

    rsf = CoxnetSurvivalAnalysis(l1_ratio=1.0, alpha_min_ratio=0.1, n_alphas=100)
    rsf.fit(train_feature, outcome_structured_train)

    print('-----------------------------CoxnetSurvivalAnalysis Feature importance--------------------------------------')
    coefficients = rsf.coef_
    contribute = coefficients.mean(axis=1)
    for i in range(len(contribute)):
        print(f'{columns[i]}: {contribute[i]}')

    c_index = concordance_index_censored(outcome_test['PFS_boolean'].astype(bool),
                                         outcome_test['PFS_time'],
                                         rsf.predict(test_feature))
    coef_df = pd.DataFrame({
        'Features': columns,
        'Coefficients': contribute
    })
    max_score = 100  # 假设最高分为 100 分
    coef_df['Scores'] = coef_df['Coefficients'] * max_score / np.abs(coef_df['Coefficients']).max()
    print(
        '-----------------------------Test part CoxnetSurvivalAnalysis C-index: --------------------------------------')
    print(f'cox model c_index:{c_index[0]}')
    return coef_df, c_index[0]

#---------------------------------------random forest model feature selection---------------------------------------
def fold_rsf_selected(df, outcome):
    from sklearn.model_selection import StratifiedKFold
    from sksurv.ensemble import RandomSurvivalForest
    from sklearn.inspection import permutation_importance
    from collections import Counter

    feature_candidate = []
    outcome_input = list(zip(outcome['PFS_boolean'].astype(bool), outcome['PFS_time']))
    outcome_structured = np.array(outcome_input, dtype=[('Event', 'bool'), ('Time', 'float')])
    feature = df.iloc[:, 1:]

    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for train_idx, test_idx in kf.split(feature, outcome['PFS_boolean'].astype(bool)):
        scale = StandardScaler()
        feature_train = scale.fit_transform(feature.iloc[train_idx])
        outcome_train = outcome_structured[train_idx]

        rsf = RandomSurvivalForest()
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [5, 10, 15],
            'min_samples_split': [2, 4, 6, 8, 10],
            'max_features': ['sqrt', 'log2']
        }
        skf = StratifiedKFold(n_splits=3)
        grid = GridSearchCV(estimator=rsf, param_grid=param_grid, cv=skf)
        grid.fit(feature_train, outcome_train)
        best_params = grid.best_params_
        best_model = RandomSurvivalForest(**best_params)
        best_model.fit(feature_train, outcome_train)
        perm_importance = permutation_importance(best_model, feature_train, outcome_train, n_repeats=30, random_state=42)
        feature_importances = perm_importance.importances_mean
        sorted_indices = np.argsort(feature_importances)[::-1]
        coefficients_new = feature_importances[sorted_indices]
        indict = np.argmax(np.diff(coefficients_new))
        select_featuer = feature.columns[sorted_indices[:indict]]
        feature_candidate.extend(select_featuer)
    count = Counter(feature_candidate)
    keys_with_count_five = [key for key, value in count.items() if value >= 3]
    print(f"Common top features across all folds: {keys_with_count_five}")
    return keys_with_count_five

def test_predict_calculate_rsf(test_feature, outcome_test, train_feature, outcome_train, columns):
    from sksurv.ensemble import RandomSurvivalForest
    from sksurv.metrics import concordance_index_censored
    from sklearn.inspection import permutation_importance

    outcome_train = list(zip(outcome_train['PFS_boolean'].astype(bool), outcome_train['PFS_time']))
    outcome_structured_train = np.array(outcome_train, dtype=[('Event', 'bool'), ('Time', 'float')])

    rsf = RandomSurvivalForest()
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [5, 10, 15],
        'min_samples_split': [2, 10],
        'max_features': ['sqrt', 'log2']
    }

    grid = GridSearchCV(estimator=rsf, param_grid=param_grid, cv=5)
    grid.fit(train_feature, outcome_structured_train)
    print("Best parameters:", grid.best_params_)
    print("Best cross-validation score: {:.3f}".format(grid.best_score_))
    print(
        '-----------------------------Random forest Feature importance--------------------------------------')
    best_model = grid.best_estimator_
    perm_importance = permutation_importance(best_model, train_feature, outcome_structured_train, n_repeats=30, random_state=42)
    feature_importances = perm_importance.importances_mean
    for i in range(len(columns)):
        print(f'{columns[i]}: {feature_importances[i]}')
    coef_df = pd.DataFrame({
        'Features': columns,
        'Coefficients': feature_importances
    })
    max_score = 100  # 假设最高分为 100 分
    coef_df['Scores'] = coef_df['Coefficients'] * max_score / np.abs(coef_df['Coefficients']).max()
    print(
        '-----------------------------RandomSurvivalForest C-index--------------------------------------')

    c_index = concordance_index_censored(outcome_test['PFS_boolean'].astype(bool),
                                         outcome_test['PFS_time'],
                                         best_model.predict(test_feature))
    print(f'Random forest model -c_index:{c_index[0]}')
    return coef_df, c_index[0]


#---------------------------------------Gridient Boost feature selection---------------------------------------
def fold_GBS_feature_selected(df, outcome):
    from sklearn.model_selection import StratifiedKFold
    from sksurv.ensemble import ComponentwiseGradientBoostingSurvivalAnalysis
    from sksurv.ensemble import GradientBoostingSurvivalAnalysis
    from sklearn.inspection import permutation_importance
    from collections import Counter

    est_cph_tree = GradientBoostingSurvivalAnalysis(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)

    feature = df.iloc[:, 1:]
    outcome_input = list(zip(outcome['PFS_boolean'].astype(bool), outcome['PFS_time']))
    outcome_structured = np.array(outcome_input, dtype=[('Event', 'bool'), ('Time', 'float')])
    skf = StratifiedKFold(n_splits=5)
    feature_candidate = []

    for train_index, test_index in skf.split(feature, outcome['PFS_boolean'].astype(bool)):
        X_train, X_test = feature.iloc[train_index], feature.iloc[test_index]
        y_train, y_test = outcome_structured[train_index], outcome_structured[test_index]

        # Build and train the model
        est_cph_tree = GradientBoostingSurvivalAnalysis(n_estimators=100, learning_rate=1.0, max_depth=1,
                                                        random_state=0)
        est_cph_tree.fit(X_train, y_train)
        # perm_importance = permutation_importance(est_cph_tree, X_train, y_train, n_repeats=30, random_state=42)
        feature_importances = est_cph_tree.feature_importances_
        sorted_indices = np.argsort(feature_importances)[::-1]

        # top_ten_indices = sorted_indices[:10]
        coefficients_new = feature_importances[sorted_indices]
        indict = np.argmax(np.diff(coefficients_new))
        select_featuer = feature.columns[sorted_indices[:indict]]
        feature_candidate.extend(select_featuer)

    count = Counter(feature_candidate)
    keys_with_count_five = [key for key, value in count.items() if value >= 4]
    print(f"Common top features across all folds: {keys_with_count_five}")
    return keys_with_count_five


def test_predict_calculate_GBS(test_feature, outcome_test, train_feature, outcome_train, columns):
    from sksurv.ensemble import GradientBoostingSurvivalAnalysis
    from sksurv.metrics import concordance_index_censored
    from sklearn.inspection import permutation_importance

    outcome_train = list(zip(outcome_train['PFS_boolean'].astype(bool), outcome_train['PFS_time']))
    outcome_structured_train = np.array(outcome_train, dtype=[('Event', 'bool'), ('Time', 'float')])

    outcome_test_struct = list(zip(outcome_test['PFS_boolean'].astype(bool), outcome_test['PFS_time']))
    outcome_structured_test = np.array(outcome_test_struct, dtype=[('Event', 'bool'), ('Time', 'float')])

    rsf = GradientBoostingSurvivalAnalysis(random_state=1234)
    param_grid={
        'n_estimators': [100, 200],
        'max_depth': [5, 10, 15],
        'learning_rate': [1e-4, 1e-2, 1.0]
    }

    grid = GridSearchCV(estimator=rsf, param_grid=param_grid, cv=5)
    grid.fit(train_feature, outcome_structured_train)
    print("Best parameters:", grid.best_params_)
    print("Best cross-validation score: {:.3f}".format(grid.best_score_))

    print(
            '-----------------------------GradientBoostingSurvivalAnalysis Feature importance--------------------------------------')
    best_model = grid.best_estimator_
    perm_importance = permutation_importance(best_model, train_feature, outcome_structured_train,
                                             n_repeats=30, random_state=42)
    feature_importances = perm_importance.importances_mean
    for i in range(len(columns)):
        print(f'{columns[i]}: {feature_importances[i]}')
    print(
        '-----------------------------RandomSurvivalForest C-index--------------------------------------')


    c_index = concordance_index_censored(outcome_test['PFS_boolean'].astype(bool),
                                         outcome_test['PFS_time'],
                                         best_model.predict(test_feature))

    print(f'GBS model-- c_index:{c_index[0]}')


#-------------------------------------------Main function-------------------------------------------------------
def Load_radiomics(patient_list):
    from sklearn.model_selection import train_test_split
    outcome_file = pd.read_excel('G:/Yi/PRSKI/Ning_project/data/data_processed.xlsx')
    outcome = outcome_file[['NO', 'PFS_boolean', 'PFS_time']]
    outcome = outcome.dropna()
    outcome['NO'] = outcome['NO'].astype('uint8').astype(str)

    # Name censor PFS_time
    outcome = outcome[outcome['NO'].isin(patient_list)]
    # outcome.loc[outcome['OS'] == 2, 'OS'] = 0
    outcome = outcome.drop_duplicates(subset=['NO'], keep='first')


    radiomics_Feature = pd.read_excel('G:/Yi/PRSKI/Ning_project/data/lung_cancer_1.xlsx').rename(columns={'RowName': 'NO'})
    radiomics_Feature.drop(radiomics_Feature.columns[1:23], axis=1, inplace=True)
    radiomics_Feature['NO'] = [i.split('.')[0].split('j')[-1] for i in radiomics_Feature['NO']]
    radiomics_Feature.drop(radiomics_Feature.columns[1:23], axis=1, inplace=True)

    patient_list_final = list(set(radiomics_Feature['NO']).intersection(set(outcome['NO'])))

    train_patient, test_patient = train_test_split(patient_list_final, test_size=0.2, random_state=0)

    # test_patient = [file for file in patient_list if 'AMC' in file]
    # train_patient = list(set(patient_list).difference(set(test_patient)))

    Img_train = radiomics_Feature[radiomics_Feature['NO'].isin(train_patient)].reset_index(drop=True)
    outcome_train = outcome[outcome['NO'].isin(train_patient)].reset_index(drop=True)

    Img_test = radiomics_Feature[radiomics_Feature['NO'].isin(test_patient)].reset_index(drop=True)
    outcome_test = outcome[outcome['NO'].isin(test_patient)].reset_index(drop=True)


    remove_nonvari_all_sequence_train = remove_high_corr_no_variance(Img_train)
    remove_nonvari_all_sequence_test = Img_test[remove_nonvari_all_sequence_train.columns]

    scale = StandardScaler()
    select_feature_train = scale.fit_transform(remove_nonvari_all_sequence_train)
    select_feature_test = scale.transform(remove_nonvari_all_sequence_test)
    Laaso_Feature = Lasso(select_feature_train, outcome_train)


    # select_feature_by_Lasso = fold_Cox_selected(remove_nonvari_all_sequence_train, outcome_train)
    select_feature_by_Lasso = Laaso_Feature
    scale = StandardScaler()
    select_feature_train = scale.fit_transform(remove_nonvari_all_sequence_train[select_feature_by_Lasso])
    select_feature_test = scale.transform(remove_nonvari_all_sequence_test[select_feature_by_Lasso])
    coe, C_index = test_predict_calculate_cox(select_feature_test, outcome_test, select_feature_train, outcome_train, select_feature_by_Lasso)
    # plot_nomogram(coe)


    select_feature_by_rsf = fold_rsf_selected(remove_nonvari_all_sequence_train, outcome_train)
    if len(select_feature_by_rsf)>0:
        scale = StandardScaler()
        select_feature_train = scale.fit_transform(remove_nonvari_all_sequence_train[select_feature_by_rsf])
        select_feature_test = scale.transform(remove_nonvari_all_sequence_test[select_feature_by_rsf])
        test_predict_calculate_rsf(select_feature_test, outcome_test, select_feature_train, outcome_train,
                                   select_feature_by_rsf)

    select_feature_by_GBS = fold_GBS_feature_selected(remove_nonvari_all_sequence_train, outcome_train)
    if len(select_feature_by_GBS)>0:
        scale = StandardScaler()
        select_feature_train = scale.fit_transform(remove_nonvari_all_sequence_train[select_feature_by_GBS])
        select_feature_test = scale.transform(remove_nonvari_all_sequence_test[select_feature_by_GBS])
        test_predict_calculate_GBS(select_feature_test, outcome_test, select_feature_train, outcome_train,
                                   select_feature_by_GBS)




if __name__ == '__main__':
    # Patient_ID = os.listdir('./Imge_path')
    Patient_ID = load_patient_list('G:/Yi/PRSKI/Ning_project/data/lung_cancer_1.xlsx')
    Load_radiomics(patient_list=Patient_ID)
