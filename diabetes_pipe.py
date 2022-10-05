import joblib
import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.25)
    quartile3 = dataframe[variable].quantile(0.75)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

def grab_outliers(dataframe, col_name, index=False):
    low, up = outlier_thresholds(dataframe, col_name)

    if dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].shape[0] > 10:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].head())
    else:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))])

    if index:
        outlier_index = dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].index
        return outlier_index

def diabetes_data_prep(dataframe):
    # HANDLING MISSING VALUES

    # Glucose
    dataframe.loc[(dataframe["Glucose"].isnull()) & (dataframe["Outcome"] == 1), "Glucose"] = dataframe.loc[
        dataframe["Outcome"] == 1, "Glucose"].mean()
    dataframe.loc[(dataframe["Glucose"].isnull()) & (dataframe["Outcome"] == 0), "Glucose"] = dataframe.loc[
        dataframe["Outcome"] == 0, "Glucose"].mean()

    # Blood Pressure
    outlier_ind_BloodPressure = grab_outliers(dataframe, "BloodPressure", True)

    dataframe.loc[(dataframe["BloodPressure"].isnull()) & (dataframe["Outcome"] == 1), "BloodPressure"] = \
        dataframe.loc[
            (~dataframe.index.isin(outlier_ind_BloodPressure)) & (dataframe["Outcome"] == 1), "BloodPressure"].mean()
    dataframe.loc[(dataframe["BloodPressure"].isnull()) & (dataframe["Outcome"] == 0), "BloodPressure"] = \
        dataframe.loc[
            (~dataframe.index.isin(outlier_ind_BloodPressure)) & (dataframe["Outcome"] == 0), "BloodPressure"].mean()

    # SkinThickness
    outlier_ind_SkinThickness = grab_outliers(dataframe, "SkinThickness", True)

    dataframe.loc[(dataframe["SkinThickness"].isnull()) & (dataframe["Outcome"] == 1), "SkinThickness"] = \
        dataframe.loc[
            (~dataframe.index.isin(outlier_ind_SkinThickness)) & (dataframe["Outcome"] == 1), "SkinThickness"].mean()
    dataframe.loc[(dataframe["SkinThickness"].isnull()) & (dataframe["Outcome"] == 0), "SkinThickness"] = \
        dataframe.loc[
            (~dataframe.index.isin(outlier_ind_SkinThickness)) & (dataframe["Outcome"] == 0), "SkinThickness"].mean()

    # Insulin
    outlier_ind_Insulin = grab_outliers(dataframe, "Insulin", True)

    dataframe.loc[(dataframe["Insulin"].isnull()) & (dataframe["Outcome"] == 1), "Insulin"] = \
        dataframe.loc[(~dataframe.index.isin(outlier_ind_Insulin)) & (dataframe["Outcome"] == 1), "Insulin"].mean()
    dataframe.loc[(dataframe["Insulin"].isnull()) & (dataframe["Outcome"] == 0), "Insulin"] = \
        dataframe.loc[(~dataframe.index.isin(outlier_ind_Insulin)) & (dataframe["Outcome"] == 0), "Insulin"].mean()

    # BMI
    outlier_ind_BMI = grab_outliers(dataframe, "BMI", True)

    dataframe.loc[(dataframe["BMI"].isnull()) & (dataframe["Outcome"] == 1), "BMI"] = \
        dataframe.loc[(~dataframe.index.isin(outlier_ind_BMI)) & (dataframe["Outcome"] == 1), "BMI"].mean()
    dataframe.loc[(dataframe["BMI"].isnull()) & (dataframe["Outcome"] == 0), "BMI"] = \
        dataframe.loc[(~dataframe.index.isin(outlier_ind_BMI)) & (dataframe["Outcome"] == 0), "BMI"].mean()

    ## Normalization & Scaling

    dataframe.loc[:, "Pregnancies"] = np.log1p(dataframe["Pregnancies"])
    dataframe.loc[:, "BMI"] = np.log1p(dataframe["BMI"])
    dataframe.loc[:, "Age"] = np.log1p(dataframe["Age"])

    X = dataframe.drop("Outcome", axis=1)
    y= dataframe["Outcome"]

    return X, y

# Base Models
def base_models(X, y, scoring="roc_auc"):
    print("Base Models....")
    classifiers = [('LR', LogisticRegression(max_iter=1000, random_state=26)),
                   ("CART", DecisionTreeClassifier(random_state=26)),
                   ("RF", RandomForestClassifier(random_state=26)),
                   ('Adaboost', AdaBoostClassifier(random_state=26)),
                   ('GBM', GradientBoostingClassifier(random_state=26)),
                   ('XGBoost', XGBClassifier(random_state=26, use_label_encoder=False, eval_metric='logloss')),
                   ('LightGBM', LGBMClassifier(random_state=26)),
                   # ('CatBoost', CatBoostClassifier(verbose=False))
                   ]

    for name, classifier in classifiers:
        cv_results = cross_validate(classifier, X, y, cv=5, scoring=scoring)
        print(f"{scoring}: {round(cv_results['test_score'].mean(), 4)} ({name}) ")


cart_params = {'max_depth': range(1, 20),
               "min_samples_split": range(2, 30)}

rf_params = {"max_depth": [8, 15, None],
             "max_features": [5, 7, "auto"],
             "min_samples_split": [15, 20],
             "n_estimators": [200, 300]}

xgboost_params = {"learning_rate": [0.1, 0.01],
                  "max_depth": [5, 8],
                  "n_estimators": [100, 200],
                  "colsample_bytree": [0.5, 1]}

lightgbm_params = {"learning_rate": [0.01, 0.1],
                   "n_estimators": [300, 500],
                   "colsample_bytree": [0.7, 1]}

classifiers = [("CART", DecisionTreeClassifier(random_state=26), cart_params),
               ("RF", RandomForestClassifier(random_state=26), rf_params),
               ('XGBoost', XGBClassifier(random_state=26, use_label_encoder=False, eval_metric="logloss"), xgboost_params),
               ('LightGBM', LGBMClassifier(random_state=26), lightgbm_params)]

def hyperparameter_optimization(X, y, cv=3, scoring="roc_auc"):
    print("Hyperparameter Optimization....")
    best_models = {}
    for name, classifier, params in classifiers:
        print(f"########## {name} ##########")
        cv_results = cross_validate(classifier, X, y, cv=cv, scoring=scoring)
        print(f"{scoring} (Before): {round(cv_results['test_score'].mean(), 4)}")

        gs_best = GridSearchCV(classifier, params, cv=cv, n_jobs=-1, verbose=False).fit(X, y)
        final_model = classifier.set_params(**gs_best.best_params_)

        cv_results = cross_validate(final_model, X, y, cv=cv, scoring=scoring)
        print(f"{scoring} (After): {round(cv_results['test_score'].mean(), 4)}")
        print(f"{name} best params: {gs_best.best_params_}", end="\n\n")
        best_models[name] = final_model
    return best_models

def voting_classifier(best_models, X, y):
    print("Voting Classifier...")
    voting_clf = VotingClassifier(estimators=[('RF', best_models["RF"]),
                                              ('LightGBM', best_models["LightGBM"])],
                                  voting='soft').fit(X, y)
    cv_results = cross_validate(voting_clf, X, y, cv=5, scoring=["accuracy", "f1"])
    print(f"Accuracy: {cv_results['test_accuracy'].mean()}")
    print(f"F1Score: {cv_results['test_f1'].mean()}")
    return voting_clf

################################################
# Pipeline Main Function
################################################

def main():
    df = pd.read_csv(r"diabetes/diabetes.csv")
    X, y = diabetes_data_prep(df)
    base_models(X, y)
    best_models = hyperparameter_optimization(X, y)
    voting_clf = voting_classifier(best_models, X, y)
    joblib.dump(voting_clf, "voting_clf.pkl")
    return voting_clf

if __name__ == "__main__":
    print("Process started")
    main()
