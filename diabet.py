###################
# LIBRARIES
###################

import pandas as pd
import numpy as np
import seaborn as sns
import missingno as msno
from matplotlib import pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 170)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

###################
# FUNCTIONS
###################

def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.describe([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

def cat_summary(dataframe, col_name, plot=False):

    if dataframe[col_name].dtypes == "bool":
        dataframe[col_name] = dataframe[col_name].astype(int)

        print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                            "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
        print("##########################################")

        if plot:
            sns.countplot(x=dataframe[col_name], data=dataframe)
            plt.show(block=True)
    else:
        print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                            "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
        print("##########################################")

        if plot:
            sns.countplot(x=dataframe[col_name], data=dataframe)
            plt.show(block=True)

def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist()
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)

def grab_col_names(dataframe, cat_th=10,  car_th=20):
    """
    Veri setindeki kategorik, numerik ve kategorik fakat kardinal de??i??kenlerin isimlerini verir.

    Parameters
    ----------
    dataframe: dataframe
        de??i??ken isimleri al??nmak istenen dataframe'dir.
    cat_th: int, float
        numerik fakat kategorik olan de??i??kenler i??in s??n??f e??ik de??eri
    car_th: int, float
        kategorik fakat kardinal de??i??kenler i??in s??n??f e??ik de??eri

    Returns
    -------
    cat_cols: list
        Kategorik de??i??ken listesi
    num_cols: list
        Numerik de??i??ken listesi
    cat_but_car: list
        Kategorik g??r??n??ml?? kardinal de??i??ken listesi

    Notes
    ------
    cat_cols + num_cols + cat_but_car = toplam de??i??ken say??s??
    num_but_cat cat_cols'un i??erisinde.

    """
    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if str(dataframe[col].dtypes) in ["category", "object", "bool"]]

    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < 10 and dataframe[col].dtypes in ["int", "float"]]

    cat_but_car = [col for col in df.columns if
                   dataframe[col].nunique() > 20 and str(dataframe[col].dtypes) in ["category", "object"]]

    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes in ["int", "float"]]
    num_cols = [col for col in num_cols if col not in cat_cols]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')

    return cat_cols, num_cols, cat_but_car

def target_summary_with_cat(dataframe, target, categorical_col):
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean()}), end="\n\n\n")

def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")

def high_correlated_cols(dataframe, plot=False, corr_th=0.90):
    corr = dataframe.corr()
    cor_matrix = corr.abs()
    upper_triangle_matrix = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(bool))
    drop_list = [col for col in upper_triangle_matrix.columns if any(upper_triangle_matrix[col] > corr_th)]
    if plot:
        import seaborn as sns
        import matplotlib.pyplot as plt
        sns.set(rc={'figure.figsize': (15, 15)})
        sns.heatmap(corr, cmap="RdBu")
        plt.show()
    return drop_list

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

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = round(low_limit, 0)
    dataframe.loc[(dataframe[variable] > up_limit), variable] = round(up_limit, 0)

def remove_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    df_without_outliers = dataframe[~((dataframe[col_name] < low_limit) | (dataframe[col_name] > up_limit))]
    return df_without_outliers

def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")
    if na_name:
        return na_columns

def missing_vs_target(dataframe, target, na_columns):
    temp_df = dataframe.copy()
    for col in na_columns:
        temp_df[col + '_NA_FLAG'] = np.where(temp_df[col].isnull(), 1, 0)
    na_flags = temp_df.loc[:, temp_df.columns.str.contains("_NA_")].columns
    for col in na_flags:
        print(pd.DataFrame({"TARGET_MEAN": temp_df.groupby(col)[target].mean(),
                            "Count": temp_df.groupby(col)[target].count()}), end="\n\n\n")

def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

def rare_analyser(dataframe, target, cat_cols):
    for col in cat_cols:
        print(col, ":", len(dataframe[col].value_counts()))
        print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
                            "RATIO": dataframe[col].value_counts() / len(dataframe),
                            "TARGET_MEAN": dataframe.groupby(col)[target].mean()}), end="\n\n\n")

def rare_encoder(dataframe, rare_perc):
    temp_df = dataframe.copy()

    rare_columns = [col for col in temp_df.columns if temp_df[col].dtypes == 'O'
                    and (temp_df[col].value_counts() / len(temp_df) < rare_perc).any(axis=None)]

    for var in rare_columns:
        tmp = temp_df[var].value_counts() / len(temp_df)
        rare_labels = tmp[tmp < rare_perc].index
        temp_df[var] = np.where(temp_df[var].isin(rare_labels), 'Rare', temp_df[var])

    return temp_df

def check_skew(df_skew, column):
    skew = stats.skew(df_skew[column])
    skewtest = stats.skewtest(df_skew[column])
    plt.title('Distribution of ' + column)
    sns.distplot(df_skew[column], color="g")
    print("{}'s: Skew: {}, : {}".format(column, skew, skewtest))
    return

diabetes = pd.read_csv("diabetes/diabetes.csv")
df = diabetes.copy()

df.isnull().any() # it returns false
check_df(df) # however, NA values are coded as 0.

num_cols = [col for col in df.columns if df[col].dtypes != "object" and col != "Outcome"]

# "Pregnancies" can take the value 0 but others cannot

for col in num_cols:
    if col == "Pregnancies":
        continue
    else:
        df.loc[df[col] == 0, col] = np.nan


from sklearn.model_selection import train_test_split

# Since I'm thinking about filling missing values, to avoid data leakage I split the data.
# If it needs more elaboration, taking means and medians with full data resulting in information in test data
# being included in preprocessing part. This may lead to high performance in both training and test data
# but in production your model may perform poorly.

train, test = train_test_split(df, test_size=0.2, stratify=df.Outcome, random_state=26)

nan_cols = missing_values_table(train, True)

msno.matrix(train)
plt.show()

for col in nan_cols:
    fig = plt.figure(figsize=(8,6))
    g = sns.kdeplot(x=train[col], shade=True, hue=train["Outcome"])
    g.set_title("Col name: " + str(col))
    g.xaxis.set_minor_locator(AutoMinorLocator(5))
    g.yaxis.set_minor_locator(AutoMinorLocator(5))
    g.tick_params(which="both", width=2)
    g.tick_params(which="major", length=7)
    g.tick_params(which="minor", length=4)
    plt.show()

for col in nan_cols:
    fig = plt.figure(figsize=(8,6))
    g = sns.kdeplot(x=train[col], shade=True, color="green")
    g.set_title("Col name: " + str(col))
    g.xaxis.set_minor_locator(AutoMinorLocator(5))
    g.yaxis.set_minor_locator(AutoMinorLocator(5))
    g.tick_params(which="both", width=2)
    g.tick_params(which="major", length=7)
    g.tick_params(which="minor", length=4)
    plt.show()

for col in nan_cols:
    fig = plt.figure(figsize=(8,6))
    g = sns.distplot(x=train[col], kde=False, color="purple", hist_kws=dict(edgecolor="black", linewidth=2))
    g.set_title("Col name: " + str(col))
    g.xaxis.set_minor_locator(AutoMinorLocator(5))
    g.yaxis.set_minor_locator(AutoMinorLocator(5))
    g.tick_params(which="both", width=2)
    g.tick_params(which="major", length=7)
    g.tick_params(which="minor", length=4)
    plt.show()

for col in [col for col in train.columns if col != "Outcome"]:
    target_summary_with_num(train, "Outcome", col)

# HANDLING MISSING VALUES

# Glucose
# There is no extreme skew, so I just go with conditional means to fill.

train.loc[(train["Glucose"].isnull()) & (train["Outcome"] == 1), "Glucose"] = train.loc[train["Outcome"] == 1, "Glucose"].mean()
train.loc[(train["Glucose"].isnull()) & (train["Outcome"] == 0), "Glucose"] = train.loc[train["Outcome"] == 0, "Glucose"].mean()

# Blood Pressure
fig = plt.figure(figsize=(8,6))
g = sns.boxplot(x=train["BloodPressure"], palette="rainbow")
g.yaxis.set_minor_locator(AutoMinorLocator())
g.tick_params(which="both", width=2)
g.tick_params(which="major", length=7)
g.tick_params(which="minor", length=4)
plt.show()

for j in train["Outcome"].unique():
    plt.figure(figsize=(8,6))
    g = sns.boxplot(x=train.loc[train["Outcome"] == j, "BloodPressure"], palette="rainbow")
    g.yaxis.set_minor_locator(AutoMinorLocator())
    g.set_title("Outcome: " + str(j))
    g.tick_params(which="both", width=2)
    g.tick_params(which="major", length=7)
    g.tick_params(which="minor", length=4)
    plt.show()

# There are some outliers, so I take that into account then go for the mean anyway since the distribution
# resembles normal dist a lot.

outlier_ind_BloodPressure = grab_outliers(train, "BloodPressure", True)

fig = plt.figure(figsize=(8,6))
g = sns.kdeplot(x=train.loc[~train.index.isin(outlier_ind_BloodPressure), "BloodPressure"], hue=train["Outcome"], shade=True)
g.yaxis.set_minor_locator(AutoMinorLocator())
g.tick_params(which="both", width=2)
g.tick_params(which="major", length=7)
g.tick_params(which="minor", length=4)
plt.show()

train.loc[(train["BloodPressure"].isnull()) & (train["Outcome"] == 1), "BloodPressure"] = \
    train.loc[(~train.index.isin(outlier_ind_BloodPressure)) & (train["Outcome"] == 1), "BloodPressure"].mean()
train.loc[(train["BloodPressure"].isnull()) & (train["Outcome"] == 0), "BloodPressure"] = \
    train.loc[(~train.index.isin(outlier_ind_BloodPressure)) & (train["Outcome"] == 0), "BloodPressure"].mean()

# SkinThickness

outlier_ind_SkinThickness = grab_outliers(train, "SkinThickness", True)

fig = plt.figure(figsize=(8,6))
g = sns.kdeplot(x=train.loc[~train.index.isin(outlier_ind_SkinThickness), "SkinThickness"], hue=train["Outcome"], shade=True)
g.yaxis.set_minor_locator(AutoMinorLocator())
g.tick_params(which="both", width=2)
g.tick_params(which="major", length=7)
g.tick_params(which="minor", length=4)
plt.show()

train.loc[(train["SkinThickness"].isnull()) & (train["Outcome"] == 1), "SkinThickness"] = \
    train.loc[(~train.index.isin(outlier_ind_SkinThickness)) & (train["Outcome"] == 1), "SkinThickness"].mean()
train.loc[(train["SkinThickness"].isnull()) & (train["Outcome"] == 0), "SkinThickness"] = \
    train.loc[(~train.index.isin(outlier_ind_SkinThickness)) & (train["Outcome"] == 0), "SkinThickness"].mean()

# Insulin
outlier_ind_Insulin = grab_outliers(train, "Insulin", True)

fig = plt.figure(figsize=(8,6))
g = sns.boxplot(x=train["Insulin"], palette="rainbow")
g.yaxis.set_minor_locator(AutoMinorLocator())
g.tick_params(which="both", width=2)
g.tick_params(which="major", length=7)
g.tick_params(which="minor", length=4)
plt.show()

for j in train["Outcome"].unique():
    plt.figure(figsize=(8,6))
    g = sns.boxplot(x=train.loc[train["Outcome"] == j, "Insulin"], palette="rainbow")
    g.yaxis.set_minor_locator(AutoMinorLocator())
    g.set_title("Outcome: " + str(j))
    g.tick_params(which="both", width=2)
    g.tick_params(which="major", length=7)
    g.tick_params(which="minor", length=4)
    plt.show()

train.loc[(train["Insulin"].isnull()) & (train["Outcome"] == 1), "Insulin"] = \
    train.loc[(~train.index.isin(outlier_ind_Insulin)) & (train["Outcome"] == 1), "Insulin"].mean()
train.loc[(train["Insulin"].isnull()) & (train["Outcome"] == 0), "Insulin"] = \
    train.loc[(~train.index.isin(outlier_ind_Insulin)) & (train["Outcome"] == 0), "Insulin"].mean()

# BMI
outlier_ind_BMI = grab_outliers(train, "BMI", True)

fig = plt.figure(figsize=(8,6))
g = sns.boxplot(x=train["BMI"], palette="rainbow")
g.yaxis.set_minor_locator(AutoMinorLocator())
g.tick_params(which="both", width=2)
g.tick_params(which="major", length=7)
g.tick_params(which="minor", length=4)
plt.show()

for j in train["Outcome"].unique():
    plt.figure(figsize=(8,6))
    g = sns.boxplot(x=train.loc[train["Outcome"] == j, "BMI"], palette="rainbow")
    g.yaxis.set_minor_locator(AutoMinorLocator())
    g.set_title("Outcome: " + str(j))
    g.tick_params(which="both", width=2)
    g.tick_params(which="major", length=7)
    g.tick_params(which="minor", length=4)
    plt.show()

train.loc[(train["BMI"].isnull()) & (train["Outcome"] == 1), "BMI"] = \
    train.loc[(~train.index.isin(outlier_ind_BMI)) & (train["Outcome"] == 1), "BMI"].mean()
train.loc[(train["BMI"].isnull()) & (train["Outcome"] == 0), "BMI"] = \
    train.loc[(~train.index.isin(outlier_ind_BMI)) & (train["Outcome"] == 0), "BMI"].mean()


# OUTLIERS
# For outliers we have different options:
# a) We can suppress them by rounding to the nearest limit that we calculate from IQR
# b) We can trim them: we need to choose how much we want to trim
# c) We can decide to do nothing
##############################################

"""
def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.25)
    quartile3 = dataframe[variable].quantile(0.75)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

for col in [col for col in train.columns if str(col) not in ["DiabetesPedigreeFunction", "Age", "Outcome"]]:
    replace_with_thresholds(train, col)
"""

##############################################

"""
# Drop outliers
outliers_arr = []
for col in [col for col in train.columns if train[col].dtypes != "object"]:
    outlier_indexes = grab_outliers(train, col, True)
    for i in outlier_indexes:
        if i not in outliers_arr:
            outliers_arr.append(i)
        else:
            continue

print("Outlier percentage: %" + str((len(outliers_arr) / len(train)) * 100))
"""

##############################################

for col in [col for col in train.columns if train[col].dtypes != "object"]:
    fig = plt.figure(figsize=(8,6))
    g = sns.kdeplot(x=train[col], shade=True)
    g.set_title("Col name: " + str(col))
    g.xaxis.set_minor_locator(AutoMinorLocator(5))
    g.yaxis.set_minor_locator(AutoMinorLocator(5))
    g.tick_params(which="both", width=2)
    g.tick_params(which="major", length=7)
    g.tick_params(which="minor", length=4)
    plt.show()

for col in [col for col in train.columns if train[col].dtypes != "object"]:
    fig = plt.figure(figsize=(8, 6))
    g = sns.distplot(x=train[col], kde=False, color="orange", hist_kws=dict(edgecolor="black", linewidth=2))
    g.set_title("Col name: Log " + str(col))
    g.xaxis.set_minor_locator(AutoMinorLocator(5))
    g.yaxis.set_minor_locator(AutoMinorLocator(5))
    g.tick_params(which="both", width=2)
    g.tick_params(which="major", length=7)
    g.tick_params(which="minor", length=4)
    plt.show()

# Normalization & Scaling

train.loc[:, "Pregnancies"] = np.log1p(train.loc[:, "Pregnancies"])
train.loc[:, "BMI"] = np.log1p(train.loc[:, "BMI"])
train.loc[:, "Age"] = np.log1p(train.loc[:, "Age"])

###################
# TEST
###################
# HANDLING MISSING VALUES

# Glucose
# There is no extreme skew, so I just go with conditional means to fill.

test.loc[(test["Glucose"].isnull()) & (test["Outcome"] == 1), "Glucose"] = test.loc[test["Outcome"] == 1, "Glucose"].mean()
test.loc[(test["Glucose"].isnull()) & (test["Outcome"] == 0), "Glucose"] = test.loc[test["Outcome"] == 0, "Glucose"].mean()

# Blood Pressure

outlier_ind_BloodPressure = grab_outliers(test, "BloodPressure", True)

test.loc[(test["BloodPressure"].isnull()) & (test["Outcome"] == 1), "BloodPressure"] = \
    test.loc[(~test.index.isin(outlier_ind_BloodPressure)) & (test["Outcome"] == 1), "BloodPressure"].mean()
test.loc[(test["BloodPressure"].isnull()) & (test["Outcome"] == 0), "BloodPressure"] = \
    test.loc[(~test.index.isin(outlier_ind_BloodPressure)) & (test["Outcome"] == 0), "BloodPressure"].mean()

# SkinThickness

outlier_ind_SkinThickness = grab_outliers(test, "SkinThickness", True)

test.loc[(test["SkinThickness"].isnull()) & (test["Outcome"] == 1), "SkinThickness"] = \
    test.loc[(~test.index.isin(outlier_ind_SkinThickness)) & (test["Outcome"] == 1), "SkinThickness"].mean()
test.loc[(test["SkinThickness"].isnull()) & (test["Outcome"] == 0), "SkinThickness"] = \
    test.loc[(~test.index.isin(outlier_ind_SkinThickness)) & (test["Outcome"] == 0), "SkinThickness"].mean()

# Insulin
outlier_ind_Insulin = grab_outliers(test, "Insulin", True)

test.loc[(test["Insulin"].isnull()) & (test["Outcome"] == 1), "Insulin"] = \
    test.loc[(~test.index.isin(outlier_ind_Insulin)) & (test["Outcome"] == 1), "Insulin"].mean()
test.loc[(test["Insulin"].isnull()) & (test["Outcome"] == 0), "Insulin"] = \
    test.loc[(~test.index.isin(outlier_ind_Insulin)) & (test["Outcome"] == 0), "Insulin"].mean()

# BMI
outlier_ind_BMI = grab_outliers(test, "BMI", True)

test.loc[(test["BMI"].isnull()) & (test["Outcome"] == 1), "BMI"] = \
    test.loc[(~test.index.isin(outlier_ind_BMI)) & (test["Outcome"] == 1), "BMI"].mean()
test.loc[(test["BMI"].isnull()) & (test["Outcome"] == 0), "BMI"] = \
    test.loc[(~test.index.isin(outlier_ind_BMI)) & (test["Outcome"] == 0), "BMI"].mean()


## Normalization & Scaling

test.loc[:, "Pregnancies"] = np.log1p(test["Pregnancies"])
test.loc[:, "BMI"] = np.log1p(test["BMI"])
test.loc[:, "Age"] = np.log1p(test["Age"])

###################
"""
for col in [col for col in test.columns if str(col) not in ["DiabetesPedigreeFunction", "Age", "Outcome"]]:
    replace_with_thresholds(test, col)
"""
"""
outliers_test_arr = []
for col in [col for col in test.columns if test[col].dtypes != "object"]:
    outlier_indexes = grab_outliers(test, col, True)
    for i in outlier_indexes:
        if i not in outliers_arr:
            outliers_arr.append(i)
        else:
            continue

print("Outlier percentage: %" + str((len(outliers_arr) / len(train)) * 100))
"""
###################
###################
# MODELLING       #
###################
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, cross_validate
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, plot_confusion_matrix

X_train = train.drop("Outcome", axis=1)
y_train = train["Outcome"]
X_test = test.drop("Outcome", axis=1)
y_test = test["Outcome"]


###################
# BASE MODELS
###################

# Base Logistic Regression
lr = LogisticRegression(max_iter=1000, random_state=26, class_weight="balanced")

cv_results = cross_validate(lr,
                           X_train, y_train,
                           cv=5,
                           scoring=["f1", "accuracy", "precision", "recall", "roc_auc"])

print(cv_results["test_f1"].mean())
print(cv_results['test_accuracy'].mean())
print(cv_results['test_precision'].mean())
print(cv_results['test_recall'].mean())



# F1 optimized logistic regression
## To take a different approach towards imbalance of the data in respect to outcome
## we may change the class_weight accordingly and search through best values for F1 score
lr = LogisticRegression(max_iter=1000, random_state=26)

#Setting the range for class weights
weights = np.linspace(0.0,0.99,200)

#Creating a dictionary grid for grid search
param_grid = {'class_weight': [{0:x, 1:1.0-x} for x in weights]}

#Fitting grid search to the train data with 5 folds
gridsearch = GridSearchCV(estimator= lr,
                          param_grid= param_grid,
                          cv=StratifiedKFold(),
                          n_jobs=-1,
                          scoring='f1',
                          verbose=2).fit(X_train, y_train)

#Plotting the score for different values of weight
plt.figure(figsize=(12,8))
weight_data = pd.DataFrame({ 'score': gridsearch.cv_results_['mean_test_score'], 'weight': (1- weights)})
sns.lineplot(weight_data['weight'], weight_data['score'])
plt.xlabel('Weight for class 1')
plt.ylabel('F1 score')
plt.xticks([round(i/10,1) for i in range(0,11,1)])
plt.title('Scoring for different class weights', fontsize=24)
plt.show()

weight_data.sort_values(by="score", ascending=False).iloc[0]

lr_model = LogisticRegression(max_iter=1000, random_state=26, class_weight={0: (1-.728), 1: 0.728})

cv_results = cross_validate(lr_model,
                           X_train, y_train,
                           cv=5,
                           scoring=["f1", "accuracy", "precision", "recall", "roc_auc"])

print(cv_results["test_f1"].mean())
print(cv_results['test_accuracy'].mean())
print(cv_results['test_precision'].mean())
print(cv_results['test_recall'].mean())

lr_model.fit(X_train, y_train)

y_pred = lr_model.predict(X_test)

print(f"Accuracy: {round(accuracy_score(y_pred, y_test), 3)}")
print(f"Recall: {round(recall_score(y_pred,y_test),3)}")
print(f"Precision: {round(precision_score(y_pred,y_test), 3)}")
print(f"F1: {round(f1_score(y_pred,y_test), 3)}")

fig = plt.figure(figsize=(8, 6))
g = fig.add_subplot(1,1,1)
plot_confusion_matrix(lr_model, X_test, y_test, ax=g)
plt.show()

print(classification_report(y_test, y_pred))

# Base Random Forest *

rf = RandomForestClassifier(random_state=26, class_weight="balanced")

cv_results = cross_validate(rf,
                            X_train, y_train,
                            cv=5,
                            scoring=["f1", "accuracy", "precision", "recall", "roc_auc"])

cv_results['test_accuracy'].mean()
cv_results['test_precision'].mean()
cv_results['test_recall'].mean()
cv_results["test_f1"].mean()

rf = RandomForestClassifier(random_state=26, class_weight="balanced_subsample")

cv_results = cross_validate(rf,
                            X_train, y_train,
                            cv=5,
                            scoring=["f1", "accuracy", "precision", "recall", "roc_auc"])

print(cv_results['test_accuracy'].mean())
print(cv_results['test_precision'].mean())
print(cv_results['test_recall'].mean())
print(cv_results["test_f1"].mean())

rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

print(f"Accuracy: {round(accuracy_score(y_pred, y_test), 3)}")
print(f"Recall: {round(recall_score(y_pred,y_test),3)}")
print(f"Precision: {round(precision_score(y_pred,y_test), 3)}")
print(f"F1: {round(f1_score(y_pred,y_test), 3)}")

# Base LGBM

lgbm = LGBMClassifier(random_state=26, class_weight="balanced")

cv_results = cross_validate(lgbm,
                            X_train, y_train,
                            cv=5,
                            scoring=["f1", "accuracy", "precision", "recall", "roc_auc"])

cv_results['test_accuracy'].mean()
cv_results['test_precision'].mean()
cv_results['test_recall'].mean()
cv_results["test_f1"].mean()

lgbm.fit(X_train, y_train)

y_pred = lgbm.predict(X_test)

print(f"Accuracy: {round(accuracy_score(y_pred, y_test), 3)}")
print(f"Recall: {round(recall_score(y_pred,y_test),3)}")
print(f"Precision: {round(precision_score(y_pred,y_test), 3)}")
print(f"F1: {round(f1_score(y_pred,y_test), 3)}")


# Base Models
def base_models(X, y, scoring=["f1", "accuracy", "precision", "recall"]):
    print("Base Models....")
    classifiers = [('LR', LogisticRegression(max_iter=1000, random_state=26)),
                   ("CART", DecisionTreeClassifier(random_state=26)),
                   ("RF", RandomForestClassifier(random_state=26)),
                   ('Adaboost', AdaBoostClassifier(random_state=26)),
                   # ('GBM', GradientBoostingClassifier(random_state=26)),
                   # ('XGBoost', XGBClassifier(random_state=26, use_label_encoder=False, eval_metric='logloss')),
                   ('LightGBM', LGBMClassifier(random_state=26)),
                   # ('CatBoost', CatBoostClassifier(verbose=False))
                   ]

    for name, classifier in classifiers:
        cv_results = cross_validate(classifier, X, y, cv=5, scoring=scoring)
        print(f"{scoring[0]}: {round(cv_results['test_f1'].mean(), 4)} ({name}) ")
        print(f"{scoring[1]}: {round(cv_results['test_accuracy'].mean(), 4)} ({name}) ")
        print(f"{scoring[2]}: {round(cv_results['test_precision'].mean(), 4)} ({name}) ")
        print(f"{scoring[3]}: {round(cv_results['test_recall'].mean(), 4)} ({name}) ")


base_models(X_train, y_train)

# MODELLING & HYPERPARAMETER TUNING

cart_params = {"random_state": 26,
               'max_depth': range(2, 20),
               "min_samples_split": range(2, 30)}

rf_params = {"random_state": 26,
             "max_depth": [8, 15, None],
             "max_features": [5, 7, "auto"],
             "min_samples_split": [15, 20],
             "n_estimators": [300, 500, 800, 1200]}

xgboost_params = {"random_state": 26,
                  "learning_rate": [0.1, 0.01, 0.2],
                  "max_depth": [5, 8, None],
                  "n_estimators": [300, 500, 800, 1200],
                  "colsample_bytree": [0.5, 0.7, 1]}

lightgbm_params = {"random_state": 26,
                   "learning_rate": [0.01, 0.1, 0.2],
                   "n_estimators": [300, 500, 800, 1200],
                   "colsample_bytree": [0.5, 0.7, 1]}

classifiers = [("CART", DecisionTreeClassifier(), cart_params),
               ("RF", RandomForestClassifier(), rf_params),
               # ('XGBoost', XGBClassifier(use_label_encoder=False, eval_metric="logloss"), xgboost_params),
               ('LightGBM', LGBMClassifier(), lightgbm_params)]

def hyperparameter_optimization(X, y, cv=5, scoring=["f1", "accuracy", "precision", "recall"]):
    print("Hyperparameter Optimization....")
    best_models = {}
    for name, classifier, params in classifiers:
        print(f"########## {name} ##########")
        cv_results = cross_validate(classifier, X, y, cv=cv, scoring=scoring)
        print(f"{scoring[0]} (Before): {round(cv_results['test_f1'].mean(), 4)}")
        print(f"{scoring[1]} (Before): {round(cv_results['test_accuracy'].mean(), 4)}")
        print(f"{scoring[2]} (Before): {round(cv_results['test_precision'].mean(), 4)}")
        print(f"{scoring[3]} (Before): {round(cv_results['test_recall'].mean(), 4)}")

        gs_best = GridSearchCV(classifier, params, cv=cv, n_jobs=-1, verbose=False).fit(X, y)
        final_model = classifier.set_params(**gs_best.best_params_)

        cv_results = cross_validate(final_model, X, y, cv=cv, scoring=scoring)
        print(f"{scoring[0]} (After): {round(cv_results['test_f1'].mean(), 4)}")
        print(f"{scoring[1]} (After): {round(cv_results['test_accuracy'].mean(), 4)}")
        print(f"{scoring[2]} (After): {round(cv_results['test_precision'].mean(), 4)}")
        print(f"{scoring[3]} (After): {round(cv_results['test_recall'].mean(), 4)}")
        print(f"{name} best params: {gs_best.best_params_}", end="\n\n")
        best_models[name] = final_model
    return best_models

best_models_dict = hyperparameter_optimization(X_train, y_train)


model_final = best_models_dict["LightGBM"].fit(X_train, y_train)

import joblib

joblib.dump(model_final, "diabetes/LightGBM.pkl")

lgbm_model_from_dir = joblib.load("diabetes/LightGBM.pkl")

random_selection = X_test.sample(1)

lgbm_model_from_dir.predict(random_selection)