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
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler

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
    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.

    Parameters
    ----------
    dataframe: dataframe
        değişken isimleri alınmak istenen dataframe'dir.
    cat_th: int, float
        numerik fakat kategorik olan değişkenler için sınıf eşik değeri
    car_th: int, float
        kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    -------
    cat_cols: list
        Kategorik değişken listesi
    num_cols: list
        Numerik değişken listesi
    cat_but_car: list
        Kategorik görünümlü kardinal değişken listesi

    Notes
    ------
    cat_cols + num_cols + cat_but_car = toplam değişken sayısı
    num_but_cat cat_cols'un içerisinde.

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

train.loc[(train["BloodPressure"].isnull()) & (df["Outcome"] == 1), "BloodPressure"] = \
    train.loc[(~train.index.isin(outlier_ind_BloodPressure)) & (train["Outcome"] == 1), "BloodPressure"].mean()
train.loc[(train["BloodPressure"].isnull()) & (df["Outcome"] == 0), "BloodPressure"] = \
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
g = sns.boxplot(x=train["Insulin"], palette="rainbow")
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