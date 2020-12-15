
# ------------------------------------------------ Importing Required Libraries -------------------------------------------- #

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn import preprocessing
from boruta import BorutaPy

# ------------------------------------------------- Preprocessing -------------------------------------- #

df = pd.read_csv("C:/Users/Sanket/Documents/Kaggle_Automobile_Train.csv")
df

df.describe()

df.info
df.shape
df.columns

# ----------- Unique variables

for val in df:
    print(val, "  ", df[val].unique().shape)

# ---------- Dropping column having unique values

df = df.drop("Unnamed: 0",axis=1)
df.columns
df.shape

# --------- Converting Name column into column having only first letter i.e company name of car

New_Name = []
for val in df["Name"]:
    New_Name.append(val.split()[0])

df = df.drop("Name", axis=1)

df["New_Name"] = New_Name

df.columns
df["New_Name"].value_counts()

# --------- Observing Factor levels for transformation

for val in df:
    print(df[val].value_counts())

df.dtypes

# --------- Transformation of columns

df["Year"].value_counts()

df["Car_Used"] = 2020 - df["Year"]

df = df.drop("Year", axis=1)
df.columns

# --------- Quasi Constant featurefor biasness

(df["Owner_Type"].value_counts()/len(df)).values[0]

Biased_Variables = []

for val in df:
    bias = (df[val].value_counts()/len(df)).values[0]
    if bias > 0.9:
        Biased_Variables.append(val)
print(Biased_Variables)

# -------- Replacing null to NaN

df = df.replace('null bhp', np.nan)

# -------- Replacing 0.0 kmpl to nan in Mileage

df = df.replace("0.0 kmpl", np.nan)

# --------- Converting Seats into categorical Variable

df["Seats"] = df["Seats"].astype("object")
df["Seats"].dtypes

# --------- NA values

df.isnull().any()
df.isnull().sum()/len(df)*100

df = df.drop("New_Price", axis = 1)
df.columns

def imputenull(data):
    for col in data.columns:
        if data[col].dtypes == "int64" or data[col].dtypes == 'float64':
            data[col].fillna((data[col].mean()), inplace = True)
        else:
            data[col].fillna(data[col].value_counts().index[0], inplace = True)                # index[0] will give mode.........

imputenull(df)
df.isnull().sum()
df.info()
df.shape

# --------- Converting columns from chatagorical to int

EnginE = []
for val in df["Engine"]:
    EnginE.append(val.split()[0])

df = df.drop("Engine", axis=1)
df["Engine"] = EnginE
df["Engine"] = df["Engine"].astype("int64")
df['Engine'].dtypes

Power = []
for val in df["Power"]:
    Power.append(val.split()[0])

df = df.drop("Power", axis=1)
df["Power"] = Power
df["Power"] = df["Power"].astype("float64")
df['Power'].dtypes

Mileage = []
for val in df["Mileage"]:
    Mileage.append(val.split()[0])

df = df.drop("Mileage", axis=1)
df["Mileage"] = Mileage
df["Mileage"] = df["Mileage"].astype("float64")
df['Mileage'].dtypes

# ---------- Converting Seats into Chatagorical Variable again

df["Seats"] = df["Seats"].astype("object")
df["Seats"].dtypes

# ---------- Skewness

df.skew()

# ---------- Dropping Highly skewed values

p = df[df["Kilometers_Driven"]>6000000]
df.drop(2328, axis=0, inplace=True)
df.skew()

# --------- Seperating Numeric and Categorical Data

df_factor = df.select_dtypes(include=["object"])
df_num = df.select_dtypes(include=["int64","float64"])

# ------- Scaling of data

for val in df_num:
    df_num[val] = np.log10(df_num[val])

df_num.skew()

# ---------- Outlier Treatment

from scipy import stats

z = np.abs(stats.zscore(df_num["Price"]))
print(z)
threshold = 3
print(np.where(z > threshold))
x= np.where(z > 3)

#no of outliers
no_of_outlier=  len(x[0])
no_of_outlier

df.iloc[np.where(z > 3)[:1]]

# ----------- Dropping rows with outliers

df_num.drop(df_num.index[x[:1]], inplace=True)
df_factor.drop(df_factor.index[x[:1]], inplace=True)

df_num.skew()

# ---------- EDA

# Density Plot and Histogram of Price
sns.distplot(df_num['Price'], hist=True, kde=True,bins=50, color = 'darkblue',hist_kws={'edgecolor':'black'},kde_kws={'linewidth': 2}).set_title("After log Transformation")

sns.pairplot(df_num)

sns.boxplot(x=df_num["Price"])

sns.relplot(x = "Kilometers_Driven", y = "Price", data = df)

sns.relplot(x = "Fuel_Type", y = "Price", data = df)

sns.relplot(x = "New_Name",y = "Price", data = df)

sns.barplot(x = "New_Name", y = "Price", data = df)

sns.scatterplot(x = "Car_Used", y = "Price", data = df)

sns.relplot(x = "Transmission", y = "Price", data = df)

# --------- Corrplot

corrmat = df_num.corr()
corrmat

fig,ax = plt.subplots()
fig.set_size_inches(11,11)
sns.heatmap(corrmat)

# --------- Check for threshold of corrplot

def checkcorrelation(dataset, threshold):
    col_corr = set()
    cor_matrix = dataset.corr()
    for i in range(len(cor_matrix.columns)):
        for j in range(i):
            if abs(cor_matrix.iloc[i,j]) > threshold:
                colname = cor_matrix.columns[i]
                col_corr.add(colname)
    return col_corr

checkcorrelation(df_num,0.9)

# --------- Combining Fact and Numeric data

New_df = pd.concat([df_num,df_factor], axis=1)
New_df.columns
New_df.shape

# ---------- Model Building

tar_var = New_df['Price']
New_df.drop("Price", axis = 1, inplace=True)
New_df.columns

# ----------- Label Encoding

le = preprocessing.LabelEncoder()
df = New_df.apply(le.fit_transform)
df.head()

# ----------- Boruta for feature selection

X = np.array(df)
Y = np.array(tar_var)

rf = RandomForestRegressor()

boruta_feature_selector = BorutaPy(rf, n_estimators = 'auto', random_state=4242, max_iter = 20, perc = 90, verbose=2)
boruta_feature_selector.fit(X,Y)

boruta_feature_selector.support_

Imp_feature = pd.DataFrame({"Feature_Name":New_df.columns, "Importance":list(boruta_feature_selector.support_)})

columns = New_df.columns

final_features = list()
indexes = np.where(boruta_feature_selector.support_ == True)
for x in np.nditer(indexes):
    final_features.append(columns[x])
print(final_features)

df = df[['Kilometers_Driven', 'Car_Used', 'Engine', 'Power', 'Mileage', 'Location', 'Transmission', 'New_Name']]

# ---------------- Splitting

x_train,x_test,y_train,y_test = train_test_split(df,tar_var, random_state=1, test_size=0.3)             # random_state = set.seed

x_train.shape
x_test.shape
y_train.shape
y_test.shape

# ----------------- Implementing Model

# ------------ Linear Regression

from sklearn import linear_model as lm

model = lm.LinearRegression()
result = model.fit(x_train,y_train)

print(model.intercept_)
print(model.coef_)
model.fit

# --------------- Predicting the values

predictions = model.predict(x_test)

# --------------- Getting model evolution from predicted values

from sklearn.metrics import mean_squared_error,r2_score

mse = mean_squared_error(y_test,predictions)
r2 = r2_score(y_test,predictions)
rmse = mse**(1/2)
print(mse)
print(rmse)
print(r2)

from statsmodels.stats.outliers_influence import variance_inflation_factor

resiuduals = y_test - predictions

vif = [variance_inflation_factor(x_train.values,i) for i in range(x_train.shape[1])]
multicoolinearity = pd.DataFrame({'vif':vif[0:]}, index=x_train.columns).T

# ---------- Error should be normally destributed
sns.distplot(resiuduals)
resiuduals.skew()

np.mean(resiuduals)

# ---------- QQ plot
import statsmodels.api as sm

sm.qqplot(resiuduals,stats.t,fit=True,line='45')
py.show()

# ----------- Homoscedasticity
fig, ax = plt.subplots(figsize=(6,2.5))
_ = ax.scatter(predictions,resiuduals)

# ----------- No autocorrelation of residuals
import statsmodels.tsa.api as smt

acf = smt.graphics.plot_acf(resiuduals, lags=40, alpha=0.05)
acf.show

# ---------------------- Ridge regression

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge

ridge = Ridge()

parameters = {'alpha': [1e-15,1e-10,1e-8,1e-4,1e-3,1e-2,1,5,10,20]}

ridge_regressor = GridSearchCV(ridge, parameters, scoring='neg_mean_squared_error',cv=5)

ridge_regressor.fit(x_train,y_train)

print(ridge_regressor.best_params_)
print(ridge_regressor.best_score_)

ridge_pred = ridge_regressor.predict(x_test)

mse = mean_squared_error(y_test,ridge_pred)
r2 = r2_score(y_test,ridge_pred)
rmse = mse**(1/2)
print(mse)
print(rmse)
print(r2)

# --------------- Lasso Regression

from sklearn.linear_model import Lasso

lasso = Lasso()

parameters = {'alpha': [1e-15,1e-10,1e-8,1e-4,1e-3,1e-2,1,5,10,20]}

lasso_regressor = GridSearchCV(lasso, parameters, scoring='neg_mean_squared_error',cv=5)

lasso_regressor.fit(x_train,y_train)

print(lasso_regressor.best_params_)
print(lasso_regressor.best_score_)

lasso_pred = lasso_regressor.predict(x_test)

mse = mean_squared_error(y_test,lasso_pred)
r2 = r2_score(y_test,lasso_pred)
rmse = mse**(1/2)
print(mse)
print(rmse)
print(r2)

# --------------- Random Forest

from sklearn.ensemble import RandomForestRegressor

regressor_random = RandomForestRegressor(n_estimators = 200,random_state=0)
regressor_random.fit(x_train, y_train)
y_pred_random = regressor_random.predict(x_test)

regressor_random.score(x_test, y_test)
mse = mean_squared_error(y_test,y_pred_random)
rmse = mse**(1/2)
print(rmse)
print(mse)

# ---------------- Decision Tree

from sklearn.tree import DecisionTreeRegressor

dt = DecisionTreeRegressor()

dt.fit(x_train,y_train)

predictions_dt = dt.predict(x_test)

dt.score(x_test, y_test)
mse = mean_squared_error(y_test,predictions_dt)
rmse = mse**(1/2)
print(mse)
print(rmse)

# ---------- Checking the score by changing complexity parameter

from sklearn import tree
plt.figure(figsize=(7,4))
tree.plot_tree(dt,filled=True)

path = dt.cost_complexity_pruning_path(x_train,y_train)
ccp_alphas, impurities = path.ccp_alphas, path.impurities            # ---- The weakest link is characterized by an effective alpha,
                                                                     # where the nodes with the smallest effective alpha are pruned first
dts = []
for ccp_alpha in ccp_alphas:
    dt = DecisionTreeRegressor(random_state=0,ccp_alpha=ccp_alpha)
    dt.fit(x_train,y_train)
    dts.append(dt)
print("Number of nodes in the last tree is :{} with ccp_alpha : {}".format(dts[-1].tree_.node_count,ccp_alphas[-1]))

train_scores = [dt.score(x_train,y_train) for dt in dts]
test_scores = [dt.score(x_test,y_test) for dt in dts]

fig, ax = plt.subplots()
ax.set_xlabel("alpha")
ax.set_ylabel("accuracy")
ax.set_title("Accuracy vs alpha for train & test")
ax.plot(ccp_alphas,train_scores,marker='o',label='train',drawstyle = 'steps-post')
ax.plot(ccp_alphas,test_scores,marker='o',label='test',drawstyle='steps-post')
ax.legend()
plt.show()

new_dt = DecisionTreeRegressor(random_state=12,ccp_alpha=0.01)
new_dt.fit(x_train,y_train)

predictions_new_dt = new_dt.predict(x_test)

new_dt.score(x_test, y_test)
mse = mean_squared_error(y_test,predictions_new_dt)
rmse = mse**(1/2)
print(mse)
print(rmse)


# ---------------- Compairing Results

y_test_original = np.exp(y_test)
Pred_linear = np.exp(predictions)
Pred_dt = np.exp(predictions_dt)
Pred_rf = np.exp(y_pred_random)
pred_ridge = np.exp(ridge_pred)
Pred_lasso = np.exp(lasso_pred)


Predictions_Testing = pd.DataFrame({"Test":y_test_original,"Random Forest":Pred_rf,"Decision Tree":Pred_dt,"Linear Regression":Pred_linear,
                                    "Ridge Regression":pred_ridge,"Lasso Regression":Pred_lasso})




# ---------------------------------------------------- Predictions on Test Data ------------------------------------------------------------------


df_test = pd.read_csv("C:/Users/Sanket/Documents/Kaggle_Automobile.csv")
df_test.columns

# ---------- Dropping column having unique values

df_test = df_test.drop("Unnamed: 0",axis=1)
df_test.columns
df_test.shape

# --------- Converting Name column into column having only first letter i.e company name f car

New_Name = []
for val in df_test["Name"]:
    New_Name.append(val.split()[0])

df_test = df_test.drop("Name", axis=1)

df_test["New_Name"] = New_Name

df_test.columns
df_test["New_Name"].value_counts()

# --------- Transformation of columns

df_test["Year"].value_counts()

df_test["Car_Used"] = 2020 - df_test["Year"]

df_test = df_test.drop("Year", axis=1)
df_test.columns

# -------- Replacing null to NaN

df_test["Power"][df_test["Power"]=="null bhp"].shape

df_test = df_test.replace('null bhp', np.nan)

# -------- Replacing 0.0 kmpl to nan in Mileage

df_test["Mileage"][df_test["Mileage"]=="0.0 kmpl"]

df_test = df_test.replace("0.0 kmpl", np.nan)

# --------- Converting Seats into Chatagorical Variable

df_test["Seats"] = df_test["Seats"].astype("object")
df_test["Seats"].dtypes

# --------- NA values

df_test.isnull().any()
df_test.isnull().sum()/len(df)*100

df_test = df_test.drop("New_Price", axis = 1)
df_test.columns

imputenull(df_test)
df_test.isnull().sum()

# --------- Converting columns from chatagorical to int

EnginE = []
for val in df_test["Engine"]:
    EnginE.append(val.split()[0])

df_test = df_test.drop("Engine", axis=1)
df_test["Engine"] = EnginE
df_test["Engine"] = df_test["Engine"].astype("int64")
df_test['Engine'].dtypes

Power = []
for val in df_test["Power"]:
    Power.append(val.split()[0])

df_test = df_test.drop("Power", axis=1)
df_test["Power"] = Power
df_test["Power"] = df_test["Power"].astype("float64")
df_test['Power'].dtypes

Mileage = []
for val in df_test["Mileage"]:
    Mileage.append(val.split()[0])

df_test = df_test.drop("Mileage", axis=1)
df_test["Mileage"] = Mileage
df_test["Mileage"] = df_test["Mileage"].astype("float64")
df_test['Mileage'].dtypes

# --------- Converting Seats into Chatagorical Variable again

df_test["Seats"] = df_test["Seats"].astype("object")
df_test["Seats"].dtypes

# ---------- Skewness

df_test.skew()

# --------- Seperating Numeric and Chatagorical Data

df_test_factor = df_test.select_dtypes(include=["object"])
df_test_num = df_test.select_dtypes(include=["int64","float64"])

#----------- Scaling of data

for val in df_test_num:
    df_test_num[val] = np.log10(df_test_num[val])

df_test_num.skew()

# --------- Combining Fact and Numeric data

New_df_test = pd.concat([df_test_num,df_test_factor], axis=1)
New_df_test.columns
New_df_test.shape
New_df.shape

# ---------- Model Building

# ------- Label encoding

New_df_1 = pd.concat([New_df,New_df_test], axis=0)
New_df_1.shape

# ----------- Label Encoding

le = preprocessing.LabelEncoder()
df_test = New_df_1.apply(le.fit_transform)
df_test.head()

new_df_train = df_test.head(6007)
new_df_test = df_test.tail(len(df_test)-6007)

new_df_test.shape
new_df_train.shape

# ---------- Model Building

from sklearn import linear_model as lm

model = lm.LinearRegression()
result = model.fit(new_df_train,tar_var)

print(model.intercept_)
print(model.coef_)
model.fit

# --------------- Predicting the values

predictions = model.predict(new_df_test)

Final_Predictions = np.exp(predictions)

# --------------- Random Forest

from sklearn.ensemble import RandomForestRegressor
regressor_random = RandomForestRegressor(random_state=0)
regressor_random.fit(new_df_train, tar_var)
y_pred_random = regressor_random.predict(new_df_test)

Final_Predictions_rf = np.exp(y_pred_random)

# ------------ Decision Tree

from sklearn.tree import DecisionTreeRegressor

dt = DecisionTreeRegressor()

dt.fit(new_df_train,tar_var)

predictions_dt_1 = dt.predict(new_df_test)

Final_Predictions_dt = np.exp(predictions_dt_1)

# ---------------------- Ridge regression

ridge = Ridge()

parameters = {'alpha': [1e-15,1e-10,1e-8,1e-4,1e-3,1e-2,1,5,10,20]}

ridge_regressor = GridSearchCV(ridge, parameters, scoring='neg_mean_squared_error',cv=5)

ridge_regressor.fit(new_df_train,tar_var)

print(ridge_regressor.best_params_)
print(ridge_regressor.best_score_)

ridge_pred = ridge_regressor.predict(new_df_test)

final_predictions_ridge = np.exp(ridge_pred)

# --------------- Lasso Regression

lasso = Lasso()

parameters = {'alpha': [1e-15,1e-10,1e-8,1e-4,1e-3,1e-2,1,5,10,20]}

lasso_regressor = GridSearchCV(lasso, parameters, scoring='neg_mean_squared_error',cv=5)

lasso_regressor.fit(new_df_train,tar_var)

print(lasso_regressor.best_params_)
print(lasso_regressor.best_score_)

lasso_pred = lasso_regressor.predict(new_df_test)

final_predictions_lasso = np.exp(lasso_pred)

# ---------------- Final Results

Result = pd.DataFrame({"Linear Regression":Final_Predictions,"Random Forest":Final_Predictions_rf,"Decision Tree":Final_Predictions_dt,
                       "Ridge Regression":final_predictions_ridge,"Lasso Regression":final_predictions_lasso})

