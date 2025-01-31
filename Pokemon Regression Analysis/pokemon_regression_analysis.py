import string
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

DATASET_FILE = "dataset.csv"

#Column names used as header
colnames = ['id', 'name', 'type_1', 'type_2', 'total','hp', 'attack', 'defense', 'special_attack','special_defense', 'speed', 'generation', 'is_legendary']

#Load dataset
data = pd.read_csv(DATASET_FILE, 
                   sep=',',
                   header=0,
                   names=colnames,
                   na_values='?')

#Remove any duplicates
data = data.drop_duplicates('id', keep='first', inplace=False)
data.reset_index(inplace=True, drop=True)
np.random.seed(12)
data.iloc[np.random.choice(721, np.random.randint(1,20), replace=False).tolist(), 6] = np.nan
data.iloc[np.random.choice(721, np.random.randint(1,20), replace=False).tolist(), 2] = np.nan
data.iloc[np.random.choice(721, np.random.randint(1,20), replace=False).tolist(), 8] = np.nan
print("Loaded `Pokemon` dataset into a dataframe of size ({} x {})".format(data.shape[0], data.shape[1]))

print(data.head())

#Handle the each missing values
import warnings   #Warnings were suppressed, clean output provided
warnings.filterwarnings("ignore")

print("There are missing values in the dataset: {}".
      format(data.isnull().any().any()))  #Checked whether there are missing values in the dataset
print("N. of missing values for attribute 'type_1': {}".  
     format(data.type_1.isnull().sum()))   #Missing values in the 'type_1' column are counted
print("N. of missing values for attribute 'total': {}".
     format(data.total.isnull().sum()))  #Missing values in the 'total' column are counted
print("N. of missing values for attribute 'hp': {}".
     format(data.hp.isnull().sum()))  #Missing values in the 'hp' column are counted
print("N. of missing values for attribute 'attack': {}".
     format(data.attack.isnull().sum()))  #Missing values in the 'attack' column are counted
print("N. of missing values for attribute 'defense': {}".
     format(data.defense.isnull().sum()))  #Missing values in the 'defense' column are counted
print("N. of missing values for attribute 'special_attack': {}".
     format(data.special_attack.isnull().sum()))  #Missing values in the 'special_attack' column are counted
print("N. of missing values for attribute 'special_defense': {}".
     format(data.special_defense.isnull().sum()))  #Missing values in the 'special_defense' column are counted
print("N. of missing values for attribute 'speed': {}".
     format(data.speed.isnull().sum()))  #Missing values in the 'speed' column are counted
print("N. of missing values for attribute 'generation': {}".
     format(data.generation.isnull().sum()))  #Missing values in the 'generation' column are counted
print("N. of missing values for attribute 'is_legendary': {}".
     format(data.is_legendary.isnull().sum()))  #Missing values in the 'is_legendary' column are counted

#Missing values in the columns were filled using appropriate techniques
data.type_1.fillna(data.type_1.mode()[0], inplace=True)
data.attack.fillna(data.attack.median(), inplace=True)
data.special_attack.fillna(data.special_attack.median(), inplace=True)

#Columns with too many missing values and unused columns were removed
data.drop("id", axis=1, inplace=True)
data.drop("name", axis=1, inplace=True)
data.drop("type_2", axis=1, inplace=True)

print(data.head())

#Analyze the data using box plots
fig, axes = plt.subplots(2, 4, figsize=(18,12))  #Created a Figure containing 2x4 subplots
#A boxplot was created for each data column
_ = sns.boxplot(data.total, color="#FF0000", ax=axes[0,0])
_ = sns.boxplot(data.hp, color="#00FF00", ax=axes[0,1])
_ = sns.boxplot(data.attack, color="#0000FF", ax=axes[0,2])
_ = sns.boxplot(data.defense, color="#FFFF00", ax=axes[0,3])
_ = sns.boxplot(data.special_attack, color="#FFA500", ax=axes[1,0])
_ = sns.boxplot(data.special_defense, color="#800080", ax=axes[1,1])
_ = sns.boxplot(data.speed, color="#FFC0CB", ax=axes[1,2])
_ = sns.boxplot(data.generation, color="#808080", ax=axes[1,3])
plt.subplots_adjust(wspace=.3, hspace=.3)
plt.show()

#Find the number of outliers at each column below this line
columns = ['total', 'hp', 'attack', 'defense', 'special_attack', 'special_defense', 'speed', 'generation'] 

for column in columns:
    q1, q3 = data.loc[data[column].notnull(), column].quantile([.25, .75])  #1st and 3rd quartiles were calculated for each column
    print(f"1st Quartile of '{column}': {q1:.2f}")
    print(f"3rd Quartile of '{column}': {q3:.2f}")
    
    IQR = q3 - q1  #IQR value was calculated for each column
    print(f"IQR of '{column}': {IQR:.2f}")
    
    fence_left = q1 - 1.5 * IQR   #Fence values were calculated
    fence_right = q3 + 1.5 * IQR
    print(f"Fence range: [{fence_left:.2f}, {fence_right:.2f}]")
    
    outlier_count = data[(data[column] > fence_right) | (data[column] < fence_left)].shape[0]  #Outlier count was calculated
    print(f"Number of instances containing outlier of '{column}': {outlier_count}")
    print()

#Handle the outliers
#The data points were winsorized
stats.mstats.winsorize(data.total, limits=0.05, inplace=True)
stats.mstats.winsorize(data.hp, limits=0.05, inplace=True)
stats.mstats.winsorize(data.attack, limits=0.05, inplace=True)
stats.mstats.winsorize(data.defense, limits=0.05, inplace=True)
stats.mstats.winsorize(data.special_attack, limits=0.05, inplace=True)
stats.mstats.winsorize(data.special_defense, limits=0.05, inplace=True)
stats.mstats.winsorize(data.speed, limits=0.05, inplace=True)
stats.mstats.winsorize(data.generation, limits=0.05, inplace=True)

fig, axes = plt.subplots(2, 4, figsize=(18,12))  #Created a Figure containing 2x4 subplots
#A boxplot was created for each data column
_ = sns.boxplot(data.total, color="#FF0000", ax=axes[0,0])
_ = sns.boxplot(data.hp, color="#00FF00", ax=axes[0,1])
_ = sns.boxplot(data.attack, color="#0000FF", ax=axes[0,2])
_ = sns.boxplot(data.defense, color="#FFFF00", ax=axes[0,3])
_ = sns.boxplot(data.special_attack, color="#FFA500", ax=axes[1,0])
_ = sns.boxplot(data.special_defense, color="#800080", ax=axes[1,1])
_ = sns.boxplot(data.speed, color="#FFC0CB", ax=axes[1,2])
_ = sns.boxplot(data.generation, color="#808080", ax=axes[1,3])
plt.subplots_adjust(wspace=.3, hspace=.3)
plt.show()

#Handle the categorical data
categorical_features = ['type_1','generation']   #Categorical columns list was created
data = pd.get_dummies(data, columns=categorical_features)  #One-hot encoding was applied to categorical columns
print(data.head())

#Train linear regression model to estimate total
columns = data.columns.tolist()  #Column names were converted to a list
columns.insert(len(columns), columns.pop(columns.index("total")))  #The 'total' column was moved to the last position.
data = data.loc[:, columns]  #The dataset was reordered according to column sequence

x = data.iloc[:,:-1]  #Features were selected
y = data.total

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)  #The dataset was split into %80 training and %20 test
print("Training Set shape: {}".format(x_train.shape))
print("Test Set shape: {}".format(x_test.shape))

def fit(model, x_train, y_train):  #The function was defined to train the model with the training data
    model.fit(x_train, y_train)

def evaluate(true_values, predicted_values):  #True and predicted values were compared
    print("Mean Squared Error (MSE) = {:.6f}".format(mean_squared_error(true_values, predicted_values)))
    print("Coefficient of Determination (R2 score) = {:.6f}".format(r2_score(true_values, predicted_values)))
   
def plot_true_vs_predicted(y_true, y_predicted):  #A graph was created to compare the true and predicted values
    fig, ax = plt.subplots()
    _ = ax.scatter(y_true, y_predicted, edgecolors=(0, 0, 0), color='red')
    _ = ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'k--', lw=4)
    _ = ax.set_xlabel('True Total')
    _ = ax.set_ylabel('Predicted Total')
    plt.show()

model = linear_model.LinearRegression()  #A linear regression model was created

fit(model, x_train, y_train)  #The model was trained with the training data

print("***** Evaluate Predictions on Training Set *****")
y_train_pred = model.predict(x_train)  #Predictions were made for the training set
evaluate(y_train, y_train_pred)
plot_true_vs_predicted(y_train, y_train_pred)
print()

print("***** Evaluate Predictions on Test Set *****")
y_test_pred = model.predict(x_test)  #Predictions were made for the test set
evaluate(y_test, y_test_pred)
plot_true_vs_predicted(y_test,y_test_pred)