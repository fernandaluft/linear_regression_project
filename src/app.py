#Imports
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

#Importing the dataset
url='https://raw.githubusercontent.com/4GeeksAcademy/linear-regression-project-tutorial/main/medical_insurance_cost.csv'
df = pd.read_csv(url)

#Exploring the data
df.head()
df.info()

#Checking for duplicates
df.duplicated().sum()

#Dropping duplicates
df.drop_duplicates(inplace=True)
df = df.reset_index(drop=True)

#Descriptive statistics and plot of categorical variables
cat_variables = df.describe(include=['O'])

#Plots
fig, axs = plt.subplots(1,3, figsize=(12,4))

sns.countplot(ax=axs[0],data=df, x=df['sex'])
sns.countplot(ax=axs[1], data=df, x=df['smoker'])
sns.countplot(ax=axs[2], data=df, x=df['region'])

plt.tight_layout()
plt.show()

#Descriptive statistics and plot of numerical variables
num_variables = df.describe()

#Plots
col_n_list = [i for i in num_variables.columns]

num_plots = len(col_n_list)
total_cols = 2
total_rows = num_plots//total_cols
fig, axs = plt.subplots(nrows=total_rows, ncols=total_cols,figsize=(10*total_cols, 6*total_rows), constrained_layout=True)

index = 0
for col in col_n_list:

    row = index //total_cols
    pos = index % total_cols
    sns.distplot(df[col], kde=True, rug = False, ax=axs[row][pos])
    
    index += 1
plt.tight_layout()
plt.show()

#Correlation matrix numerical variables
sns.heatmap(df[col_n_list].corr(), annot=True, fmt='.2f')

#Feature scaling

#Copy of the dataset and factorising categorical variables
df_encoded = df.copy()
df_encoded['sex'] = pd.factorize(df_encoded['sex'])[0]
df_encoded['smoker'] = pd.factorize(df_encoded['smoker'])[0]
df_encoded['region'] = pd.factorize(df_encoded['region'])[0]

#Normalisation of numerical variables
scaler = MinMaxScaler()
df_encoded[['age', 'bmi', 'children', 'charges']] = scaler.fit_transform(df_encoded[['age', 'bmi', 'children', 'charges']])

#Correlation Matrix
fig = plt.figure(figsize=(8,6))
sns.heatmap(df_encoded.corr(), annot=True, fmt=".2f")
plt.tight_layout()
plt.show()

#Data split
X = df_encoded.drop('charges', axis=1)
y = df_encoded['charges']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print('Train data:\nX:', X_train.shape, 'y:', y_train.shape)
print('Test data:\nX:', X_test.shape, 'y:', y_test.shape)

###Machine Learning Model###

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(f"Intercept (a): {model.intercept_}")
print(f"Coefficients (b): {model.coef_}")

##Checking accuracy of the model

#Mean Squared Error, R2 Score
msq=mean_squared_error(y_test, y_pred)
r2t=r2_score(y_test, y_pred)
print(f"The MSE of the test data is {msq} and the R2 Score is {r2}.")