from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
data1 = pd.read_csv("data1.csv")
data2 = pd.read_csv("data2.csv")
data1.head()
data2.head(10)
merge = pd.merge(data1, data2)
merge.head(10)
merge.isnull().sum()
merge.drop('Code', axis=1, inplace=True)
merge.head(10)
merge.size, merge.shape
merge.rename(columns={'Code': 'country', 'Year': 'year', 'Schizophrenia': 'schizophrenia', 'Bipolar Disorder': 'bipolar_disorder',
                      'Eating Disorder': 'eating_disorder', 'Anxiety': 'anxiety', 'Drug Usage': 'drug_usage', 'Depression': 'depression',
                      'Alcohol': 'alcohol', 'Mental Fitness': 'mental_fitness'}, inplace=True)
merge.head(10)
numeric_columns = merge.select_dtypes(include=[np.number]).columns
correlation_matrix = merge[numeric_columns].corr()
plt.figure(figsize=(12, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='Blues')
plt.plot()
sns.pairplot(merge, corner=True)
plt.show()
mean = merge['mental_fitness'].mean()
mean
fig = px.pie(merge, values='mental_fitness', names='year')
fig.show()
fig = px.line(merge, x="year", y="mental_fitness", color='country', markers=True,color_discrete_sequence=['red', 'blue'], template='plotly_dark')
fig.show()
merge.info()
l = LabelEncoder()
for i in merge.columns:
    if merge[i].dtype == 'object':
        merge[i] = l.fit_transform(merge[i])
x = merge.drop('mental_fitness', axis=1)
y = merge['mental_fitness']
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=.20, random_state=2)
print("Xtrain: ", xtrain.shape)
print("Xtest: ", xtest.shape)
print("Ytrain: ", ytrain.shape)
print("Ytest: ", ytest.shape)
lr = LinearRegression()
lr.fit(xtrain, ytrain)
ytrain_pred = lr.predict(xtrain)
mse = mean_squared_error(ytrain, ytrain_pred)
rmse = (np.sqrt(mean_squared_error(ytrain, ytrain_pred)))
r2 = r2_score(ytrain, ytrain_pred)
print('MSE is {}'.format(mse))
print('RMSE is {}'.format(rmse))
print('R2 score is {}'.format(r2))
rf = RandomForestRegressor()
rf.fit(xtrain, ytrain)
ytrain_pred = rf.predict(xtrain)
mse = mean_squared_error(ytrain, ytrain_pred)
rmse = (np.sqrt(mean_squared_error(ytrain, ytrain_pred)))
r2 = r2_score(ytrain, ytrain_pred)
print('MSE is {}'.format(mse))
print('RMSE is {}'.format(rmse))
print('R2 score is {}'.format(r2))  # very good result as compared to linear
