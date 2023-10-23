import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

path = '/content/advertising.csv'
df = pd.read_csv(path)

#Some EDA
df.head()
df.tail()
df.shape
df.describe()
df['Country'].nunique()
df['City'].nunique()
df.isnull().sum()
#There are no null values.


#Mean daily internet usage of consumers who clicked and did not click on ad
df.groupby('Clicked on Ad').mean()['Daily Internet Usage']


#Mean daily time spent on site by consumers who clicked and did not click on ad
df.groupby('Clicked on Ad').mean()['Daily Time Spent on Site']


#Data visualization

#Understanding how many consumers of different ages clicked on the ad
kids = (df['Age']>0) & (df['Age']<=18)
young = (df['Age']>18) & (df['Age']<=30)
midage = (df['Age']>30) & (df['Age']<=50)
old = (df['Age']>50)
clicked = [( np.sum(kids & df['Clicked on Ad'] == 1)), np.sum(young & df['Clicked on Ad'] ==1), np.sum(midage & df['Clicked on Ad'] ==1), np.sum(old & df['Clicked on Ad'] ==1) ]
did_not_click = [( np.sum(kids & df['Clicked on Ad'] == 0)), np.sum(young & df['Clicked on Ad'] ==0), np.sum(midage & df['Clicked on Ad'] ==0), np.sum(old & df['Clicked on Ad'] ==0)]

x_axis = np.arange(4)

plt.rcParams['figure.figsize'] = [5,5]
plt.bar(x_axis,clicked ,color = 'yellow',
        width = 0.25, edgecolor = 'black',
        label='Clicked on ad')
plt.bar(x_axis+0.25 ,did_not_click,color = 'red',
        width = 0.25, edgecolor = 'black',
        label='Did not click on ad')
plt.xticks(x_axis + 0.25/2,['Kids','Young','Midage','Old'])
plt.ylabel('Count')
plt.title('Age wise count of consumers who clicked and did not click on ad')
plt.legend()
#It can be concluded that middle-aged people have mostly clicked on the adds.



data = df.groupby(['Clicked on Ad','Male']).size()
women = [data[0][0],data[1][0]]
men = [data[0][1],data[1][1]]
x_axis = np.arange(2)
plt.rcParams['figure.figsize'] = [5, 5]
plt.bar(x_axis,women,color = 'g',
        width = 0.25, edgecolor = 'black',
        label='Women')
plt.bar(x_axis+0.25 ,men,color = 'b',
        width = 0.25, edgecolor = 'black',
        label='Men')
plt.xticks(x_axis + 0.25/2,['Did not click on ad','Clicked on ad'])
plt.ylabel('Count')
plt.title('Gender wise count of men and women who clicked and did not click on ad')
plt.legend()

#It can be concluded that men and women have almost equally clicked on ads, i.e., gender does not really affect the probability of a consumer to click on an add.

#Splitting input and output data
x = df[['Daily Time Spent on Site',	'Age',	'Area Income',	'Daily Internet Usage']]
x[0:10]
y = df['Clicked on Ad']
y[0:10]

#Splitting training and testing data
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,random_state = 0)
x_train.shape
x_test.shape

#Applying the Regressor
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()

#Fitting the model
model.fit(x_train,y_train)

#Predicting the output
y_pred = model.predict(x_test)
y_pred

y_test

#Checking the accuracy of the model

from sklearn.metrics import accuracy_score
accuracy_score(y_pred,y_test)*100

Individual prediction
model.predict([[71.33,	43,	62213.90,	241.09]])
The prediction implies that the consumer is likely to click on the data.
