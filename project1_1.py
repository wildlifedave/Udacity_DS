import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

df = pd.read_csv('./titanic.csv')
df.head()


#checking NaN values
df.isna().sum()


#filling NaN values for age and fare with median of that column
df[['age', 'fare']] = df[['age', 'fare']].fillna(df[['age', 'fare']].median())

#checking results



df.isna().sum()





#dropping irrelevant columns




df=df.drop(columns=['body'])





#creating dummy variables





df=pd.get_dummies(df, columns=['embarked', 'sex','pclass'])





#turning boat column into a binary. if they were on a boat 1. if not 0





df[['boat']]=((df.notnull()).astype('int'))[['boat']]





#checking results





df.head()





df.dtypes





#heatmap & chart too see any correlations





plt.subplots(figsize=(20,15))
sns.heatmap(df.corr(), annot=True)




#what factors influenced you survining?





df.corr()[['survived']].sort_values('survived', ascending=False)





#what factors influenced you being on a boat?




df.corr()[['boat']].sort_values('boat', ascending=False)





#what do some of these questions mean for a child?





df_children=df.loc[df['age'] < 18]
df_children





plt.subplots(figsize=(20,15))
sns.heatmap(df_children.corr(), annot=True)




df_children.corr()[['survived']].sort_values('survived', ascending=False)




df_adults=df.loc[df['age'] >= 18]
df_adults




#Survivor rate of children & Adults





len(df_children.loc[df_children['survived'] == 1])/len(df_children)




len(df_adults.loc[df_adults['survived'] == 1])/len(df_adults)




#Survivability not on a boat men vs. women




df_male=df.loc[df['sex_male']== 1]
df_female=df.loc[df['sex_female']== 1]
df_male_noboat=df_male.loc[df_male['boat']== 0]
df_female_noboat=df_female.loc[df_female['boat']== 0]
df_male_noboat_survived=df_male_noboat.loc[df_male_noboat['survived']==1]
df_female_noboat_survived=df_female_noboat.loc[df_female_noboat['survived']==1]





len(df_female_noboat_survived)/len(df_female)


len(df_male_noboat_survived)/len(df_male)


#survival rate is significantly lower for men when compared to women who did not make it to a boat.




#preparing for regression





#drop columns not needed




df_clean = df.drop(['cabin', 'home.dest', 'ticket','name'], axis=1)





list(df_clean.columns)





X = df_clean[['age', 'sibsp', 'parch', 'fare', 'boat', 'embarked_C', 'embarked_Q', 'embarked_S', 'sex_female', 'sex_male', 'pclass_1', 'pclass_2', 'pclass_3']]
y = df_clean[['survived']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=42)

#instantiate
#fit training
#predict test data
#score your model on test

lm_model = LinearRegression(normalize=True)
lm_model.fit(X_train, y_train)





X_train.shape





X_test.shape




print(lm_model.coef_)




y_pred = lm_model.predict(X_test)
df_preds = pd.DataFrame({'Actual': y_test.squeeze(), 'Predicted': y_pred.squeeze()})
print(df_preds)





from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print(f'Mean absolute error: {mae:.2f}')
print(f'Mean squared error: {mse:.2f}')
print(f'Root mean squared error: {rmse:.2f}')





r2_score(y_test, y_pred)





#the above results were the best of my 3 regression models.
#it had the best r2 score with 89 and a good mix of RMSE and MAE.
#this regrssion contained the most independent variables.
#but their coefficients don't necessarily make common sense.





#now just seeing how the model responds with a single indpendent vairable, boat





X = df_clean[['boat']]
y = df_clean[['survived']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=42)

#instantiate
#fit training
#predict test data
#score your model on test

lm_model = LinearRegression(normalize=True)
lm_model.fit(X_train, y_train)

y_pred = lm_model.predict(X_test)
df_preds = pd.DataFrame({'Actual': y_test.squeeze(), 'Predicted': y_pred.squeeze()})
print(df_preds)





from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print(f'Mean absolute error: {mae:.2f}')
print(f'Mean squared error: {mse:.2f}')
print(f'Root mean squared error: {rmse:.2f}')





r2_score(y_test, y_pred)





#results are pretty good. about the same but only SLIGHTLY worse.
#~88 r2 value and .05 and .17 on MAE and RMSE respecively.




#trying logicistic regression to see if it has any better RMSE or r2 results. 
#also to contain the coefficients better.




from sklearn.linear_model import LogisticRegression




X = df_clean[['age', 'sibsp', 'parch', 'fare', 'boat', 'embarked_C', 'embarked_Q', 'embarked_S', 'sex_female', 'sex_male', 'pclass_1', 'pclass_2', 'pclass_3']]
y = df_clean[['survived']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=42)

#instantiate
#fit training
#predict test data
#score your model on test

lg_model = LogisticRegression()
lg_model.fit(X_train, y_train)

y_pred = lg_model.predict(X_test)
df_preds = pd.DataFrame({'Actual': y_test.squeeze(), 'Predicted': y_pred.squeeze()})
print(df_preds)





from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print(f'Mean absolute error: {mae:.2f}')
print(f'Mean squared error: {mse:.2f}')
print(f'Root mean squared error: {rmse:.2f}')




r2_score(y_test, y_pred)





#the results above look pretty good. 
#r2 value of about 87-88 and RMSE and MAE are .17 and .03 respectively. 




print(lg_model.coef_)

