#!/usr/bin/env python
# coding: utf-8

# # SUICIDE RATE PREDICTION

# The objective of this notebook is to predict the suicide rates using Machine Learning algorithms and analyzing them to find correlated factors causing increase in suicide rates globally.

# ### Steps Involved
# 
# Loading the data
# 
# Familiarizing with data
# 
# Visualizing the data
# 
# Data Preprocessing & EDA
# 
# Splitting the data
# 
# Training the data
# 
# Model Performance Comparision
# 
# Statistical Tests
# 
# Conclusion

# In[1]:


#importing required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from warnings import filterwarnings
filterwarnings(action='ignore')


# ### LOADING THE DATA
# The dataset is borrowed from Kaggle, https://www.kaggle.com/russellyates88/suicide-rates-overview-1985-to-2016. 
# The source of those datasets is WHO, World Bank, UNDP and a dataset published in Kaggle.
# The overview of this dataset is, it has 27820 samples with 12 features. 

# In[2]:


#Loading data into Datframe
data=pd.read_csv("C:\\Users\\Arindam\\Documents\\Downloads\\archive (12)\\master.csv")
data.head()


# ### FAMILARIZING WITH DATASET
# 
# Few dataframe methods are used to look into the data and its features.

# In[3]:


#shape of dataset
data.shape


# In[4]:


# renaming some columns for clarity
data.rename(columns={"suicides/100k pop":"suicide_rate","HDI for year":"HDI_for_year",
                  " gdp_for_year ($) ":"gdp_for_year"," gdp_per_capita ($) ":"gdp_per_capita",
                    "gdp_per_capita ($)":"gdp_per_capita"}, inplace=True)
print(data.columns)


# In[5]:


#features in dataset
data.columns


# In[6]:


#information about the dataset
data.info()


# In[7]:


#suicide count country wise
data['country'].value_counts()


# In[8]:


#suicide count age group wise
data['age'].value_counts()


# In[9]:


#suicide count generation wise
data['generation'].value_counts()


# In[10]:


#correlation matrix
data.corr()


# ### OBSERVATIONS
# 
# 1)HDI for year column has  many missing values therefore need to be removed. None of the other columns have any missing value.
# 
# 2) There are 6 unique age groups
# 
# 3) Age is grouped into year buckets as categorical format which needs to be encoded.
# 
# 4) Gender need be encoded.
# 
# 5) Scale required numerical features.
# 
# 6) The generation feature has 6 types of generations.
# 
# 7) Generation could be encoded as well.

# ### VISUALIZING THE DATA
# 
# Plots and graphs are used to see the distribution of data and relation between different features

# In[11]:


#correlation heatmap
plt.figure(figsize=(16,9))
sns.heatmap(data.corr(), annot=True)
plt.show()


# In[12]:


#histplot
data.hist(bins = 50,figsize = (15,11))


# In[13]:


#density plot
data.plot(kind ='density',subplots = True, layout =(3,2),figsize=(16,9),sharex = False)


# In[14]:


#gender vs Suicide count bar plot
plt.figure(figsize=(5,5))
sns.barplot(data.sex,data.suicides_no)
plt.xlabel("Gender")
plt.ylabel("Suicides count")
plt.title("Gender Vs Suicide Count")


plt.show()


# #### The above bar plot shows that the suicide cases are more in male population.

# In[15]:


#suicide count line plot of males and females year wise
plt.figure(figsize=(9,5))
data_men = data[data.sex == "male"]
data_women = data[data.sex == "female"]
sns.lineplot(data_men.year, data.suicides_no)
sns.lineplot(data_women.year, data.suicides_no)
plt.legend(["male", 'female'])
plt.show()


# #### It has been observed that in every year suicide cases are more in male population than in female

# In[16]:


# Age group vs suicide count
plt.figure(figsize=(9,5))
sns.barplot(x=data['age'], y=data['suicides_no'])
plt.xlabel('Age Group')
plt.ylabel('Suicide Count')
plt.title('Age Group Vs Suicide Count ')
plt.show()


# #### The above boxplot shows that the suicide cases are more in the age group of 35-54 years followed by 55- 74 years. 

# In[17]:


# Generation vs Suicide Count
plt.figure(figsize=(9,5))
sns.barplot(x=data['generation'], y=data['suicides_no'])
plt.xlabel('Generation')
plt.ylabel('Suicide Count')
plt.title('Generation Vs Suicide Count ')
plt.show()


# #### The above barplot shows that the suicide cases are more in the boomers, silent and X generations. These generations are made up of people born until 1976 based on the details provided.
# 

# In[18]:


#Suicide count vs Year  grouped by Generation
plt.figure(figsize=(16,9))
generation=['Generation X','Silent','Millenials','Boomers', 'G.I. Generation ','Generation Z']
for i in generation:
    data_gen= data[data.generation==i]
    sns.lineplot(data_gen.year,data_gen.suicides_no,ci=False)
    
plt.legend(['Generation X','Silent','Millenials','Boomers', 'G.I. Generation ','Generation Z'])
plt.title("Suicide_count Vs Year grouped by Generation")
plt.xlabel("Suicide Count")
plt.ylabel("Year")
plt.show()


# #### There were more suicides in boomer generation during 1990 to 2010. whereas before 1990 silent generation has more suicide count. it has been observes that the suicide rate of generation X has subsequently increased after 2010.The suicide count for G.I. Generation is least and morever suicides in this generation has been observed after 2007

# In[19]:


# Age group vs Suicide count grouped by gender
plt.figure(figsize=(10,3))
sns.barplot(x = "age", y = "suicides_no", hue = "sex", data = data)
plt.title("Age Group Vs Suicide Count Grouped by Gender")
plt.xlabel("Age Groups")
plt.ylabel("Suicides Count")
plt.legend()
plt.show()


# #### suicides in male population is more irrespective of age groups

# In[20]:


#Gender & Sucide Count grouped by Age Group bar plot

plt.figure(figsize=(7,7))
sns.barplot(y="sex", x="suicides_no", hue="age", data=data)
plt.title('Gender & Sucide Count grouped by Age Group')
plt.ylabel("Gender")
plt.xlabel("Suicide Count")
plt.show()


# #### From the above graph, we can infer that 35-54 years age group is more prone to suicides irrespective of the gender frollowed by 55-74 years age group.

# In[21]:


# Genration vs suicide count grouped by gender
plt.figure(figsize=(10,3))
sns.barplot(x = "generation", y = "suicides_no", hue = "sex", data = data)
plt.title("Generation Vs Suicide Count Grouped by Gender")
plt.xlabel("Generation")
plt.ylabel("Suicides Count")
plt.legend()
plt.show()


# ### males and females tend to commit suicide the most in boomer generation followed by silent and generation X.

# In[22]:


# generation vs suicide count grouped by generation
plt.figure(figsize=(7,7))
sns.barplot(y="sex", x="suicides_no", hue="generation", data=data)
plt.title('Gender & Sucide Count grouped by Generation')
plt.ylabel("Gender")
plt.xlabel("Suicide Count")
plt.show()


# #### Even when considered generation, males are more prone to commit suicide.

# In[23]:


#Country & Suicide_rate Bar plot
plt.figure(figsize=(15,25))
sns.barplot(x = "suicide_rate", y = "country", data = data)
plt.title('Country - Suicide_rate Bar plot')
plt.show()


# #### The above plot shows that highes suicide rate country is Lithuania followed by Srilanka

# In[24]:


#Top 10 countries with highest number of suicides
plt.figure(figsize=(10,9))
data.groupby('country').sum().suicides_no.sort_values(ascending=False).head(10).plot(kind='bar',figsize=(8,2))
plt.xticks(rotation=30,fontsize=9)
plt.xlabel('Countries')
plt.title('Top 10 countries with highest no of suicides')
plt.show()


# In[25]:


#Top 10 countries with lowest number of suicides
data.groupby('country').sum().suicides_no.sort_values(ascending=False).tail(10).plot(kind='bar',figsize=(8,2),color="#524636")
plt.xticks(rotation=30,fontsize=9)
plt.xlabel('Countries')
plt.title('Top 10 countries with lowest number of suicides')
plt.show()


# In[26]:


# year vs suicide rate
data[['year','suicide_rate']].groupby(['year']).sum().plot()


# #### The observations from the above plot are that the suicide rate had grown rapidly from year 1990 & the rate of suicide has drastically reduced in year 2016. The dataset was collected during early 2016. So all the suicide cases of 2016 are not recorded in the dataset.

# In[27]:


# country vs gdp per capita barplot
plt.figure(figsize=(15,25))
sns.barplot(x = "gdp_per_capita", y = "country", data = data)
plt.title('Country - gdp per capita Bar plot')
plt.show()


# #### Luxembourg has the highest gdp per capita followed by qafar

# In[28]:


# GDP per capita vs Suicide Count
plt.figure(figsize=(16,9))
plt.scatter(x=data.suicides_no,y=data.gdp_per_capita)
plt.xlabel("Suicide Count")
plt.ylabel("GDP per capita")
plt.title("GDP per capita vs Suicide Count ")
plt.show()


# #### Suicide Rate is inversely proportional to GDP per Capita

# In[29]:


# Population vs Suicide Count
plt.figure(figsize=(16,9))
plt.scatter(x=data.population,y=data.suicides_no)
plt.xlabel("Population")
plt.ylabel("Suicide Count")
plt.title("Population vs Suicide Count ")
plt.show()


# In[30]:


#scatter matrix for checking outlier
plt.figure(figsize=(20,10))
attributes = ['suicides_no', 'population', 'suicide_rate','HDI_for_year', 
              'gdp_for_year','gdp_per_capita']
pd.plotting.scatter_matrix(data[attributes], figsize=(20,10))
plt.show()


# In[31]:


#Boxplot for the outliers
data.plot(kind='box',subplots=True,figsize=(16,9),layout=(2,3))


# ### DATA PREPROCESSING AND EDA
# Here, we clean the data by applying data preprocesssing techniques and transform the data to use it in the models.

# In[32]:


data.describe()


# In[33]:


#checking for null values
data.isnull().sum()


# In[34]:


#dropping HDI column due to null values
data = data.drop(['HDI_for_year'], axis = 1)
data.shape


# In[35]:


#The column country-year is just a combination of country and year columns. So dropping that column
data = data.drop(['country-year'], axis = 1)
data.shape


# In[36]:


data.columns


# In[37]:


#dropping off null rows if any
data = data.dropna()
data.shape


# The non-numerical labeled columns, country, year, gender, age_group and generation are to be converted to numerical labels that can be don by using SkLearn's LabelEncoder.

# In[38]:


from sklearn.preprocessing import LabelEncoder


# In[39]:


#encoding the categorical features with LabelEncoder
categorical = ['country', 'year','age', 'sex', 'generation']
le = LabelEncoder()

for column in categorical:
    data[column] = le.fit_transform(data[column])


# In[40]:


#creating a copy of dataset for statistical test
copy_data=data.copy


# In[41]:


#checking the datatypes
data.dtypes


# In[42]:


# Converting the column 'gdp_for_year' to float from object
data['gdp_for_year'] = data['gdp_for_year'].str.replace(',','').astype(float)


# In[43]:


data.dtypes


# In[44]:


#Scaling the numerical data columns with RobustScalar
numerical = ['suicides_no', 'population', 'suicide_rate', 
              'gdp_for_year','gdp_per_capita']

from sklearn.preprocessing import RobustScaler

rc = RobustScaler()
data[numerical] = rc.fit_transform(data[numerical])


# In[45]:


data.head(15)


# ### SPLITTING THE DATASET

# In[46]:


#separating data and target column
Y = data['suicide_rate']
X = data.drop('suicide_rate',axis=1)


# In[47]:


X.shape


# In[48]:


Y.shape


# In[49]:


# Splitting the dataset into train and test sets: 80-20 split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 12)
X_train.shape, X_test.shape


# In[50]:


X_train.shape, X_test.shape
y_train.shape, y_test.shape


# ### Model building and Training
#  Supervised machine learning is one of the most commonly used and successful types of machine learning. Supervised learning is used whenever we want to predict a certain outcome/label from a given set of features, and we have examples of features-label pairs. We build a machine learning model from these features-label pairs, which comprise our training set. Our goal is to make accurate predictions for new, never-before-seen data.
#     
#      There are two major types of supervised machine learning problems, called classification and regression. Our data set comes under regression problem, as the prediction of suicide rate is a continuous number, or a floating-point number in programming terms. The supervised machine learning models (regression) considered to train the dataset in this notebook are:
# 
# Linear Regression
# 
# Polynomial Regression
# 
# Decision Tree
# 
# Random Forest
# 
# Gradient Boosting
# 
# XGBoost
# 
# Bagging Regression
# 
# MultiLayer Perceptrons
#              .

# In[51]:


# Creating holders to store the model performance results
ML_Model = []
acc_train = []
acc_test = []
rmse_train = []
rmse_test = []
mse_train=[]
mse_test=[]
#function to call for storing the results
def storeResults(model, a,b,c,d,e,f):
  ML_Model.append(model)
  acc_train.append(round(a, 3))
  acc_test.append(round(b, 3))
  rmse_train.append(round(c, 3))
  rmse_test.append(round(d, 3))
  mse_train.append(round(e, 3))
  mse_test.append(round(f, 3))


# In[52]:


#importing requiredlibraries
from sklearn.metrics import r2_score, mean_squared_error


# In[53]:


# function for fitting the model to datasets and calculating the metrics
def function(modelname,model):
    
    model.fit(X_train, y_train)
    y_train_model = model.predict(X_train)
    y_test_model = model.predict(X_test)
    acc_train_model = r2_score(y_train, y_train_model)
    acc_test_model = r2_score( y_test,y_test_model)
    mse_train_model = mean_squared_error(y_train, y_train_model)
    mse_test_model = mean_squared_error(y_test, y_test_model)
    rmse_train_model = np.sqrt(mean_squared_error(y_train, y_train_model))
    rmse_test_model= np.sqrt(mean_squared_error(y_test, y_test_model))
    print('MSE train data: {:.5}           MSE test data: {:.5}'.format( mse_train_model, mse_test_model))
    print('RMSE train data: {:.5}          RMSE test data: {:.5}'.format(rmse_train_model,rmse_train_model))
    print('Accuracy train data: {:.5}            Accuracy test data: {:.5}'.format(acc_train_model,acc_test_model))
    storeResults(modelname, acc_train_model, acc_test_model, rmse_train_model, rmse_test_model,mse_train_model,mse_test_model )


# ### 1) Linear Regresssion

# In[54]:


from sklearn.linear_model import LinearRegression


# In[55]:


Linear_Regression=LinearRegression();
function("Linear Regression",Linear_Regression)


# In[56]:


from sklearn.model_selection import cross_val_score
lr_cv = cross_val_score(LinearRegression(), X, Y, cv = 7)
# accuracy +/- 2 standard deviations
print("Accuracy: {:.2} (+/- {:.2})".format(lr_cv.mean(), lr_cv.std() * 2)) 


# #### The model preformance is not very good, but we can see that the scores on the training and test sets are very close together. This means we are likely underfitting, not overfitting.

# ### 2) Polynomial Regression

# In[57]:


from sklearn.preprocessing import PolynomialFeatures

X_poly =  PolynomialFeatures(degree = 2).fit_transform(X)

X_train_poly, X_test_poly, y_train_poly, y_test_poly = train_test_split( X_poly, Y, test_size=0.1, random_state=42)


# In[58]:


pr = LinearRegression()
function("Polynomial Regression" ,pr)


# #### The result is not so good even afer introducing polynomial features

# ### 3)Decision Tree Regression

# In[59]:


# Decision Tree regression model 
from sklearn.tree import DecisionTreeRegressor

# instantiate the model 
DecisionTreeRegression= DecisionTreeRegressor(max_depth=9)
# fit the model 
function("Decision Tree Regression",DecisionTreeRegression)


# In[60]:


training_accuracy = []
test_accuracy = []
# try max_depth from 1 to 30
depth = range(1, 31)
for n in depth:
  # fit the model
  tree = DecisionTreeRegressor(max_depth=n)
  tree.fit(X_train, y_train)
  # record training set accuracy
  training_accuracy.append(tree.score(X_train, y_train))
  # record generalization accuracy
  test_accuracy.append(tree.score(X_test, y_test))

#plotting the training & testing accuracy for max_depth from 1 to 30
plt.plot(depth, training_accuracy, label="training accuracy")
plt.plot(depth, test_accuracy, label="test accuracy")
plt.ylabel("Accuracy")  
plt.xlabel("max_depth")
plt.legend()


# #### OBSERVATIONS: The model preformance is gradually increased on incresing the max_depth parameter. But after max_depth = 9, the model overfits. So the model is considered with max_depth = 9 

# ### RANDOM FOREST REGRESSION

# In[61]:


# Random Forest regression model
from sklearn.ensemble import RandomForestRegressor

# instantiate the model
RandomForestRegression = RandomForestRegressor(max_depth=9)
function("RandomForestRegression",RandomForestRegression )


# In[62]:


training_accuracy = []
test_accuracy = []
# try max_depth from 1 to 30
depth = range(1, 31)
for n in depth:
  # fit the model
  forest = RandomForestRegressor(max_depth=n)
  forest.fit(X_train, y_train)
  # record training set accuracy
  training_accuracy.append(forest.score(X_train, y_train))
  # record generalization accuracy
  test_accuracy.append(forest.score(X_test, y_test))

#plotting the training & testing accuracy for max_depth from 1 to 30
plt.plot(depth, training_accuracy, label="training accuracy")
plt.plot(depth, test_accuracy, label="test accuracy")
plt.ylabel("Accuracy")  
plt.xlabel("max_depth")
plt.legend()


# #### The random forest gives us an accuracy of 98.7% on training data and 98.03% on test data that is  better than the linear models or a single decision tree.

# ### Gradient Boosting Regression 

# In[63]:


# Gradient Boosted Regression Trees model
from sklearn.ensemble import GradientBoostingRegressor

# instantiate the model
gbrt = GradientBoostingRegressor(learning_rate=0.5)

# fit the model 
function("GradientBoostingRegressor",gbrt)


# In[64]:


training_accuracy = []
test_accuracy = []
r = []
# try learning_rate from 0.1 to 0.9
rate = range(1, 10)
for n in rate:
  # fit the model
  gbrt = GradientBoostingRegressor(learning_rate=n*0.1)
  gbrt.fit(X_train, y_train)
  r.append(n*0.1)
  # record training set accuracy
  training_accuracy.append(gbrt.score(X_train, y_train))
  # record generalization accuracy
  test_accuracy.append(gbrt.score(X_test, y_test))

#plotting the training & testing accuracy for learning_rate from 0.1 to 0.9

plt.plot(r, training_accuracy, label="training accuracy")
plt.plot(r, test_accuracy, label="test accuracy")
plt.ylabel("Accuracy")  
plt.xlabel("learning_rate")
plt.legend()


# #### OBSERVATIONS: The model provides a good accuracy
# 
# #### Hyper tuning is performed for Gradient Boosted Regression Tree model. The tuned parameters are learning_rate, n_estimators & max_depth. Even upon changing the n_estimators with the combination of other two, there is no change in the model performance.
# 

# ### XGBoost Regression

# In[65]:


from xgboost import XGBRegressor

# instantiate the model
xgb = XGBRegressor(learning_rate=0.2,max_depth=4)
#fit the model
function("XGBRegressor",xgb)


# #### XGboost Regression provides a good accuracy score of 98.8 % on test data and  99.3%on training data

# ### Bagging Regression

# In[66]:


from sklearn.ensemble import BaggingRegressor

#instantiate the model
br = BaggingRegressor(n_estimators=1)

#fit the model
function("BaggingRegressor",br)


# Evaluating training and testing set performance with different numbers of n_estimators from 1 to 50. The plot shows the training and test set accuracy on the y-axis against the setting of n_estimators on the x-axis.

# In[67]:


training_accuracy = []
test_accuracy = []
# try n_estimators from 1 to 50
est = range(1, 51)
for n in est:
  # fit the model
  br = BaggingRegressor(n_estimators=n)
  br.fit(X_train, y_train)
  # record training set accuracy
  training_accuracy.append(br.score(X_train, y_train))
  # record generalization accuracy
  test_accuracy.append(br.score(X_test, y_test))

#plotting the training & testing accuracy for n_estimators from 1 to 50
plt.plot(est, training_accuracy, label="training accuracy")
plt.plot(est, test_accuracy, label="test accuracy")
plt.ylabel("Accuracy")  
plt.xlabel("n_estimators")
plt.legend()


# #### From the above plot, it is clear that the model performs very well on this dataset. Even with tuning of n_estimators parameters, the training accuracy always stayed above 99% & the test data accuracy is always above 97%. This may or may not be the case of overfitting.

# ### MultiLayer Perceptrons

# In[68]:


# Multilayer Perceptrons model
from sklearn.neural_network import MLPRegressor

# instantiate the model
mlp = MLPRegressor(hidden_layer_sizes=([100,100]))
function("MLPRegressor",mlp)


# ### COMPARISION OF MODELS

# In[69]:


#creating a dataframe for storing the metrics result
results = pd.DataFrame({ 'ML Model': ML_Model,    
    'Train Accuracy': acc_train,
    'Test Accuracy': acc_test,
    'Train RMSE': rmse_train,
    'Test RMSE': rmse_test,
    'Mse_train':  mse_train,'mse test':
mse_test})


# In[70]:


results


# In[71]:


results.sort_values(by=['Test Accuracy', 'Train Accuracy'], ascending=False)


# ### Among all the trained models, XGBoost performance is better. It is understandable because this model is very good in execution Speed & model performance.

# ## Statistical test
# 
# #### Statistical tests are used in hypothesis testing. They can be used to:determine whether a predictor variable has a statistically significant relationship with an outcome variable.
# 
# #### estimate the difference between two or more groups

# In[72]:


#importig required libraries
from scipy import stats


# ###  Test 1: To check the difference in suicide rates between male and female
# Using independent sample t-test to check the difference in suicide rates between male and female. The hypothesis statements for this test are: 
# 
# **H0:** There is no difference in the suicide rates among male and female (Null).<br>
# **H1:** There is difference in the suicide rates among male and female (Alternate).

# In[73]:


#collecting male suicide rate data
male = data['suicide_rate'][data['sex'] == 1]
male


# In[74]:


#collecting female suicide rate data
female = data['suicide_rate'][data['sex'] == 0]
female


# In[75]:


#calculating p value
ttest,pval = stats.ttest_rel(male, female)


# In[76]:


if pval<0.05:
    print("Reject null hypothesis")
else:
    print("Accept null hypothesis")


# ### **TEST CONCLUSION**
# #### By performing T-test, the result obtained is to reject the null hypothesis. This basically means that there isdifferent in suicide rates of male & female.

# ### Test 2: To find out the dependence of suicide rate on the age.
# Finding out whether there is a dependence of suicide rate on the age using the Chi- Square test. The hypothesis statements for this test are: 
# 
# **H0:** Suicide rate and age are independent (Null).<br>
# **H1:** Suicide rate and age are dependent (Alternate). 

# In[77]:


#creating contingency Table
contingency_table = pd.crosstab(data.suicide_rate, data.age)


# In[78]:


contingency_table.head()


# In[79]:


#Significance level 5%
alpha=0.05


# In[80]:


chistat, p, dof, expected = stats.chi2_contingency(contingency_table )


# In[81]:


#critical_value
critical_value=stats.chi2.ppf(q=1-alpha,df=dof)
print('critical_value:',critical_value)


# In[82]:


print('Significance level: ',alpha)
print('Degree of Freedom: ',dof)
print('chi-square statistic:',chistat)
print('critical_value:',critical_value)
print('p-value:',p) 
#Here, pvalue = 0.0 and a low pvalue suggests that your sample provides enough evidence that you can reject  H0  for the entire population.


# In[83]:


#compare chi_square_statistic with critical_value and p-value which is the 
 #probability of getting chi-square>0.09 (chi_square_statistic)
if chistat>=critical_value:
    print("Reject H0,There is a dependency between Age group & Suicide rate.")
else:
    print("Retain H0,There is no relationship between Age group & Suicide rate.")
    
if p<=alpha:
    print("Reject H0,There is a dependency between Age group & Suicide rate.")
else:
    print("Retain H0,There is no relationship between Age group & Suicide rate.")


# ### Test Conclusion:
# 
# #### By performing Chi- Square test, the result obtained is to reject the null hypothesis. This basically means that there is dependency between Age group & Suicide rate.

# ### CONCLUSION

# ### *Understand the working of different Machine Learning Models on the Dataset and understanding their parameters,how to tune the and how they affect the model performance.*
# ### *The final conclusion on the suicide dataset are that the irrespective of age group and generation, male population are more prone to commit suicide than female.*

# In[ ]:





# In[ ]:




