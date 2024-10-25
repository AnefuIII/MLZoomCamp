#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

import seaborn as sns
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import pickle


# ### Data Preparation

# In[2]:


#pip install kagglehub


# In[3]:


# import kagglehub

# # Download latest version
# path = kagglehub.dataset_download("blastchar/telco-customer-churn")

# print("Path to dataset files:", path)


# In[4]:


data = 'WA_Fn-UseC_-Telco-Customer-Churn.csv'


# In[5]:


df = pd.read_csv(data)
df.head()


# In[6]:


df.head().T


# #### Preparing Column names and Values to get rid of inconsistent Alphabet cases to all lower case and space bars to _

# In[7]:


# Columns
df.columns = df.columns.str.lower().str.replace(' ', '_')

df.head().T


# In[8]:


# values

string_values =list(df.dtypes[df.dtypes == 'object'].index)

for col in string_values:
    df[col] = df[col].str.lower().str.replace(' ', '_')


# In[9]:


df.head().T


# In[10]:


#
df.dtypes


# In[11]:


# total charges should be numeric but it is object data type

tc = pd.to_numeric(df.totalcharges, errors = 'coerce')


# In[12]:


# missing values in tc
tc.isnull().sum()


# In[13]:


# view data frame of null totalcharges

df[tc.isnull()][['customerid', 'totalcharges']]


# In[14]:


df.totalcharges.isnull().sum()


# In[15]:


# view data frame of null totalcharges

df['totalcharges']=df.totalcharges.replace('_',np.nan).astype(float)


# In[16]:


df.totalcharges.isnull().sum()


# In[17]:


#fill the missing values

df.totalcharges = df.totalcharges.fillna(0)


# In[18]:


df.totalcharges.isnull().sum()


# In[19]:


# Check if all columns are read correctly

(df.churn == 'yes').head()


# In[20]:


#convert to numbers

df.churn = (df.churn == 'yes').astype(int)


# In[21]:


df.churn.head()


# ### Setup validation framework

# In[22]:


from sklearn.model_selection import train_test_split


# In[23]:


df_full_train, df_test = train_test_split(df, test_size = 0.2, random_state = 1)


# In[24]:


print(f'lenght of full train is {len(df_full_train)},\nlenght of test is {len(df_test)}')


# In[25]:


# further divide the full train to obtain 20 percent for the validation
# 25% of 80% is 20% of the original dataset length

df_train, df_val = train_test_split(df_full_train, test_size = 0.25, random_state = 1)


# In[26]:


print(f'lenght of full train is {len(df_full_train)},\nlenght of test is {len(df_test)}\n'
      f'length of validation is {len(df_val)}')


# In[27]:


df_train.head()


# In[28]:


# reset index because the elements are now shuffled

df_train = df_train.reset_index(drop = True)
df_val = df_val.reset_index(drop = True)
df_test = df_test.reset_index(drop = True)


# In[29]:


df_train.head()


# In[30]:


# target values
y_train = df_train.churn.values
y_val = df_val.churn.values
y_test = df_test.churn.values


# In[31]:


#Delet churn variable from the data frame

del df_train['churn']
del df_val['churn']
del df_test['churn']


# In[32]:


df_train.shape


# ### EDA
# 
# use df_full_train for EDA

# In[33]:


df_full_train.isnull().sum()


# In[34]:


df.duplicated().sum()


# ##### Churn rate: for categorical variables is the mean because the number of ones is the sum of the record

# In[35]:


df_full_train.churn.value_counts()


# In[36]:


df_full_train.churn.value_counts(normalize = True)


# In[37]:


df_full_train.churn.mean()


# In[38]:


global_churn_rate = df.churn.mean()


# In[39]:


round(global_churn_rate, 2)


# In[40]:


df_full_train.columns


# In[41]:


numerical = ['tenure', 'monthlycharges', 'totalcharges' ]

categorical = [ 'gender', 'seniorcitizen', 'partner', 'dependents',
       'phoneservice', 'multiplelines', 'internetservice',
       'onlinesecurity', 'onlinebackup', 'deviceprotection', 'techsupport',
       'streamingtv', 'streamingmovies', 'contract', 'paperlessbilling',
       'paymentmethod']


# In[42]:


df_full_train[categorical].nunique()


# ### Churn rate analysis

# In[43]:


churn_female = df_full_train[df_full_train['gender'] == 'female'].churn.mean()
churn_female


# In[44]:


churn_male = df_full_train[df_full_train['gender'] == 'male'].churn.mean()
churn_male


# In[45]:


print(f'Churn rate for males {global_churn_rate - churn_male}\nChurn rate female {global_churn_rate - churn_female}')


# In[46]:


churn_no_partner = df_full_train[df_full_train['partner'] == 'no'].churn.mean()
churn_no_partner


# In[47]:


churn_partner = df_full_train[df_full_train['partner'] == 'yes'].churn.mean()
churn_partner


# In[48]:


print(f'Churn rate with partner {global_churn_rate - churn_partner}\nChurn rate without {global_churn_rate - churn_male}')


# ##### Observation: churn rates for groups should be close to the global churn rate. if they are far, irrespective of the signs,they are more likely to churn. In order words, if the group churn rate is higher, then they are less likely to churn, if it is lower, they are more likely to churn

# ### Churn Risk Ratio
# 
# Risk ratio greater than 1 are more likely to churn, less than 1 are less likely to churn

# In[49]:


churn_partner/global_churn_rate


# In[50]:


churn_no_partner/global_churn_rate


# ##### Risk Ratios greater than 1 are more likely to churn, those less than 1 are less likely to churn

# In[51]:


df_group = df_full_train.groupby('gender').churn.agg(['mean', 'count']) #agg takes a list of aggregations we can perform
df_group['diff'] = df_group['mean'] - global_churn_rate
df_group['risk'] = df_group['mean'] / global_churn_rate


# In[52]:


df_group


# In[53]:


# for all categorical vriables
from IPython.display import display


for c in categorical:
    
    print(c)
    df_group = df_full_train.groupby(c).churn.agg(['mean', 'count']) #agg takes a list of aggregations we can perform
    df_group['diff'] = df_group['mean'] - global_churn_rate
    df_group['risk'] = df_group['mean'] / global_churn_rate
    display(df_group)
    print()
    print()


# ### Mutual Information

# In[54]:


from sklearn.metrics import mutual_info_score


# In[55]:


mutual_info_score(df_full_train.churn, df_full_train.contract )


# In[56]:


mutual_info_score(df_full_train.churn, df_full_train.gender)


# In[57]:


mutual_info_score(df_full_train.churn, df_full_train.partner)


# In[58]:


for c in categorical:
    mis = mutual_info_score(df_full_train.churn, df_full_train[c] )
    #print(f'{c} : {mis.sort()}')
    print(c, ' ', mis)


# In[59]:


# we can use Apply

def mutual_info_churn_score(series):
    return mutual_info_score(df_full_train.churn, series)

df_full_train[categorical].apply(mutual_info_churn_score)


# In[60]:


mi = df_full_train[categorical].apply(mutual_info_churn_score)
mi.sort_values(ascending = False)


# ### Correlation

# In[61]:


df_full_train[numerical].corrwith(df_full_train.churn)


# In[62]:


df_full_train[numerical].corrwith(df_full_train.churn).abs()


# In[63]:


df_full_train[df_full_train.tenure <= 2].churn.mean()


# In[64]:


df_full_train[df_full_train.tenure > 2].churn.mean()


# In[65]:


df_full_train[(df_full_train.tenure > 2) & (df_full_train.tenure <= 12)].churn.mean()


# In[66]:


df_full_train[df_full_train.tenure > 12].churn.mean()


# In[67]:


df_full_train[df_full_train.monthlycharges <= 20].churn.mean()


# In[68]:


df_full_train[(df_full_train.monthlycharges > 20) & (df_full_train.monthlycharges <= 50)].churn.mean()


# In[69]:


df_full_train[df_full_train.monthlycharges > 50].churn.mean()


# ### One Hot Encoding

# In[70]:


from sklearn.feature_extraction import DictVectorizer


# In[71]:


df_train[['gender', 'contract']].iloc[:10]


# In[72]:


df_train[['gender', 'contract']].iloc[:20].to_dict(orient = 'records')


# In[73]:


dicts = df_train[['gender', 'contract']].iloc[:500].to_dict(orient = 'records')


# In[74]:


dv = DictVectorizer(sparse = False)


# In[75]:


dv.fit(dicts)


# In[76]:


dv.transform(dicts)


# In[77]:


dv.feature_names_


# In[78]:


# Another way to get the feature names

dv.get_feature_names_out()


# In[79]:


train_dict = df_train[categorical + numerical].to_dict(orient = 'records') 
train_dict[0]


# In[80]:


dv.fit(train_dict)


# In[81]:


dv.transform(train_dict[:1])


# In[82]:


list(dv.transform(train_dict[:5])[0])


# In[83]:


# Another way to get the feature names

dv.feature_names_


# In[84]:


train_dict = df_train[categorical + numerical].to_dict(orient = 'records')

dv = DictVectorizer(sparse = False)

X_train = dv.fit_transform(train_dict)

X_train.shape


# In[85]:


val_dict = df_val[categorical + numerical].to_dict(orient = 'records')

#dv = DictVectorizer(sparse = False)

X_val = dv.transform(val_dict)

X_val.shape


# In[86]:


X_val


# ### Sigmoid for logistic regression

# In[87]:


def sigmoid(z):
    return 1/(1 + np.exp(-z))


# In[88]:


sigmoid(-100)


# In[89]:


z = np.linspace(-7 ,7, 51)


# In[90]:


z


# In[91]:


plt.plot(z, sigmoid(z))


# In[92]:


def LinReg(xi):
    
    result = w0
    
    for j in range(len(xi)):
        result = w0 + xi[j] * w[j]
        
    return result


# In[93]:


def LogReg(xi):
    
    result = w0
    
    for j in range(len(xi)):
        result = w0 + xi[j] * w[j]
        
    return sigmoid(result)


# ### Logistic Regression using Scikit-Learn

# In[94]:


from sklearn.linear_model import LogisticRegression


# In[95]:


model = LogisticRegression(solver='lbfgs', max_iter=1000)

model.fit(X_train, y_train)


# In[96]:


# Hard predict

model.predict(X_val)


# In[97]:


# Soft predict. the both columns are for the negative and positive class
# we are interested in the POSITIVE class

model.predict_proba(X_val)


# In[98]:


#weights

model.coef_


# In[99]:


model.coef_[0].round(3) #.shape


# In[100]:


#model.coef_.shape


# In[101]:


# y intercept
model.intercept_[0]


# In[102]:


# Soft predict
y_pred1 = model.predict_proba(X_val)[:, 1]
y_pred1


# In[103]:


# we can use a default threshold
y_pred1 >= 0.5


# In[104]:


churn_decision1 = y_pred1 >= 0.5


# In[105]:


churn_decision1


# In[106]:


# potential churn customers

df_val[churn_decision1].customerid


# In[107]:


churn_decision1.astype('int')


# In[108]:


# Accuracy

(y_val == churn_decision1.astype('int')).mean()


# In[109]:


# Accuracy

(y_val == churn_decision1.astype('int')).mean()


# In[110]:


df_pred = pd.DataFrame()
df_pred['probability'] = y_pred1
df_pred['prediction'] = churn_decision1.astype('int')
df_pred['actual'] = y_val

df_pred


# In[111]:


df_pred['correct'] = df_pred.prediction == df_pred.actual
df_pred


# ### Model intepretation

# In[112]:


a = [1,2,3]
b = 'abc'

zip(a,b)


# In[113]:


list(zip(a,b))


# In[114]:


dict(zip(a,b))


# In[115]:


list(zip(dv.feature_names_ , model.coef_[0].round(3)))


# In[116]:


small = ['contract', 'tenure', 'monthlycharges']


# In[117]:


df_small = df_train[small][:10]
df_small.head(2)


# In[118]:


dict_train_small = df_train[small].to_dict(orient = 'records')
dict_val_small = df_val[small].to_dict(orient = 'records')


# In[119]:


dv_small = DictVectorizer(sparse = False)
dv_small.fit(dict_train_small)


# In[120]:


dv_small.feature_names_


# In[121]:


X_train_small = dv_small.transform(dict_train_small)


# In[122]:


model_small = LogisticRegression(solver='lbfgs', max_iter=1000)

model_small.fit(X_train_small, y_train)


# In[123]:


w0 = model_small.intercept_[0]
w0


# In[124]:


w = model_small.coef_[0]
w.round(3)


# In[125]:


dict(zip(dv.feature_names_, w.round(3)))


# ### Train Final Model and test with test data

# In[126]:


dicts_full_train = df_full_train[categorical + numerical].to_dict(orient = 'records')


# In[127]:


X_full_train = DictVectorizer(sparse = False).fit_transform(dicts_full_train)


# In[128]:


y_full_train = df_full_train.churn.values
y_full_train


# In[129]:


model = LogisticRegression(solver='lbfgs', max_iter=1000 )
model.fit(X_full_train, y_full_train)


# In[130]:


# TEST DATA
dicts_test = df_test[categorical + numerical].to_dict(orient = 'records')

X_test = dv.transform(dicts_test)


# In[131]:


y_pred = model.predict_proba(X_test)[:, 1]


# In[132]:


y_pred


# In[133]:


churn_decision = (y_pred >= 0.5)


# In[134]:


churn_decision


# In[135]:


(churn_decision == y_test).mean()


# ### Using our final model

# In[136]:


customer = dicts_test[10]
customer


# In[137]:


customer = dv.transform([customer])
customer


# In[138]:


model.predict_proba(customer)[0, 1]


# In[139]:


# check
y_test[10]


# In[140]:


y_test


# In[141]:


customer2 = dicts_test[-1]
customer2


# In[142]:


customer2 = dv.transform([customer2])
customer2


# In[143]:


model.predict_proba(customer2)[0, 1]


# In[144]:


# check
y_test[-1]


# ### Evaluation

# ### Accuracy score
# 
# NOTE: USING FIRST MODEL OR MODEL TRAINED WITH FULL DATA

# In[145]:


len(y_val)


# In[146]:


(y_val == churn_decision1).mean()


# In[147]:


(y_val == churn_decision1).sum()


# In[148]:


1132/1409


# In[149]:


thresholds = np.linspace(0, 1, 21)
thresholds


# In[150]:


scores = []

for t in thresholds:
    churn_decision = (y_pred >= t)
    score = (y_val == churn_decision).mean()
    print('%.2f: %.3f' % (t, score))
    scores.append(score)


# In[151]:


plt.plot(thresholds, scores)


# In[152]:


#using scikitlearn

from sklearn.metrics import accuracy_score

accuracy_score(y_val, y_pred>= 0.5)


# In[153]:


scores = []

for t in thresholds:
    score = accuracy_score(y_val, y_pred>= t)
    print('%.2f: %.3f' % (t, score))
    scores.append(score)


# In[154]:


# How many false and how many true values

from collections import Counter


# In[155]:


Counter(y_pred >= 1)


# In[156]:


Counter(y_val)


# In[157]:


#Churning
(y_val).mean()


# In[158]:


#Not churning
1 - (y_val).mean()


# #### The issue with accuracy score is that it is poor when there is class imbalance

# ### Confusion Matrix

# In[159]:


actual_positive = (y_val == 1)
actual_negative = (y_val == 0)


# In[160]:


t = 0.5
predict_positive = (y_pred1 >= t)
predict_negative = (y_pred1 < t)


# In[161]:


tp = (actual_positive & predict_positive).sum()
tn = (actual_negative & predict_negative).sum()

print(tp, tn)


# In[162]:


fp = (predict_positive & actual_negative).sum()
fn = (predict_negative & actual_positive).sum()

print(fp, fn)


# In[163]:


confusion_matrix = np.array([
    [tn, fp],
    [fn, tp]
])

confusion_matrix


# In[164]:


(confusion_matrix/confusion_matrix.sum()).round(2)


# ### Precision
# 
# #### Precision is the fraction of positive PREDICTIONS that are correct

# In[165]:


accuracy = (tp + tn) / (tp + fp + tn + fn)
accuracy.round(2)


# In[166]:


precision = tp / (tp + fp)
precision


# In[167]:


tp, tp + fp


# #### Observation: approx 33%, 319 customers were incorrectly classified as churn and sent discount

# ### Recall
# #### Recall is the fraction of correctly classified (ACTUAL) positive examples. 

# In[168]:


Recall = tp / (tp + fn)
Recall


# In[169]:


tp, (tp + fn)


# #### Observation: We failed to Identify 55 percent of people who are churning

# ### ROC Curves
# 
# #### Receiver Operating Characteristics checks for all thresholds: True positive Rate (same as Recall) and False Positive rate

# In[170]:


tpr = tp / (tp + fn)
tpr


# In[171]:


fpr = fp / (fp + tn)
fpr


# In[172]:


thresholds = np.linspace(0, 1, 101)

scores = []

for t in thresholds:
    #Actual values
    actual_positive = (y_val == 1)
    actual_negative = (y_val == 0)
    
    #Prediction values
    predict_positive = (y_pred1 >= t)
    predict_negative = (y_pred1 < t)
    
    tp = (actual_positive & predict_positive).sum()
    tn = (actual_negative & predict_negative).sum()

    fp = (predict_positive & actual_negative).sum()
    fn = (predict_negative & actual_positive).sum()
    
    scores.append((t, tp, fp, tn, fn))
    
scores


# In[173]:


pd.DataFrame(scores).head()


# In[174]:


columns = ['threshold', 'tp', 'fp', 'tn', 'fn']

df_scores = pd.DataFrame(scores, columns = columns)
df_scores.head()


# In[175]:


df_scores['tpr'] = df_scores.tp / (df_scores.tp + df_scores.fn)

df_scores['fpr'] = df_scores.fp / (df_scores.fp + df_scores.tn)

df_scores.head()


# In[176]:


df_scores[::10]


# In[177]:


plt.plot(df_scores.threshold, df_scores['tpr'], label = 'TPR')
plt.plot(df_scores.threshold, df_scores['fpr'], label = 'FPR')
plt.legend()


# ### Random Model

# In[178]:


np.random.seed(1)

y_rand = np.random.uniform(0, 1, size = len(y_val))
y_rand


# In[179]:


((y_rand == 0.5) == y_val).mean()


# In[180]:


def tpf_fpr_df(y_val, y_pred):
    thresholds = np.linspace(0, 1, 101)

    scores = []

    for t in thresholds:
        #Actual values
        actual_positive = (y_val == 1)
        actual_negative = (y_val == 0)

        #Prediction values
        predict_positive = (y_pred1 >= t)
        predict_negative = (y_pred1 < t)

        tp = (actual_positive & predict_positive).sum()
        tn = (actual_negative & predict_negative).sum()

        fp = (predict_positive & actual_negative).sum()
        fn = (predict_negative & actual_positive).sum()

        scores.append((t, tp, fp, tn, fn))
    
    
    # Create dataframe
    columns = ['threshold', 'tp', 'fp', 'tn', 'fn']  
    df_scores = pd.DataFrame(scores, columns = columns)

    df_scores['tpr'] = df_scores.tp / (df_scores.tp + df_scores.fn)
    df_scores['fpr'] = df_scores.fp / (df_scores.fp + df_scores.tn)
    
    return df_scores


# In[181]:


df_rand = tpf_fpr_df(y_val, y_rand)
df_rand#.head()


# In[182]:


df_rand[::10]


# In[183]:


plt.plot(df_rand.threshold, df_rand['tpr'], label = 'TPR')
plt.plot(df_rand.threshold, df_rand['fpr'], label = 'FPR')
plt.legend()


# ### Ideal Model
# 
# An Ideal model could be obtained by sorting the probabilities and then deciding the threshold that clear cuts the two groups. in this case churning verse non churning.

# In[184]:


#First we need of know the number of negative and positive examples

num_neg = (y_val == 0).sum()
num_pos = (y_val == 1).sum()

num_neg, num_pos


# In[185]:


y_ideal = np.repeat([0, 1], [num_neg, num_pos])
y_ideal


# In[186]:


y_ideal_pred = np.linspace(0, 1, len(y_val))


# In[187]:


1 - y_val.mean()


# In[188]:


((y_ideal_pred >= 0.726) == y_ideal).mean()


# In[189]:


df_ideal = tpf_fpr_df(y_ideal, y_ideal_pred)
df_ideal[::10]


# In[190]:


plt.plot(df_ideal.threshold, df_ideal['tpr'], label = 'TPR')
plt.plot(df_ideal.threshold, df_ideal['fpr'], label = 'FPR')
plt.legend()


# ### Putting Everything Together

# In[191]:


plt.plot(df_ideal.threshold, df_ideal['tpr'], label = 'TPR', color = 'black')
plt.plot(df_ideal.threshold, df_ideal['fpr'], label = 'FPR', color = 'black')


# plt.plot(df_scores.threshold, df_scores['tpr'], label = 'TPR')
# plt.plot(df_scores.threshold, df_scores['fpr'], label = 'FPR')


plt.plot(df_rand.threshold, df_rand['tpr'], label = 'TPR')
plt.plot(df_rand.threshold, df_rand['fpr'], label = 'FPR')

plt.legend()


# In[192]:


plt.figure(figsize = (10,8))

plt.plot(df_scores.fpr, df_scores.tpr, label = 'Model', color = 'black')
plt.plot([0,1], [0,1], label = 'Random', color = 'red')
#plt.plot(df_rand.fpr, df_rand.tpr, label = 'Random', color = 'red')
# plt.plot(df_ideal.fpr, df_ideal.tpr, label = 'Ideal', color = 'green')

plt.xlabel('FPR')
plt.ylabel('TPR')


plt.legend()


# NOTE: IF MODEL GOES BELOW BASELINE OR RANDOM, WE NEED TO FLIP POSTITIVE AND NEGATIVE PREDICTION

# In[193]:


from sklearn.metrics import roc_curve


# In[194]:


fpr, tpr, threshold = roc_curve(y_val, y_pred1)


# In[195]:


fpr, tpr, threshold


# In[196]:


plt.figure(figsize = (10,8))

plt.plot(fpr, tpr, label = 'Model', color = 'black')
plt.plot([0,1], [0,1], label = 'Random', color = 'red')
#plt.plot(df_rand.fpr, df_rand.tpr, label = 'Random', color = 'red')
# plt.plot(df_ideal.fpr, df_ideal.tpr, label = 'Ideal', color = 'green')

plt.xlabel('FPR')
plt.ylabel('TPR')


plt.legend()


# ### Area Under the Curve (AUC)
# 
# * Explains the probability that a randomly selected positive example is greater than a randomly selected negative example
# * A good AUC means the model orders the groups well
# * Good for binary classification problems

# In[197]:


from sklearn.metrics import auc


# In[198]:


auc(fpr, tpr)


# In[199]:


auc(df_scores.fpr, df_scores.tpr )


# In[200]:


auc(df_ideal.fpr, df_ideal.tpr)


# In[201]:


# We can also use roc auc

from sklearn.metrics import roc_auc_score


# In[202]:


roc_auc_score(y_val, y_pred1)


# In[203]:


#roc auc in two lines of code
fpr, tpr, threshold = roc_curve(y_val, y_pred1)
auc(fpr, tpr)


# In[204]:


neg = y_pred1[y_val == 0]
pos = y_pred1[y_val == 1]


# In[205]:


import random


# In[206]:


pos_index = random.randint(0, len(pos) - 1)
neg_index = random.randint(0, len(neg) - 1)


# In[207]:


pos_index > neg_index


# In[208]:


n = 10000
success = 0

for i in range(n):
    pos_index = random.randint(0, len(pos) - 1)
    neg_index = random.randint(0, len(neg) - 1)
    
    if pos[pos_index] > neg[neg_index]:
        success += 1
        
success/n


# In[209]:


# Vectorizing AUC with np.random
random.seed(1)
n = 10000
pos_index = np.random.randint(0, len(pos), size = n)
neg_index = np.random.randint(0, len(neg), size = n)


# In[210]:


pos[pos_index] > neg[neg_index]


# In[211]:


(pos[pos_index] > neg[neg_index]).mean()


# ### k fold cross validation

# In[212]:


def train(df_train, y_train):
    dicts = df_train[categorical + numerical].to_dict(orient = 'records')
    
    dv = DictVectorizer(sparse = False)
    X_train = dv.fit_transform(dicts)
    
    model= LogisticRegression(solver='lbfgs', max_iter=1000)
    model.fit(X_train, y_train)
    
    return dv, model


# In[213]:


dv, model = train(df_train, y_train)


# In[214]:


def predict(df_val, dv, model):
    dicts = df_val[categorical + numerical].to_dict(orient = 'records')
    
    X_val = dv.transform(dicts)
    
    y_pred = model.predict_proba(X_val)[:, 1]
    
    return y_pred


# In[215]:


y_pred = predict(df_val, dv, model)


# In[216]:


y_pred


# In[217]:


from sklearn.model_selection import KFold


# In[218]:


kfold = KFold(n_splits = 10, shuffle = True, random_state = 1)


# In[219]:


next(kfold.split(df_full_train))


# In[220]:


train_idx, val_idx = next(kfold.split(df_full_train))


# In[221]:


len(train_idx), len(val_idx)


# In[222]:


len(df_full_train)


# In[223]:


df_train = df_full_train.iloc[train_idx]
df_val = df_full_train.iloc[val_idx]


# In[224]:


#!pip install tqdm


# In[225]:


from tqdm.auto import tqdm


# In[226]:


scores = []

for train_idx, val_idx in tqdm(kfold.split(df_full_train)):
    df_train = df_full_train.iloc[train_idx]
    df_val = df_full_train.iloc[val_idx]
    
    y_train = df_train.churn.values
    y_val = df_val.churn.values
    
    dv, model = train(df_train, y_train)
    y_pred = predict(df_val, dv, model)
    
    auc = roc_auc_score(y_val, y_pred)
    scores.append(auc)


# In[227]:


scores


# In[228]:


print('%.3f +- %.3f' % (np.mean(scores), np.std(scores)))


# ### Parameter Tuning
# Adding C, regularisation to the training function

# In[229]:


def train(df_train, y_train, C = 0.001):
    dicts = df_train[categorical + numerical].to_dict(orient = 'records')
    
    dv = DictVectorizer(sparse = False)
    X_train = dv.fit_transform(dicts)
    
    model= LogisticRegression(solver='lbfgs', max_iter=1000, C = C)
    model.fit(X_train, y_train)
    
    return dv, model


# In[230]:


dv, model = train(df_train, y_train, C = 0.1)


# In[231]:


n_splits = 5

C = [0.001, 0.01, 0.1, 0.5, 1, 5, 10]

for c in tqdm(C):
    scores = []
    
    kfold = KFold(n_splits = n_splits, shuffle = True, random_state = 1)

    for train_idx, val_idx in kfold.split(df_full_train):
        df_train = df_full_train.iloc[train_idx]
        df_val = df_full_train.iloc[val_idx]

        y_train = df_train.churn.values
        y_val = df_val.churn.values

        dv, model = train(df_train, y_train, C = c)
        y_pred = predict(df_val, dv, model)

        auc = roc_auc_score(y_val, y_pred)
        scores.append(auc)

    print('C = %s %.3f +-%.3f' % (c, np.mean(scores), np.std(scores)))


# #### Observation: C = 1 gave us the best paramater roc_auc_score

# In[232]:


dv, model = train(df_full_train, df_full_train.churn.values, C = 1)
y_pred = predict(df_test, dv, model)

auc = roc_auc_score(y_test, y_pred)


# In[233]:


auc


# * Use cross validation when the dataset is small or quite large, and if we want to understand the standard deviation accross different folds to understand how stable the model is. For bigger dataset use 2 or 3 splits. for smaller, use 5-10 splits
# 
# * Use usual train_test_split is the dataset is very large

# ### Saving our model and using it

# In[237]:


ouput_file = f'model_C={C}.bin'

f_out = open(ouput_file, 'wb')
pickle.dump((dv, model), f_out)
f_out.close() # If we fail to close, we may not be sure if the content is inside, and if other services can use it


# In[238]:


# ## to solve the issues that may arise from failing to close

# with open(ouput_file, 'wb') as f_out:
#     pickle.dump((dv, model), f_out)


# In[239]:


with open(ouput_file, 'rb') as f_in:
    dv, model = pickle.load(f_in)


# In[243]:


df_full_train.iloc[5,:]


# In[247]:


customer = df.iloc[5, :].to_dict()
customer.pop('churn')


# In[248]:


customer.pop('customerid')


# In[255]:


X = dv.transform(customer)
X


# In[257]:


model.predict_proba(X)[0, 1]


# In[ ]:




