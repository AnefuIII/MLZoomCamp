print('importing libraries')

import pandas as pd
pd.set_option('display.max_rows', None)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import export_text
from sklearn.feature_extraction import DictVectorizer

from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix,classification_report

import pickle

dv = DictVectorizer(sparse = False)

print('libraries imported successfully')

print('importing dataset')

df = pd.read_csv('raw_merged_heart_dataset.csv')
df.head()

print('Data Cleaning..')
# ### DATA CLEANING


'''when I attempted to Vectorize using dictvect, I noticed some numeric were regarded as object. 
so I went back to the kaggle source to confirm the dtypes
I was unable to because dictvecorizer could not convert ?
So I am trying to find and sort the ? issue
'''
columns_with_question_marks = df.columns[df.isin(['?']).any()]

#print(columns_with_question_marks)


df = df.replace('?', np.nan)
source_numericals = ['chol', 'trestbps', 'thalachh', 'oldpeak']

df[source_numericals] = df[source_numericals].apply(pd.to_numeric, errors='coerce').fillna(0)


cat_missing_columns = ['restecg', 'exang', 'slope', 'ca', 'thal']

for col in cat_missing_columns:
    mode_value = df[col].mode()[0]
    df.fillna({col: mode_value}, inplace=True) #or #df[col] = df[col].fillna(mode_value)

print('data cleaned successfully')

print('Data splitting')
# ### DATA PREPARATION (Train val test split)



df_full_train, df_test = train_test_split(df, test_size = 0.2, stratify = df.target, random_state = 42)

df_train, df_val = train_test_split(df, test_size = 0.25, stratify = df.target, random_state = 42)

print(f'The length of training set is {len(df_train)}')
print(f'The length of validation set is {len(df_val)}')
print(f'The length of test set is {len(df_test)}')


df_full_train = df_full_train.reset_index(drop = True)
df_train = df_train.reset_index(drop = True)
df_val = df_val.reset_index(drop = True)
df_test = df_test.reset_index(drop = True)

y_full_train = df_full_train.target.values
y_train = df_train.target.values
y_val = df_val.target.values
y_test = df_test.target.values


del df_full_train['target']
del df_train['target']
del df_val['target']
del df_test['target']
print('split completed')

print('Model training')
# ### TRAINING THE MODEL


def train(df_train, y_train):
    traindict = df_train.to_dict(orient = 'records')
    
    dv = DictVectorizer(sparse = False)
    X_train = dv.fit_transform(traindict)
    
    model = RandomForestClassifier(n_estimators = 11, 
                                    max_depth = 15,
                                    random_state = 1)
    model.fit(X_train, y_train)

    return dv, model


dv, model = train(df_full_train, y_full_train)
print('training complete')

testdict = df_test.to_dict(orient = 'records')

X_test = dv.transform(testdict)

y_pred = model.predict(X_test)

auc = roc_auc_score(y_test, y_pred)

print('full train auc: ', auc.round(3))

#output_file = 'model_rf_est11_depth15.bin'


# with open(output_file, 'wb') as f_out:
#     pickle.dump((dv, model), f_out)





