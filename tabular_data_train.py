import pandas as pd
from sklearn.linear_model import LogisticRegression
import numpy as np

# score 0.8355: tried to ensemble (0.1 * logistic regression + 0.9 * v5_model)
df = pd.read_csv('train.csv')

df['sex'] = df['sex'].replace('male',0)
df['sex'] = df['sex'].replace('female',1)
df = df.dropna(how='any')

category = pd.cut(df.age_approx,bins=[0,10,20,30,40,50,60,70,80,90,100],labels=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
df.insert(0,'age_categorized',category)


df['anatom_site_general_challenge'] = df['anatom_site_general_challenge'].replace('torso', 0)
df['anatom_site_general_challenge'] = df['anatom_site_general_challenge'].replace('lower extremity', 1/5)
df['anatom_site_general_challenge'] = df['anatom_site_general_challenge'].replace('upper extremity', 2/5)
df['anatom_site_general_challenge'] = df['anatom_site_general_challenge'].replace('head/neck', 3/5)
df['anatom_site_general_challenge'] = df['anatom_site_general_challenge'].replace('palms/soles', 4/5)
df['anatom_site_general_challenge'] = df['anatom_site_general_challenge'].replace('oral/genital', 1)


df = df.loc[:,['sex','age_categorized','anatom_site_general_challenge','target']]
df.reset_index(drop=True)

train_df = df.iloc[0:29000,:]
target_df = df.iloc[29000:,:]

train_df = train_df.dropna(how='any')
target_df = target_df.dropna(how='any')

train_x = train_df.iloc[:,0:-1]
train_y = train_df.iloc[:,-1]

test_x = train_df.iloc[:,0:-1]
test_y = train_df.iloc[:,-1]

clf = LogisticRegression(random_state=0).fit(train_x, train_y)
print("test score:",clf.score(test_x,test_y))

submission_df = pd.read_csv('submission_v5.csv')
test_csv = pd.read_csv("test.csv")
image_names = test_csv['image_name']
test_csv['sex'] = test_csv['sex'].replace('male',0)
test_csv['sex'] = test_csv['sex'].replace('female',1)
test_csv = test_csv.fillna(0)

category = pd.cut(test_csv.age_approx,bins=[0,10,20,30,40,50,60,70,80,90,100],labels=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
test_csv.insert(0,'age_categorized',category)


test_csv['anatom_site_general_challenge'] = test_csv['anatom_site_general_challenge'].replace('torso', 0)
test_csv['anatom_site_general_challenge'] = test_csv['anatom_site_general_challenge'].replace('lower extremity', 1/5)
test_csv['anatom_site_general_challenge'] = test_csv['anatom_site_general_challenge'].replace('upper extremity', 2/5)
test_csv['anatom_site_general_challenge'] = test_csv['anatom_site_general_challenge'].replace('head/neck', 3/5)
test_csv['anatom_site_general_challenge'] = test_csv['anatom_site_general_challenge'].replace('palms/soles', 4/5)
test_csv['anatom_site_general_challenge'] = test_csv['anatom_site_general_challenge'].replace('oral/genital', 1)


test_csv = test_csv.loc[:,['sex','age_categorized','anatom_site_general_challenge']]
test_csv.reset_index(drop=True)


file = open("ensemble_submission_v5.csv", "w")
print(len(submission_df.index.values))
print(len(test_csv.index.values))

for row_index in range(len(submission_df.index.values)):
    file.write(image_names.at[row_index] + ',' + str('%.3f'%(0.85 * float(submission_df.at[row_index,'target']) + 0.15 * float(clf.predict([test_csv.iloc[row_index,:]])))) + '\n')



