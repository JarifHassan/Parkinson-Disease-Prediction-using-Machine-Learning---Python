import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.feature_selection import SelectKBest, chi2
from tqdm.notebook import tqdm
from sklearn import metrics
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression

import warnings
warnings.filterwarnings('ignore')

#2. Importing Dataset

df = pd.read_csv('parkinson_disease.csv')
pd.set_option('display.max_columns', 10)
df.sample(5)

#3. Data Exploration and Cleaning

df.info()

print(df.describe().T)

print(df.isnull().sum().sum())

#4. Data Wranglin

df = df.groupby('id').mean().reset_index()
df.drop('id', axis = 1, inplace= True)

columns = list(df.columns)
for col in columns:
    if col =='class':
        continue
    filtered_columns = [col]
    for col1 in df.columns:
        if((col==col1) | (col=='class')):
            continue
        val = df[col].corr(df[col1])
        if val > 0.7:
            columns.remove(col1)
            continue
        else:
            filtered_columns.append(col1)
    df = df[filtered_columns]
print(df.shape)

#5. Feature Selection

X = df.drop('class', axis=1)
X_norm = MinMaxScaler().fit_transform(X)
selector = SelectKBest(chi2, k=30)
selector.fit(X_norm, df['class'])
filtered_columns = selector.get_support()
filtered_data = X.loc[:, filtered_columns]
filtered_data['class'] = df['class']
df = filtered_data
df.shape



#6. Handling Class Imbalance and Splitting Data

x = df['class'].value_counts()
plt.pie(x.values,
        labels = x.index,
        autopct= '%1.1f%%')
plt.show()

features = df.drop('class', axis=1)
target = df['class']

X_train, X_val,Y_train, Y_val = train_test_split(features, target,
                                      test_size=0.2,
                                      random_state=10)

ros = RandomOverSampler(sampling_strategy=1.0,
                        random_state=0)
X_resampled, y_resampled = ros.fit_resample(X_train, Y_train)
print(X_resampled.shape, y_resampled.value_counts())


#7. Model Training and Evaluation

from sklearn.metrics import roc_auc_score as ras

models = [LogisticRegression(class_weight='balanced'), XGBClassifier(), SVC(kernel='rbf', probability=True)]
for model in models:
    model.fit(X_resampled, y_resampled)
    print(f'{model} : ')

    # Use predict_proba for ROC AUC
    if hasattr(model, 'predict_proba'):
        train_probs = model.predict_proba(X_resampled)[:, 1]
        val_probs = model.predict_proba(X_val)[:, 1]
        print('Training ROC AUC : ', ras(y_resampled, train_probs))
        print('Validation ROC AUC : ', ras(Y_val, val_probs))
    elif hasattr(model, 'decision_function'):
        train_scores = model.decision_function(X_resampled)
        val_scores = model.decision_function(X_val)
        print('Training ROC AUC : ', ras(y_resampled, train_scores))
        print('Validation ROC AUC : ', ras(y_val, val_scores))
    else:
        print("Model doesn't have predict_proba or decision_function for ROC AUC.")
    print()


#8. Analyzing Model Performance

from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import classification_report
ConfusionMatrixDisplay.from_estimator(models[0], X_val, Y_val)
plt.show()
print(classification_report(Y_val,models[0].predict(X_val)))