
#Required Packages
import numpy as np #The Numpy numerical computing library
import pandas as pd #The Pandas data science library
import math #The Python math module
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt

#Import the dataset
data = pd.read_csv('pga_tour_stats_2020.csv')

#Filtering Data
s1='Player Name' 
s2='Approach Distance'
s3='GIR Percent'
s4='Scramble Percent'
s5='Official Money'
s6='Ball Speed'
s7='Club Head Speed'
s8='Driving Accuracy'
s9='Driving Distance'
s10='Putting Birdie Distance'
s11='GIR Putting Average'
s12='Average Putts'
s13='Par 3 Average'
s14='Par 4 Average'
s15='Par 5 Average'
s16='Scoring Average'

df = data.filter(items=[s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,s11,s12,s13,s14,s15,s16])

#Cleaning Data
df['Official Money'] = df['Official Money'].str.replace(' ','')
df['Official Money'] = df['Official Money'].str.replace(',','')
df['Official Money'] = df['Official Money'].str.replace('$','')
df['Official Money'].fillna(0, inplace=True)
df['Official Money'] = df['Official Money'].astype(int)

df['Putting Birdie Distance'] = df['Putting Birdie Distance'].str.replace('"','')
df['Putting Birdie Distance'] = df['Putting Birdie Distance'].str.replace("'",'')
df['Putting Birdie Distance'] = df['Putting Birdie Distance'].str.replace(' ','.')
df['Official Money'].fillna(0, inplace=True)
df['Putting Birdie Distance'] = df['Putting Birdie Distance'].astype(float)

df=df[:207]



#Data Analysis 1: Distribution of Variables 
f, ax = plt.subplots(nrows = 4, ncols = 4, figsize=(20,20))
distribution = df.loc[:,df.columns!='Player Name'].columns
rows = 0
cols = 0
for i, column in enumerate(distribution):
    p = sns.distplot(df[column], ax=ax[rows][cols])
    cols += 1
    if cols == 4:
        cols = 0
        rows += 1
plt.show()

#Data Analysis 2: Correlation Plot
corr = df.corr()
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values,
            cmap='coolwarm')
plt.show()

corr=df.corr()

#Data Analysis 3 
#Drivng Distance, Par 5 Scoring, and FedexCup Official Money
sns.relplot(x=s9,y=s15,size=s5, data=df)
plt.show()

#GIR Percent, Approach Distance, Par 4 Scoring Average
sns.relplot(x=s3,y=s2,size=s14, data=df)

bins=[]
for i in range (0,9):
    bins.append((np.percentile(df["Official Money"], i*12.5)))

group_names=['Bottom 12.5 Percentile', '12.5-25 Percentile', '25-37.5 Percentile', 
             '37.5-50 Percentile', '60-62.5 Percentile', 
              '62.5-75 Percentile', '75-87.5 Percentile', 'Top 12.5 Percentile']

df['Official Money'] = pd.cut(df['Official Money'], bins=bins, labels=group_names)
df.dropna(axis = 0, inplace=True)



#Machine Learning Packages
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.feature_selection import RFE
from sklearn.metrics import classification_report
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVC  
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)



#Logistic Regression
def log_reg(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    random_state = 200)
    clf = LogisticRegression().fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print('Accuracy of Logistic regression classifier on training set: {:.2f}'
          .format(clf.score(X_train, y_train)))
    print('Accuracy of Logistic regression classifier on test set: {:.2f}'
          .format(clf.score(X_test, y_test)))
    cf_mat = confusion_matrix(y_test, y_pred)
    confusion = pd.DataFrame(data = cf_mat)
    print(confusion)
    print(classification_report(y_test, y_pred))
    rfe = RFE(clf, 5)
    rfe = rfe.fit(X, y)
    print('Feature Importance')
    
    
target = df['Official Money']
# Removing the columns Player Name, Wins, and Winner from the dataframe
ml_df = df.copy()
ml_df.drop(['Player Name','Official Money'], axis=1, inplace=True)

#Support Vector 
def svc_class(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    random_state = 200)
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    
    svclassifier = SVC(kernel='rbf', C=10000)  
    svclassifier.fit(X_train_scaled, y_train) 
    y_pred = svclassifier.predict(X_test_scaled) 
    print('Accuracy of SVM on training set: {:.2f}'
          .format(svclassifier.score(X_train_scaled, y_train)))
    print('Accuracy of SVM classifier on test set: {:.2f}'
          .format(svclassifier.score(X_test_scaled, y_test)))

    
#RF
def random_forest(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    random_state = 200)
    clf = RandomForestClassifier(n_estimators=200).fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print('Accuracy of Random Forest classifier on training set: {:.2f}'
          .format(clf.score(X_train, y_train)))
    print('Accuracy of Random Forest classifier on test set: {:.2f}'
          .format(clf.score(X_test, y_test)))
    
    cf_mat = confusion_matrix(y_test, y_pred)
    confusion = pd.DataFrame(data = cf_mat)
    print(confusion)
    
    print(classification_report(y_test, y_pred))
    
    # Returning the 5 important features 
    rfe = RFE(clf, 5)
    rfe = rfe.fit(X, y)
    print('Feature Importance')
    
print(log_reg(ml_df, target))
print(svc_class(ml_df, target))
print(random_forest(ml_df, target))
