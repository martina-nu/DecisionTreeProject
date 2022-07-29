import pandas as pd
import pickle
from sklearn import model_selection
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import resample

df= pd.read_csv("https://raw.githubusercontent.com/4GeeksAcademy/decision-tree-project-tutorial/main/diabetes.csv")



df_majority = df[(df["Outcome"]==0)]
df_minority = df[(df["Outcome"]==1)]
df_minority_upsampled = resample(df_minority,replace=True,n_samples=500,random_state=42)
df = pd.concat([df_minority_upsampled,df_majority])
df["Outcome"].value_counts()


X=df.drop('Outcome',axis=1)
y=df["Outcome"]

X_train,X_test,y_train,y_test=model_selection.train_test_split(X,y, random_state=15)

best_model = DecisionTreeClassifier(criterion="gini", max_depth=11, min_samples_split=3)
best_model.fit(X_train,y_train)

filename = 'models/model.sav'

pickle.dump(best_model, open(filename,'wb'))