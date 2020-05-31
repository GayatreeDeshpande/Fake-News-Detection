# Fake News Detection using Passive Aggressive Classifier:

# Import Necessary Packages : 
import pandas as pd
import itertools
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Import Dataset :
data = pd.read_csv("D:\\Datasets\\news.csv")

data.head()
data.shape
data.isnull().any()

# Split the data into training and testing dataset :
x_train,x_test,y_train,y_test = train_test_split(data.text,data.label,test_size=0.3,random_state=42)

# Initializing TfidfVectorizer :
vect = TfidfVectorizer(stop_words = "english",max_df = 0.7)

vect_train = vect.fit_transform(x_train)
vect_test = vect.transform(x_test)

# Fitting the Model :
classify = PassiveAggressiveClassifier(max_iter = 50)
classify.fit(vect_train,y_train)

# Predicting Real and Fake News :
y_predict = classify.predict(vect_test)

# Checking Accuracy of the Model :
score = accuracy_score(y_test,y_predict)
score 

# Checking Confusion Matrix to check if there is no bias during splitting the data :
confusion_matrix(y_test,y_predict,labels=["REAL","FAKE"])
