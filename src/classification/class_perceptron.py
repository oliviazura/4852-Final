import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import f1_score

from sklearn.linear_model import Perceptron


#import preprocessed data
train_data = pd.read_csv("data/train_data.csv")
test_data = pd.read_csv("data/test_data.csv")

#seperate predictors and response
x_train = train_data[["Gender","Age","Academic Pressure","Study Satisfaction","Sleep Duration","Dietary Habits","Degree","Have you ever had suicidal thoughts ?", "Work/Study Hours","Financial Stress","Family History of Mental Illness", "GPA"]]
y_train = train_data["Depression"]

x_test = test_data[["Gender","Age","Academic Pressure","Study Satisfaction","Sleep Duration","Dietary Habits","Degree","Have you ever had suicidal thoughts ?", "Work/Study Hours","Financial Stress","Family History of Mental Illness", "GPA"]]
y_test = test_data["Depression"]

#seperate data for pipelines
data_cat_onehot = ["Gender", "Degree", "Have you ever had suicidal thoughts ?", "Family History of Mental Illness"]
data_cat_ordinal = ["Academic Pressure", "Study Satisfaction", "Sleep Duration", "Dietary Habits", "Financial Stress"]
data_num = ["Age", "Work/Study Hours", "GPA"]

onehot_pipeline = make_pipeline(
    OneHotEncoder()
)
ordinal_pipeline = make_pipeline(
    OrdinalEncoder(
        categories =[
            [1, 2, 3, 4, 5], #Academic Pressure
            [1, 2, 3, 4, 5], #Study Satisfaction
            ["Less than 5 hours", "5-6 hours", "7-8 hours", "More than 8 hours"], #Sleep Duriation
            ["Unhealthy", "Moderate", "Healthy"], #Dietary Habits
            [1, 2, 3, 4, 5] #Financial Stress
        ],
        handle_unknown = 'use_encoded_value',
        unknown_value = -1
    )
)
num_pipeline = make_pipeline(
    StandardScaler()
)

#process data
preprocessing = ColumnTransformer([
    ("onehot", onehot_pipeline, data_cat_onehot),
    ("ordinal", ordinal_pipeline, data_cat_ordinal),
    ("num", num_pipeline, data_num)
])

#using ridge regression as regularization, 
perceptron_pipeline = Pipeline([
    ("preprocessing", preprocessing),
    ("model", Perceptron(penalty = "l2", alpha = .1, eta0 = .01, random_state= 0))
])

perceptron_pipeline.fit(x_train, y_train)
train_predictions = perceptron_pipeline.predict(x_train)

train_cm = confusion_matrix(y_train, train_predictions)
print(train_cm)

test_predictions = perceptron_pipeline.predict(x_test)


test_cm = confusion_matrix(y_test, test_predictions)
print(test_cm)

cm_display = ConfusionMatrixDisplay(confusion_matrix = test_cm, display_labels=["Not Depressed", "Depressed"])
cm_display.plot()
plt.savefig("visuals/perceptron_confusion_matrix.png")

f1 = f1_score(y_test, test_predictions)
print(f1)