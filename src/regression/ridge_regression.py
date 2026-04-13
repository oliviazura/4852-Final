import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.linear_model import Ridge

from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import cross_val_score
from scipy.stats import bootstrap

#import preprocessed data
train_data = pd.read_csv("data/train_data.csv")
test_data = pd.read_csv("data/test_data.csv")

#seperate predictors and response
x_train = train_data[["Gender","Age","Academic Pressure","Study Satisfaction","Sleep Duration","Dietary Habits","Degree","Have you ever had suicidal thoughts ?", "Work/Study Hours","Financial Stress","Family History of Mental Illness", "Depression"]]
y_train = train_data["GPA"]

x_test = test_data[["Gender","Age","Academic Pressure","Study Satisfaction","Sleep Duration","Dietary Habits","Degree","Have you ever had suicidal thoughts ?", "Work/Study Hours","Financial Stress","Family History of Mental Illness", "Depression"]]
y_test = test_data["GPA"]

#seperate data for pipelines
data_cat_onehot = ["Gender", "Degree", "Have you ever had suicidal thoughts ?", "Family History of Mental Illness", "Depression"]
data_cat_ordinal = ["Academic Pressure", "Study Satisfaction", "Sleep Duration", "Dietary Habits", "Financial Stress"]
data_num = ["Age", "Work/Study Hours"]

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

#pipeline for logistic regression, using ridge regression as regularization
logistic_pipeline = Pipeline([
    ("preprocessing", preprocessing),
    ("model", Ridge(alpha = 1, random_state= 0))
])

logistic_pipeline.fit(x_train, y_train)
train_predictions = logistic_pipeline.predict(x_train)

test_predictions = logistic_pipeline.predict(x_test)

def rmse(squared_errors):
    return np.sqrt(np.mean(squared_errors))

confidence = 0.95
squared_errors = (test_predictions - y_test) ** 2
print(f"Bootstrapped mean RMSE: {rmse(squared_errors)}")

boot_result = bootstrap([squared_errors], rmse, confidence_level=confidence,
random_state=42)
rmse_lower, rmse_upper = boot_result.confidence_interval

print(f"95% CI for RMSE: ({rmse_lower:.4f}, {rmse_upper:.4f})")
