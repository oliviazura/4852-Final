import numpy as np
import pandas as pd

#import data from kaggle
data = pd.read_csv("student_depression_dataset.csv")

#focus on undergraduate students that are students from ages 18-22 (traditional 4-year program)
data = data[data['Age'] <= 22]
data = data[data['Degree'].str.startswith('B')]

#drop unneccessary columns
data = data.drop(columns = ["id", "City", "Work Pressure", "Profession"])

#save processed data
data.to_csv('processed_data.csv')
