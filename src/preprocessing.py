import numpy as np
import pandas as pd

#import data from kaggle
data = pd.read_csv("data/student_depression_dataset.csv")
data = data.reset_index(drop = True)

#focus on undergraduate students that are students from ages 18-22 (traditional 4-year program)
data = data[data['Age'] <= 22]
data = data[data['Degree'].str.startswith('B')]

#convert 10 point cgpa to 4 point gpa
data["GPA"] = data["CGPA"] / 10 * 4
data["GPA"] = data["GPA"].round(2)

#drop unneccessary columns
data = data.drop(columns = ["id","City","Job Satisfaction", "Work Pressure", "Profession", "CGPA"])

#save processed data
data.to_csv('data/processed_data.csv', index=False)
