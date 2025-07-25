import sys
import os
import pandas as pd
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass
    
    def predict(self, features):

        try:
            model_path = 'artifacts\model.pkl'
            preprocessor_path = 'artifacts\preprocessor.pkl'
            # model_path=os.path.join("artifacts","model.pkl")
            # preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
            
            model = load_object(file_path = model_path)
            preprocessor = load_object(file_path = preprocessor_path)
            features = features.fillna("none")  # or use appropriate defaults

            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            model.predict(data_scaled)

            return preds
        
        except Exception as e:
            raise CustomException(e, sys)
    
class CustomData:
    def __init__(self,
        gender: str,
        race_ethnicity: int,
        parental_level_of_education: str,
        lunch: str,
        test_preparation_course: str,
        reading_score: int,
        writing_score: int):

        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score

    def get_data_as_dataframe(self):
        try:
            # custom_data_input_dict = {
            #     'gender' : self.gender,
            #     'race_ethnicity' : self.race_ethnicity,
            #     'parental_level_of_education' : self.parental_level_of_education,
            #     'lunch' : self.lunch,
            #     'test_preparation_course' : self.test_preparation_course,
            #     'reading_score' : self.reading_score,
            #     'writing_score' : self.writing_score
            # }

            custom_data_input_dict = {
                'gender': [self.gender],
                'lunch': [self.lunch],
                'parental_level_of_education': [self.parental_level_of_education],
                'race_ethnicity': [self.race_ethnicity],
                'reading_score': [self.reading_score],
                'test_preparation_course': [self.test_preparation_course],
                'writing_score': [self.writing_score]
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)