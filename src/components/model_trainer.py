import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from dataclasses import dataclass
import numpy as np
import pandas as pd

from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor
)
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import AdaBoostRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_model
# from src.components.model_trainer import ModelTrainer

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:

    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info('split training and test input data')

            X_train, y_train, X_test, y_test = (train_array[:,:-1], train_array[:,-1], test_array[:,:-1], test_array[:,-1]
                                                )
                
            models = {
                'Random Forest': RandomForestRegressor(),
                'Decision Tree': DecisionTreeRegressor(),
                'Gradient Boosting': GradientBoostingRegressor(),
                'Linear Regression': LinearRegression(),
                'K-Neighbors Regressor': KNeighborsRegressor(),
                'XGB Regressor': XGBRegressor(),
                'CatBoost Regressor': CatBoostRegressor(verbose=0),
                'AdaBoost Regressor': AdaBoostRegressor()
            }

            params = {
                'Decision Tree': {
                    'criterion': ['squared_error', 'absolute_error'],
                    'splitter': ['best', 'random'],
                    'max_features': ['sqrt', 'log2']
                },
                'Random Forest': {
                    'criterion': ['squared_error', 'absolute_error'],
                    'max_features': ['sqrt', 'log2', None],
                    'n_estimators': [8, 16, 32, 64]
                },
                'Gradient Boosting':{
                    'loss': ['squared_error', 'huber', 'absolute_error'],
                    'learning_rate': [.1, .01, .05, .001],
                    'subsample': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
                    'criterion': ['squared_error'],

                    # 'auto' is deprecated in tree models, Use 1.0 instead — it keeps the same behavior.
                    # 'max_features':['auto', 'sqrt', 'log2'], 
                    'max_features': [1.0, 'sqrt'],
                    
                    'n_estimators': [32, 64]
                },
                'Linear Regression': {},
                'XGB Regressor': {
                    'learning_rate': [.1, .01, .05, .001],
                    'n_estimators': [32, 64]
                },
                'CatBoost Regressor': {
                    'depth': [6, 8],
                    'learning_rate': [0.01, 0.05, 0.1],
                    # Valid ones: 'RMSE', 'MAE', 'Quantile', 'LogLinQuantile', 'Poisson', 'MAPE'
                    'loss_function': ['RMSE', 'MAE'],
                    'n_estimators': [32, 64]
                },
                'AdaBoost Regressor': {
                    'learning_rate': [.1, .01, 0.5, .001],
#                     Valid: 'linear', 'square', 'exponential' ✅

# But ensure your base model supports that loss.

# Also, AdaBoostRegressor by default uses DecisionTreeRegressor, which does not support all loss values depending on version.
#                     # 'loss': ['linear', 'square', 'exponential'],
                    'loss': ['linear', 'square'],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                }
            }
        
            model_report = evaluate_model(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models, param=params)
            
            # to get the best model score from dict
            best_model_score = max(model_report.values())

            # # to get the best model's name from dict
            # best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]

            best_model_name = max(model_report, key=model_report.get)
            best_model_score = model_report[best_model_name]

            best_model = models[best_model_name]

            if best_model_score < 0.6:
                logging.info(f'no best model found and model {best_model} score is : {best_model_score}')
                raise CustomException('no best model found')
            
            
            logging.info('best found model on both training and testing dataset : ' + str(best_model))

            save_object(
                file_path = self.model_trainer_config.trained_model_file_path,
                obj = best_model
            )

            predicted = best_model.predict(X_test)
            r2_scr = r2_score(y_test, predicted)

            return r2_scr

        except Exception as e:
            raise CustomException(e, sys)
        

        