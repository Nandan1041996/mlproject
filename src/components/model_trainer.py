import os
import sys 
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object,evaluate_model,random_search_cv,grid_search_cv

from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor,GradientBoostingRegressor

from catboost import CatBoostRegressor

from dataclasses import dataclass
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    
    def initiate_model_trainer(self,train_array,test_array,preprocessor_path=None):
        try:
            logging.info("Splitting training and test input data")
            X_train,y_train,X_test,y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models = {"Random Forest": RandomForestRegressor(),
                    "Decision Tree": DecisionTreeRegressor(),
                    "Gradient Boosting": GradientBoostingRegressor(),
                    "K-Neighbors Regressor": KNeighborsRegressor(),
                    "XGBRegressor": XGBRegressor(),
                    # "CatBoosting Regressor":CatBoostRegressor(verbose=False),
                    "AdaBoost Regressor":AdaBoostRegressor(),
                    }


            params = {
                    "Random Forest" : {
                        "max_depth" : [3,2,5],
                        "n_estimators": [50,30,100],
                
                        },

                    "Decision Tree" : {
                        "criterion" : ['absolute_error', 'poisson', 'friedman_mse', 'squared_error'],
                    },

                    "Gradient Boosting":{
                        "n_estimators" : [50,30,100],
                        "learning_rate" : [0.1,0.01],
                    },

                    "K-Neighbors Regressor": {
                        "n_neighbors" : [5,3,4,6],
                    },

                    "XGBRegressor": {
                        "eta" : [0.3,0.4,0.5],
                        "max_depth" : [3,2,5],
                    },

                    # "CatBoosting Regressor":{
                    #     "learning_rate" : [0.1,0.01,1],
                    #     "depth" : [2,4,6],
                    #     "iterations":[30,50]
                    # },

                    "AdaBoost Regressor":{
                        "n_estimators" : [50,30,100],
                        "learning_rate" : [0.1,0.01]
                    }
                }
            
            
                
            model_report:dict = evaluate_model(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models = models,params = params)

            best_model_score = max(sorted(model_report.values()))   
                                                        
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            
            best_model = models[best_model_name]
            
            if best_model_score<0.6:
                raise CustomException("No best model fount")
            
            logging.info("Best model found on both training and testing dataset")

            save_object(file_path=self.model_trainer_config.trained_model_file_path,
                        obj=best_model)

            predicted = best_model.predict(X_test)

            r2_square = r2_score(y_test,predicted)
            return r2_square
        
        except Exception as e:
            raise CustomException(e,sys)
       