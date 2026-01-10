import sys
import os
import numpy as np
import pandas as pd
from dataclasses import dataclass
from src.utils import save_object

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.logger import logging
from src.exception import CustomException

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join("artifacts","preprocessor.pkl")


class DataTranformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        "This function is responsible for data transformation."
        try:
            numerical_columns = ['reading_score', 'writing_score']
            
            categorical_columns = ['gender', 
                                   'race_ethnicity', 
                                   'parental_level_of_education', 
                                   'lunch', 
                                   'test_preparation_course']

            num_pipeline = Pipeline(
                steps = [('imputer',SimpleImputer(strategy='median')),
                         ("scaler",StandardScaler())
                         ])
            
            logging.info("numerical columns imputation and scaling is done.")
            
            cat_pipeline = Pipeline(
                steps = [("imputer",SimpleImputer(strategy='most_frequent')),
                         ('one_hot_encoding',OneHotEncoder()),
                         ('scale',StandardScaler(with_mean=False))
                         ])
            
            logging.info("categorical column encoding completed")
            logging.info(f"categorical columns: {categorical_columns}")
            logging.info(f"numerical columns: {numerical_columns}")

            preprocessor = ColumnTransformer([
                ("numerical pipeline",num_pipeline,numerical_columns),
                ("cat_pipelines",cat_pipeline,categorical_columns),
                ])
            
            return preprocessor
            
        except Exception as exe:
            raise CustomException(exe,sys)

    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df =pd.read_csv(test_path)

            logging.info("Read train and test data completed")
            
            preprocessor_obj = self.get_data_transformer_object()
            logging.info("Obtaining Preprocessing Object")

            target_column_name = "math_score"
            numerical_columns = ["reading_score","writing_score"]

            input_fearure_train_df = train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_fearure_test_df = test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df = test_df[target_column_name]
            
            logging.info(f"Applying preprocessing object on training dataframe and testing dataframe.")

            input_fearure_train_arr = preprocessor_obj.fit_transform(input_fearure_train_df)

            input_fearure_test_arr = preprocessor_obj.transform(input_fearure_test_df)

            # concatinate column wise
            train_arr = np.c_[
                input_fearure_train_arr,np.array(target_feature_train_df)]
            test_arr = np.c_[
                input_fearure_test_arr,np.array(target_feature_test_df)]

            logging.info("saved preprocessing object")

            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessor_obj
            )
            a = pd.DataFrame(train_arr)
            b = pd.DataFrame(test_arr)
            return (train_arr,
                    test_arr,
                    self.data_transformation_config.preprocessor_obj_file_path)
        
        except Exception as e:
            print(e)
            return CustomException(e,sys)



