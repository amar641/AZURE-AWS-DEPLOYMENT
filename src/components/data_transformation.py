import sys
from dataclasses import dataclass
from typing import List, Optional
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.base import BaseEstimator
import os
from src.exception import CustomException   
from src.logging_config import log
from src.utils import save_object

class DataTransformationConfig:
    ## saving model as pickle file
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')
class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self) -> ColumnTransformer:
        """
        This function is responsible for data transformation
        """
        try:
            numerical_columns = ['writing_score', 'reading_score']
            categorical_columns = [
    'gender',
    'race_ethnicity',
    'parental_level_of_education',
    'lunch',
    'test_preparation_course'
]
            ## numerical pipeline,, getting numerical columns and applying transformation
            num_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ]
            )
            ## categorical pipeline,, getting categorical columns and applying transformation
            cat_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('encoder', OneHotEncoder(handle_unknown='ignore'))
                ]
            )
            
            log.info(f'Numerical columns: {numerical_columns}')
            log.info(f'Categorical columns: {categorical_columns}')
            log.info('Creating preprocessing pipeline...')
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', num_pipeline, numerical_columns),
                    ('cat', cat_pipeline, categorical_columns)
                ]
            )
            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)
    def initiate_data_transformation(self, train_path: str, test_path: str):    
        try:
            # reading train and test data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            log.info('Read train and test data completed')
            log.info('Obtaining preprocessing object')
            preprocessor_obj = self.get_data_transformer_object()
            target_column_name = 'math_score'
            numerical_columns = ['writing_score', 'reading_score']
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]
            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]
            ## transforming using preprocessor obj
            input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor_obj.transform(input_feature_test_df)
            log.info('Applying preprocessing object on training and testing datasets.')
            ## concatenate numpy arrays
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]
            log.info('Concatenated input features and target feature arrays.')
            ## save the preprocessor object
            import pickle

            save_object(self.data_transformation_config.preprocessor_obj_file_path, preprocessor_obj)
            log.info('Preprocessor object saved successfully.')
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
        except Exception as e:
            raise CustomException(e, sys)