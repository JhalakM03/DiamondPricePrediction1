from src.loger import logging
from src.exception import CustomException
from src.utils import save_object
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OrdinalEncoder    
from sklearn.pipeline import Pipeline
import sys,os
import pandas as pd
import numpy as np
from dataclasses import dataclass
from sklearn.compose import ColumnTransformer
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_data_file_path=os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()
    
    def get_data_transformation_obj(self):
        try:
            logging.info("Data transformation initiated")
            categorical_cols = ['cut', 'color','clarity']
            numerical_cols = ['carat', 'depth','table', 'x', 'y', 'z']

            cut_categories = ['Fair', 'Good', 'Very Good','Premium','Ideal']
            color_categories = ['D', 'E', 'F', 'G', 'H', 'I', 'J']
            clarity_categories = ['I1','SI2','SI1','VS2','VS1','VVS2','VVS1','IF']
            
            logging.info("Pipeline initiated")

            num_pipeline= Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ]
            )
            cat_pipeline=Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='most_frequent')),
                    ('ordinalencoder', OrdinalEncoder(categories=[cut_categories,color_categories,clarity_categories])),
                    ('scaler', StandardScaler())

                ]
            )
            preprocessor=ColumnTransformer([
                ('num_pipeline',num_pipeline,numerical_cols),
                ('cat_pipleine', cat_pipeline, categorical_cols)

            ])
            return preprocessor
            logging.info("Pipeline Completed")
            



        except Exception as e:
            logging.info("Error in Data Transformation")
            raise CustomException(e,sys)

    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info('Reading of train and test data completed')
            logging.info(f'Train dataframe head: \n {train_df.head().to_string()}')
            logging.head(f' Test dataframe head: \n {test_df.head().to_string()}')
            logging.info('Obtaining preprocessing object')

            preprocessing_obj = self.get_data_transformation_object()

            target_column_name='price'
            drop_columns=[target_column_name,'id']

            input_feature_train_df=train_df.drop(columns=drop_columns,axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=drop_columns,axis=1)
            target_feature_test_df=test_df[target_column_name]

            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            train_arr=np.c_(input_feature_train_arr,np.array(target_feature_train_df))
            test_arr=np.c_(input_feature_train_arr,np.array(target_feature_test_df))

            save_object(
                file_path=self.data_transformation_config.preprocessor_data_file_path,
                obj=preprocessing_obj

            )
            logging.ingo('Preprocessor pickle created and saved')

            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_data_file_path
            )
        except Exception as e:
            logging.INFO('Exception occured in data transformation')
            raise CustomException(e,sys)

