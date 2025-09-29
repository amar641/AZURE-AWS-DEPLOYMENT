import os
import sys
from dataclasses import dataclass
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBRegressor
from src.exception import CustomException
from src.logging_config import log
from src.utils import save_object, evaluate_model
from sklearn.metrics import r2_score,accuracy_score
from src.utils import save_object, evaluate_model


@dataclass
class ModelTrainerConfig: ##saving model as pickle file
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            log.info('Splitting training and test input data')
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )
            models = {
                "Random Forest": RandomForestClassifier(),
                "Decision Tree": DecisionTreeClassifier(),
                "Gradient Boosting": GradientBoostingClassifier(),
                "Logistic Regression": LogisticRegression(),
                "K-Neighbors Classifier": KNeighborsClassifier(),
                "XGB Classifier": XGBRegressor(),
                "AdaBoost Classifier": AdaBoostClassifier()
            }
            model_report: dict = evaluate_model(X_train, y_train, X_test, y_test, models)
            ## to get best model score from dictionary
            best_model_score = max(sorted(model_report.values()))
            ## to get best model name from dictionary
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]
            if best_model_score < 0.6:
                raise CustomException("No best model found")
            log.info(f'Best model found: {best_model_name} with accuracy score: {best_model_score}')
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            predicted = best_model.predict(X_test)
            accuracy = r2_score(y_test, predicted)
            return accuracy
        except Exception as e:
            raise CustomException(e, sys)