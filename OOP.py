import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score, log_loss
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import OneHotEncoder
import statistics
import pickle
from sklearn.preprocessing import RobustScaler
from xgboost import XGBClassifier

class DataHandler:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None
        self.input_df = None
        self.output_df = None

    def load_data(self):
        self.data = pd.read_csv(self.file_path)
        
        print(self.data.head())

    def dropNA(self):
        self.data.dropna(inplace=True)
    
    def dropCol(self, colums):
        self.data = self.data.drop(colums, axis=1)
        
    def dropDupli(self):
        self.data.drop_duplicates(inplace=True)

    def create_input_output(self, target_column):
        self.output_df = self.data[target_column]
        self.input_df = self.data.drop([target_column], axis=1)
        
class ModelHandler:
    def __init__(self, input_data, output_data):
        self.input_data = input_data
        self.output_data = output_data
        self.x_train, self.x_test, self.y_train, self.y_test, self.y_predict = [None] * 5

    def split_data(self, test_size=0.2, random_state=42):
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            self.input_data, self.output_data, test_size=test_size, random_state=random_state)

    def OHE(self, columns):
        encoder = OneHotEncoder()
        x_train_subset = self.x_train[columns]
        train_encoded = pd.DataFrame(encoder.fit_transform(x_train_subset).toarray(), columns=encoder.get_feature_names_out())
        self.x_train = self.x_train.reset_index(drop=True)
        self.x_train = pd.concat([self.x_train, train_encoded], axis=1)

        x_test_subset = self.x_test[columns]
        test_data = pd.DataFrame(encoder.transform(x_test_subset).toarray(), columns=encoder.get_feature_names_out())
        self.x_test = self.x_test.reset_index(drop=True)
        self.x_test = pd.concat([self.x_test, test_data], axis=1)
        
        self.x_train.drop(columns, axis=1, inplace=True)
        self.x_test.drop(columns, axis=1, inplace=True)

        self.export_encoder(encoder, path="encoder.pkl")  # Call the export function

    def export_encoder(self, encoder, path="encoder.pkl"):
        with open(path, "wb") as encoder_file:
            pickle.dump(encoder, encoder_file)

    def RobustScaler(self, columns):
        scaler = RobustScaler()
        self.x_train[columns] = scaler.fit_transform(self.x_train[columns])
        self.x_test[columns] = scaler.transform(self.x_test[columns])

        self.export_scaler(scaler, path="scaler.pkl")  # Call the export function

    def export_scaler(self, scaler, path="scaler.pkl"):
        with open(path, "wb") as scaler_file:
            pickle.dump(scaler, scaler_file)
    
    def createModel(self):
        self.model = XGBClassifier(
            n_estimators=150,
            max_depth=3,
            learning_rate=0.1,
            reg_alpha=0.1,
            reg_lambda=0,
        )

    def train(self):
        self.model.fit(self.x_train, self.y_train)

    def predict(self):
        self.y_predict =  self.model.predict(self.x_test)

    def pickle_dump(self, filename='best_model.pkl'):
        with open(filename, 'wb') as file:
            pickle.dump(self.model, file)
        
    def createReport(self):
        print('\nClassification Report\n')
        print(classification_report(self.y_test, self.y_predict))

    def evaluate_model(self):
        predictions = self.model.predict(self.x_test)
        return accuracy_score(self.y_test, predictions)



file_path = 'data_D.csv'  
data_handler = DataHandler(file_path)
data_handler.load_data()
data_handler.dropNA()
data_handler.dropCol(['Unnamed: 0', 'id', 'CustomerId', 'Surname'])
data_handler.create_input_output('churn')

input_df = data_handler.input_df
output_df = data_handler.output_df

model_handler = ModelHandler(input_df, output_df)

model_handler.split_data()

model_handler.OHE(['Geography', 'Gender'])
model_handler.RobustScaler(['CreditScore', 'Balance', 'EstimatedSalary'])

model_handler.createModel()
model_handler.train()
model_handler.predict()
print("Model Accuracy:", model_handler.evaluate_model())
model_handler.createReport()