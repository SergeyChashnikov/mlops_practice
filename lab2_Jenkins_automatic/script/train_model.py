import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
import os


def train_model():

    X_train = pd.read_csv('data/X_train.csv')
    y_train = pd.read_csv('data/y_train.csv')

    model_rf = RandomForestClassifier(n_estimators=150,
                                     max_depth=10,
                                     oob_score=True)
    
    model_rf.fit(X_train, y_train.values.ravel())
    #print(model_rf.oob_score_)
    os.makedirs('models', exist_ok=True)
    joblib.dump(model_rf, f'models/model.pkl')
    print('Model trained and saved in madels/model.pkl')


if __name__ == "__main__":
    train_model()