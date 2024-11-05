import numpy as np
import pandas as pd
from regression import (logreg, utils)
from sklearn.preprocessing import StandardScaler

def main():

    # load data with default settings
    # You will need to pick the features you want to use!
    features = ['AGE_DIAGNOSIS', 'Documentation of current medications',
       'Computed tomography of chest and abdomen',
       'Plain chest X-ray (procedure)', 'Penicillin V Potassium 250 MG']
    num_feats = len(features)
    X_train, X_val, y_train, y_val = utils.loadDataset(features = features, split_percent=0.8, split_state=42)

    # scale data since values vary across features
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_val = sc.transform(X_val)
    
    #print(X_train.shape, X_val.shape, y_val.shape, y_train.shape)


    """
    # for testing purposes once you've added your code
    # CAUTION & HINT: hyperparameters have not been optimized
    """
    log_model = logreg.LogisticRegression(num_feats=num_feats, max_iter=200, tol=0.001, learning_rate=0.00001, batch_size=12)
    log_model.train_model(X_train, y_train, X_val, y_val)
    log_model.plot_loss_history()

if __name__ == "__main__":
    main()
