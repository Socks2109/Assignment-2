"""
Write your logreg unit tests here. Some examples of tests we will be looking for include:
* check that fit appropriately trains model & weights get updated
* check that predict is working

More details on potential tests below, these are not exhaustive
"""
from regression import (logreg, utils)
from sklearn.preprocessing import StandardScaler
import numpy as np

BaseRegressor = logreg.BaseRegressor
LogisticRegression = logreg.LogisticRegression

features = ['AGE_DIAGNOSIS', 'Documentation of current medications',
	'Computed tomography of chest and abdomen',
	'Plain chest X-ray (procedure)', 'Penicillin V Potassium 250 MG']
num_feats = len(features)
X_train, X_val, y_train, y_val = utils.loadDataset(features = features, split_percent=0.8, split_state=42)


sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_train_weight = np.hstack([X_train, np.ones((X_train.shape[0], 1))])
X_val = sc.transform(X_val)
log_model = LogisticRegression(num_feats=num_feats, max_iter=10, tol=0.01, learning_rate=0.00001, batch_size=12)

def test_updates():
	# Check that your gradient is being calculated correctly
	# What is a reasonable gradient? Is it exploding? Is it vanishing?
	"""
	Checks whether the gradient explodes or vanishes.
	Also checks for correct gradient calculation
	"""
	assert np.any(log_model.calculate_gradient(X_train_weight, y_train)) < 1000
	assert np.any(log_model.calculate_gradient(X_train_weight, y_train)) > 1e-12


	# Check that your loss function is correct and that 
	# you have reasonable losses at the end of training
	# What is a reasonable loss?
	"""
	This section is used to check the decreasing validation loss.
	As long as the loss decreases over the number of iterations,
	it is considered a reasonable loss
	"""
	log_model.train_model(X_train, y_train, X_val, y_val)
	assert log_model.loss_history_val[-1] < log_model.loss_history_val[0]


def test_predict():
	# Check that self.W is being updated as expected 
 	# and produces reasonable estimates for NSCLC classification
	# What should the output should look like for a binary classification task?
	"""
	Checks that the prediction only produces outputs with 1s or 0s
	"""
	assert np.any(log_model.make_prediction(X_train_weight)) == 1 or 0