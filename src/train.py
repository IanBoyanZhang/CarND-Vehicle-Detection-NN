import time
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

import sklearn
if float(sklearn.__version__.split('.')[-2]) >= 18:
    from sklearn.model_selection import train_test_split
else:
    from sklearn.cross_validation import train_test_split

def train(car_features, notcar_features, return_model=True, verbose=True):
    """
    """
    # Crate an array stack of feature vectors
    X = np.vstack((car_features, notcar_features)).astype(np.float64)
    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X)
    # Apply the scaler to X
    scaled_X = X_scaler.transform(X)

    # Define the labels vector
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(
        scaled_X, y, test_size=0.2, random_state=rand_state)

    if verbose:
        print('Feature vector length:', len(X_train[0]))

    # Use a linear SVC
    svc = LinearSVC()
    # Check the training time for the SVC
    t = time.time()
    svc.fit(X_train, y_train)
    t2= time.time()
    if verbose:
        print(round(t2-t, 2), 'Seconds to train SVC...')
    # Check the score of the SVC
        print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
    # Check the prediction time for a single frame
    t = time.time()
    n_predict=10
    if verbose:
        print('My SVC predicts: ', svc.predict(X_test[0: n_predict]))
        print('For these', n_predict, 'labels: ', y_test[0:n_predict])
    t2 = time.time()
    if verbose:
        print(round(t2-t, 5), 'Seconds to predict', n_predict, 'labels with SVC')

    if return_model:
        model = {}
        model['svc'] = svc
        model['scaler'] = X_scaler
        return model

    return None
