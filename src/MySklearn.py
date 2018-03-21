import numpy as np
from sklearn.cross_validation import KFold
from sklearn import metrics

SEED = 24 # for reproducibility
NFOLDS = 5 # set folds for out-of-fold prediction

class SklearnHelper(object):
    def __init__(self, clf, seed=SEED, params=None, name=""):
        self.clf = clf(**params)
        self.name = name

    def train(self, x_train, y_train):
        self.clf.fit(x_train, y_train)

    def predict(self, x):
        return self.clf.predict(x)
    
    def fit(self,x,y):
        return self.clf.fit(x,y)
    
    def feature_importances(self,x,y):
        print(self.clf.fit(x,y).feature_importances_)


def get_oof(clf, x_train, y_train, x_test):
    ntrain = x_train.shape[0]
    ntest = x_test.shape[0]
    oof_train = np.zeros((ntrain,))
    oof_test = np.zeros((ntest,))
    oof_test_skf = np.empty((NFOLDS, ntest))
    
    kf = KFold(ntrain, n_folds= NFOLDS, random_state=SEED)
    for i, (train_index, test_index) in enumerate(kf):
        x_tr = x_train[train_index]
        y_tr = y_train[train_index]
        x_te = x_train[test_index]

        clf.train(x_tr, y_tr)

        oof_train[test_index] = clf.predict(x_te)
        oof_test_skf[i, :] = clf.predict(x_test)

    print(clf.name, "%.3f" % metrics.accuracy_score(y_train, clf.predict(x_train)))
    oof_test[:] = oof_test_skf.mean(axis=0)
    
    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)