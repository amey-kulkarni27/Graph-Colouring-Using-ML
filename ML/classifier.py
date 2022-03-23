import xgboost as xgb
from sklearn.linear_model import LogisticRegression

class Logistic():
    def __init__(self, rand_state):
        self.rand_state = rand_state
    
    def fit(self, X, y):
        lr = LogisticRegression(random_state=self.rand_state)
        self.clf = lr.fit(X, y)
    
    def predict(self, X, threshold=0.5, probvals=False):
        y = self.clf.predict(X) # For threshold = 0.5 only
        probs = self.clf.predict_proba(X)[:, 1]
        if probvals:
            return probs
        preds = probs > threshold
        return preds

class XGB():
    def __init__(self, rand_state):
        self.rand_state = rand_state
    
    def fit(self, X, y):
        xgb_cl = xgb.XGBClassifier(use_label_encoder=False)
        self.clf = xgb_cl.fit(X, y)

    def predict(self, X, threshold=0.5, probvals=False):
        y = self.clf.predict(X) # For threshold = 0.5 only
        probs = self.clf.predict_proba(X)[:, 1]
        if probvals:
            return probs
        preds = probs > threshold
        return preds

def classifier(clf_name, rand_state=0):
    if clf_name == "logistic":
        return Logistic(rand_state=rand_state)
    elif clf_name == "xgb":
        return XGB(rand_state=0)
    else:
        print("Enter a valid classifier")
        exit
