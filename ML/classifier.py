from sklearn.linear_model import LogisticRegression

class Logistic():
    def __init__(self, rand_state):
        self.rand_state = rand_state
    
    def fit(self, X, y):
        lr = LogisticRegression(random_state=self.rand_state)
        self.clf = lr.fit(X, y)
    
    def predict(self, X, threshold=0.5):
        y = self.clf.predict(X) # For threshold = 0.5 only
        probs = self.clf.predict_proba(X)[:, 1]
        preds = probs > threshold
        return preds