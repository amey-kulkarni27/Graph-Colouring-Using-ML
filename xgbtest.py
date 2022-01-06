from sklearn import datasets
import xgboost as xgb
from sklearn.model_selection import train_test_split
from classifier import Logistic

def xgboost_clf(X, y):
    xgb_clf = xgb.XGBClassifier(use_label_encoder=False)
    xgb_clf.fit(Xtrain, ytrain)
    preds = xgb_clf.predict_proba(Xtest)
    print(preds[:5])
    # print(sum(ytest==preds) / len(preds))
    # lr = Logistic(rand_state=0)
    # lr.fit(Xtrain, ytrain)
    # predsl = lr.predict(Xtest)
    # print(sum(ytest==predsl) / len(predsl))

X,y = datasets.load_digits(return_X_y=True)
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y)
xgboost_clf(X, y)