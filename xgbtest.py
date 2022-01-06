import xgboost as xgb

# Show all messages, including debugging related ones
# xgb.set_config(verbosity=2)

# config = xgb.get_config()
# assert config['verbosity'] == 2

def xgboost_clf(X, y):
    xgb_clf = xgb.XGBClassifier()
    print(type(xgb_clf))

xgboost_clf(1, 2)