import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
import pickle
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

shots = pd.read_csv('shot_logs_cleaned.csv')

y = np.ravel(shots[['FGM']])

X_num = shots[['FINAL_MARGIN', 'SHOT_NUMBER', 'SHOT_CLOCK', 'TOUCH_TIME',
       'SHOT_DIST', 'CLOSE_DEF_DIST', 'GAME_TIME']]

scaler =  StandardScaler()

scaler.fit(X_num)
X_procc = scaler.transform(X_num)

X_train, X_test, y_train, y_test = train_test_split(X_procc, y, test_size=0.1, stratify=y)

parameters_for_testing = {
    'min_child_weight':[0.0001,0.001,0.01],
    'learning_rate':[0.00001,0.0001,0.001],
    'n_estimators':[1,3,5,10],
    'max_depth':[3,4]
}

xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')

gsearch1 = GridSearchCV(estimator = xgb_model, param_grid = parameters_for_testing, scoring='precision')

xgb = gsearch1.fit(X_train, y_train)

print(f"Accuracy score for train data: {xgb.score(X_train, y_train)}")
print(f"Accuracy score for test data: {xgb.score(X_test, y_test)}")

disp_matrix = ConfusionMatrixDisplay(confusion_matrix(y_test, xgb.predict(X_test)), display_labels=xgb.classes_)
disp_matrix.plot()
plt.show()


pickle.dump(xgb, open("shots_clsf.pkl", "wb"))
pickle.dump(scaler, open("shots_scaler.pkl", "wb"))