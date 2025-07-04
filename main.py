import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import time

from pytorch_tabnet.tab_model import TabNetClassifier
from xgboost import XGBClassifier

data = load_breast_cancer()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
X_test_scaled = scaler.transform(X_test).astype(np.float32)

y_train = y_train.astype(int)
y_test = y_test.astype(int)

print("Training samples:", X_train.shape[0])
print("Features:", X_train.shape[1])
print("Class distribution:", np.bincount(y_train))

# ========== TabNet ==========
start_tabnet = time.time()
clf_tabnet = TabNetClassifier()
clf_tabnet.fit(
    X_train_scaled, y_train,
    eval_set=[(X_test_scaled, y_test)],
    eval_name=['val'],
    eval_metric=['accuracy'],
    max_epochs=100,
    patience=10,
    batch_size=64,
    virtual_batch_size=16,
    num_workers=0,
    drop_last=False,
)
end_tabnet = time.time()
y_pred_tabnet = clf_tabnet.predict(X_test_scaled)
acc_tabnet = accuracy_score(y_test, y_pred_tabnet)
print(f"TabNet Accuracy: {acc_tabnet:.4f} | Time: {end_tabnet - start_tabnet:.2f} sec")

# ========== XGBoost ==========
start_xgb = time.time()
clf_xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
clf_xgb.fit(X_train, y_train)
end_xgb = time.time()
y_pred_xgb = clf_xgb.predict(X_test)
acc_xgb = accuracy_score(y_test, y_pred_xgb)
print(f"XGBoost Accuracy: {acc_xgb:.4f} | Time: {end_xgb - start_xgb:.2f} sec")

# ========== Plot Comparison ==========
plt.bar(['TabNet', 'XGBoost'], [acc_tabnet, acc_xgb], color=['skyblue', 'orange'])
plt.ylabel('Accuracy')
plt.title('Model Comparison: Breast Cancer Dataset')
plt.ylim(0.8, 1.0)
plt.show()
