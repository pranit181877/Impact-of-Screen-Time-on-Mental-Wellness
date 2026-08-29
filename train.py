import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score

# ✅ MODELS
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

# ---------------- LOAD DATA ---------------- #

df = pd.read_csv("Labeled_ScreenTime_MentalWellness.csv")

# ---------------- CLEAN DATA ---------------- #

df = df.drop(columns=["user_id", "Unnamed: 15"], errors='ignore')

# ---------------- CREATE TARGET ---------------- #

df["wellness_label"] = pd.cut(
    df["mental_wellness_index_0_100"],
    bins=[0, 40, 70, 101],
    labels=["Low", "Medium", "High"],
    include_lowest=True
)

# ---------------- ENCODING ---------------- #

le_gender = LabelEncoder()
le_occ = LabelEncoder()
le_work = LabelEncoder()
le_target = LabelEncoder()

df["gender"] = le_gender.fit_transform(df["gender"])
df["occupation"] = le_occ.fit_transform(df["occupation"])
df["work_mode"] = le_work.fit_transform(df["work_mode"])
df["wellness_label"] = le_target.fit_transform(df["wellness_label"])

# ---------------- FEATURES ---------------- #

X = df[[
    "age",
    "gender",
    "occupation",
    "work_mode",
    "screen_time_hours",
    "sleep_hours",
    "stress_level_0_10",
    "exercise_minutes_per_week",
    "social_hours_per_week"
]]

y = df["wellness_label"]

# ---------------- SPLIT ---------------- #

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------- SCALING ---------------- #

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ---------------- TRAIN MODELS ---------------- #

model_lr = LogisticRegression(max_iter=1000)
model_rf = RandomForestClassifier()
model_svm = SVC()
model_knn = KNeighborsClassifier()
model_dt = DecisionTreeClassifier()

model_lr.fit(X_train, y_train)
model_rf.fit(X_train, y_train)
model_svm.fit(X_train, y_train)
model_knn.fit(X_train, y_train)
model_dt.fit(X_train, y_train)

# ---------------- ACCURACY ---------------- #

acc_lr = accuracy_score(y_test, model_lr.predict(X_test))
acc_rf = accuracy_score(y_test, model_rf.predict(X_test))
acc_svm = accuracy_score(y_test, model_svm.predict(X_test))
acc_knn = accuracy_score(y_test, model_knn.predict(X_test))
acc_dt = accuracy_score(y_test, model_dt.predict(X_test))

accuracy_dict = {
    "Logistic Regression": acc_lr,
    "Random Forest": acc_rf,
    "SVM": acc_svm,
    "KNN": acc_knn,
    "Decision Tree": acc_dt
}

# ---------------- SAVE FILES ---------------- #

pickle.dump(model_lr, open("model_lr.pkl", "wb"))
pickle.dump(model_rf, open("model_rf.pkl", "wb"))
pickle.dump(model_svm, open("model_svm.pkl", "wb"))
pickle.dump(model_knn, open("model_knn.pkl", "wb"))
pickle.dump(model_dt, open("model_dt.pkl", "wb"))

pickle.dump(scaler, open("scaler.pkl", "wb"))
pickle.dump(le_gender, open("le_gender.pkl", "wb"))
pickle.dump(le_occ, open("le_occ.pkl", "wb"))
pickle.dump(le_work, open("le_work.pkl", "wb"))
pickle.dump(le_target, open("le_target.pkl", "wb"))

pickle.dump(accuracy_dict, open("accuracy.pkl", "wb"))

# ---------------- DONE ---------------- #

print("✅ ALL MODELS TRAINED & SAVED SUCCESSFULLY")
print("📊 Model Accuracies:")
for k, v in accuracy_dict.items():
    print(f"{k}: {v*100:.2f}%")