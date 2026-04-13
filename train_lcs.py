import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder
from skeLCS import eLCS

df = pd.read_csv("capstone_features.csv")

feature_cols = [col for col in df.columns if col != 'image_id' and col != 'label']
X = df[feature_cols].values
y = df["label"].values

le = LabelEncoder()
y_encoded = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)

print(f"Training samples: {len(X_train)}")
print(f"Testing samples:  {len(X_test)}")
print("Training eLCS — please wait...")

model = eLCS(
    learning_iterations=2000,
    N=500,
    nu=5,
    chi=0.8,
    mu=0.04,
    theta_del=20,
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

bal_acc = balanced_accuracy_score(y_test, y_pred)
prec    = precision_score(y_test, y_pred, zero_division=0)
rec     = recall_score(y_test, y_pred, zero_division=0)

print("\n=== RESULTS ===")
print(f"Balanced Accuracy : {bal_acc * 100:.1f}%")
print(f"Precision         : {prec * 100:.1f}%")
print(f"Recall            : {rec * 100:.1f}%")