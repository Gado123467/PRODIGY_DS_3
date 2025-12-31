import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

# -------------------------------
# 1. Load Dataset (FIXED)
# -------------------------------
df = pd.read_csv("bank.csv", sep=';')

print("Columns in dataset:")
print(df.columns)

# -------------------------------
# 2. Target Column
# -------------------------------
target_column = 'y'   # customer purchase (yes/no)

# -------------------------------
# 3. Encode Categorical Variables
# -------------------------------
encoder = LabelEncoder()
for col in df.select_dtypes(include='object').columns:
    df[col] = encoder.fit_transform(df[col])

# -------------------------------
# 4. Features & Target
# -------------------------------
X = df.drop(target_column, axis=1)
y = df[target_column]

# -------------------------------
# 5. Train-Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# 6. Decision Tree Model
# -------------------------------
model = DecisionTreeClassifier(
    criterion='gini',
    max_depth=5,
    random_state=42
)

model.fit(X_train, y_train)

# -------------------------------
# 7. Predictions
# -------------------------------
y_pred = model.predict(X_test)

# -------------------------------
# 8. Evaluation
# -------------------------------
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
