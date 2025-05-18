import pandas as pd  
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv(r"Data\Original Data\healthcare-dataset-stroke-data.csv")

df.drop(columns=["id"], inplace=True)

label = LabelEncoder()
df["gender"] = label.fit_transform(df["gender"])
df["ever_married"] = label.fit_transform(df["ever_married"])
df["work_type"] = label.fit_transform(df["work_type"])
df["Residence_type"] = label.fit_transform(df["Residence_type"])
df["smoking_status"] = label.fit_transform(df["smoking_status"])

print(df.isnull().sum())
# there is 201 missing value of bmi column 

df['bmi'] = df['bmi'].fillna(df['bmi'].mean())

X = df.drop(columns=["stroke"])
y = df["stroke"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

pd.DataFrame(X_train_scaled, columns=X_train.columns).to_csv("Data/Preprocessed Data/X_train.csv", index=False)
pd.DataFrame(X_test_scaled, columns=X_test.columns).to_csv("Data/Preprocessed Data/X_test.csv", index=False)
y_train.to_csv("Data/Preprocessed Data/Y_train.csv", index=False)
y_test.to_csv("Data/Preprocessed Data/Y_test.csv", index=False)

models = {
    "DecisionTree": DecisionTreeClassifier(random_state=42, class_weight='balanced'),
    "RandomForest": RandomForestClassifier(random_state=42, class_weight='balanced'),
    "KNN": KNeighborsClassifier(),
    "SVM": SVC(probability=True),
    "NaiveBayes": GaussianNB(),
    "ANN": MLPClassifier(max_iter=1000, random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42)
}

metrics = ["precision", "recall", "f1-score"]
results = {metric: {} for metric in metrics}

for name, model in models.items():
    print(f"\n=== Training {name} ===")
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    pd.DataFrame({"Prediction": y_pred}).to_csv(f"Data/Results/prediction_{name}.csv", index=False)

    report = classification_report(y_test, y_pred, target_names=["No Stroke", "Stroke"], zero_division=0)
    print(report)

    report_dict = classification_report(y_test, y_pred, target_names=["No Stroke", "Stroke"], output_dict=True, zero_division=0)
    stroke_metrics = report_dict["No Stroke"]
    for metric in metrics:
        results[metric][name] = stroke_metrics[metric]

results_df = pd.DataFrame(results)

plt.figure(figsize=(12, 6))
sns.barplot(data=results_df.reset_index().melt(id_vars="index"), x="index", y="value", hue="variable")
plt.xlabel("Model")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.title("Model Performance on Stroke Class (Class 0)")
plt.legend(title="Metric")
plt.tight_layout()
plt.show()

