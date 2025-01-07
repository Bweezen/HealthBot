import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import accuracy_score
import pickle

data = pd.read_csv("data/symptoms.csv")

symptom_columns = [col for col in data.columns if col.startswith("Symptom")]
unique_symptoms = (
    pd.melt(data[symptom_columns].apply(lambda x: x.str.strip().str.lower()))
    .dropna()["value"]
    .unique()
    .tolist()
)

diseases = data["Disease"].unique()

data["Symptom List"] = data[symptom_columns].apply(
    lambda row: [symptom for symptom in row.dropna().str.strip().str.lower()], axis=1
)

mlb = MultiLabelBinarizer(classes=unique_symptoms)
X = mlb.fit_transform(data["Symptom List"])

y = data["Disease"]

#split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#train
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

train_accuracy = accuracy_score(y_train, model.predict(X_train))
test_accuracy = accuracy_score(y_test, model.predict(X_test))

print(f"Training Accuracy: {train_accuracy * 100:.2f}%")
print(f"Testing Accuracy: {test_accuracy * 100:.2f}%")

#save rand forest model and pickl
with open("models/disease_model.pkl", "wb") as file:
    pickle.dump(model, file)

with open("models/symptom_list.pkl", "wb") as file:
    pickle.dump(unique_symptoms, file)

print("Model trained and saved!")