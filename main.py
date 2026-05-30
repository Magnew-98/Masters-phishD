import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm

from src.agents.binary_agent_graph import app


# LOAD DATASET
df = pd.read_csv("datasets/raw/enron.csv")


df["text"] = (
    df["subject"].fillna("") +
    "\n\n" +
    df["body"].fillna("")
)

df["label"] = df["label"].map({
    0: "legitimate",
    1: "phishing"
})

# RANDOMIZE
df = df.sample(frac=1, random_state=42).reset_index(drop=True)


# TAKE FIRST 100
df = df.head(100)


predictions = []
true_labels = []


for _, row in tqdm(df.iterrows(), total=len(df)):

    email_text = row["text"]

    true_label = row["label"]

    result = app.invoke({
        "email": email_text
    })

    prediction = result["prediction"]

    predictions.append(prediction)
    true_labels.append(true_label)

    print("\n========================")
    print("TRUE:", true_label)
    print("PRED:", prediction)
    print("CONF:", result["confidence"])
    print("========================")


# METRICS
accuracy = accuracy_score(true_labels, predictions)

print("\nFINAL ACCURACY")
print(accuracy)

print("\nCLASSIFICATION REPORT")
print(classification_report(true_labels, predictions))