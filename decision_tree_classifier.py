import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# Load Dataset
df = pd.read_csv("bank.csv")

# Convert target values
df["Purchased"] = df["Purchased"].map({"No": 0, "Yes": 1})

# Features and Target
X = df[["Age", "Salary", "Previous_Purchases"]]
y = df["Purchased"]

# Split Data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42
)

# Train Model
model = DecisionTreeClassifier(
    criterion="gini",
    max_depth=3,
    random_state=42
)

model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

# ---------------------------------------------------
# Decision Tree Visualization
# ---------------------------------------------------
plt.figure(figsize=(14,8))

plot_tree(
    model,
    feature_names=X.columns,
    class_names=["No", "Yes"],
    filled=True,
    rounded=True,
    fontsize=10
)

plt.title("Decision Tree Classifier")
plt.savefig("decision_tree.png", bbox_inches="tight")
plt.close()

# ---------------------------------------------------
# Confusion Matrix Visualization
# ---------------------------------------------------
plt.figure(figsize=(6,5))

plt.imshow(cm)

plt.title("Confusion Matrix")
plt.colorbar()

plt.xticks([0,1], ["No", "Yes"])
plt.yticks([0,1], ["No", "Yes"])

for i in range(len(cm)):
    for j in range(len(cm[0])):
        plt.text(
            j,
            i,
            cm[i,j],
            ha="center",
            va="center",
            fontsize=12
        )

plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.savefig("confusion_matrix.png")
plt.close()

# ---------------------------------------------------
# Feature Importance
# ---------------------------------------------------
importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

plt.figure(figsize=(8,5))

plt.bar(
    importance["Feature"],
    importance["Importance"]
)

plt.title("Feature Importance")
plt.xlabel("Features")
plt.ylabel("Importance Score")

plt.tight_layout()
plt.savefig("feature_importance.png")
plt.close()

# ---------------------------------------------------
# HTML Dashboard
# ---------------------------------------------------
html = f"""
<!DOCTYPE html>
<html>

<head>

<title>Customer Purchase Prediction Dashboard</title>

<style>

body {{
    margin:0;
    background:#eef2f7;
    font-family:Segoe UI;
}}

.header {{
    background:linear-gradient(135deg,#2563eb,#7c3aed);
    color:white;
    text-align:center;
    padding:35px;
}}

.container {{
    width:90%;
    margin:auto;
}}

.card {{
    background:white;
    margin:20px 0;
    padding:20px;
    border-radius:15px;
    box-shadow:0 5px 15px rgba(0,0,0,0.1);
}}

.stats {{
    display:flex;
    justify-content:space-around;
    flex-wrap:wrap;
}}

.stat {{
    background:#f8fafc;
    padding:20px;
    border-radius:10px;
    width:220px;
    text-align:center;
    margin:10px;
}}

.stat h2 {{
    color:#2563eb;
}}

img {{
    width:100%;
    border-radius:10px;
}}

.footer {{
    text-align:center;
    color:gray;
    padding:20px;
}}

</style>

</head>

<body>

<div class="header">
<h1>Customer Purchase Prediction Dashboard</h1>
<p>Decision Tree Machine Learning Project</p>
</div>

<div class="container">

<div class="card">
<h2>Model Summary</h2>

<div class="stats">

<div class="stat">
<h2>{accuracy*100:.2f}%</h2>
<p>Accuracy</p>
</div>

<div class="stat">
<h2>{len(df)}</h2>
<p>Total Customers</p>
</div>

<div class="stat">
<h2>{sum(df['Purchased'])}</h2>
<p>Customers Purchased</p>
</div>

</div>

</div>

<div class="card">
<h2>Decision Tree Structure</h2>
<img src="decision_tree.png">
</div>

<div class="card">
<h2>Confusion Matrix</h2>
<img src="confusion_matrix.png">
</div>

<div class="card">
<h2>Feature Importance</h2>
<img src="feature_importance.png">
</div>

<div class="card">
<h2>Key Insights</h2>

<ul>
<li>Salary is an important factor influencing purchases.</li>
<li>Customers with more previous purchases tend to buy again.</li>
<li>The Decision Tree model successfully classifies customer behavior.</li>
<li>Age also contributes to purchase prediction.</li>
</ul>

</div>

</div>

<div class="footer">
Prodigy InfoTech Internship Project - Task 03
</div>

</body>
</html>
"""

with open("decision_tree_report.html", "w", encoding="utf-8") as file:
    file.write(html)

print("="*50)
print("PROJECT COMPLETED SUCCESSFULLY")
print("Open decision_tree_report.html from File Explorer")
print("="*50)