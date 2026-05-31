import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Step 1: Load the real dataset
df = pd.read_csv("spam.tsv", sep="\t", header=None, names=["label", "email"])

print(f"Total emails in dataset: {len(df)}")
print(f"Spam emails: {len(df[df['label'] == 'spam'])}")
print(f"Normal emails: {len(df[df['label'] == 'ham'])}")
print()

# Step 2: Split into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    df["email"], df["label"], test_size=0.2, random_state=42
)

# Step 3: Convert text into numbers
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Step 4: Train the model
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Step 5: Test accuracy
predictions = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, predictions)
print(f"Model accuracy: {accuracy * 100:.1f}%")
print()
print(classification_report(y_test, predictions))

# Step 6: Try your own emails
test_emails = [
    "Win a free iPhone now click here",
    "Hey are you coming to class tomorrow",
    "Congratulations you won a cash prize call now",
    "Can we meet at the library at 5pm",
    "URGENT your account has been compromised click here",
]

print("--- Testing custom emails ---")
for email in test_emails:
    vec = vectorizer.transform([email])
    result = model.predict(vec)[0]
    print(f'"{email}" → {result.upper()}')