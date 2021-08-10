import pandas as pd
pd.set_option('display.max_columns', None)

penguins = pd.read_csv("penguins_cleaned.csv")
# print(penguins)
# print(penguins.columns)

# Ordinal Feature Encoding
df = penguins.copy()
target = "species"
encode = ["sex", "island"]

for col in encode:
    dummy = pd.get_dummies(df[col], prefix=col)
    df = pd.concat([df, dummy], axis=1)
    del df[col]
# print(df)

# print(df["species"].unique())
target_mapper = {"Adelie":0, "Chinstrap":1, "Gentoo":2}
def target_encode(val):
    return target_mapper[val]

df["species"] = df["species"].apply(target_encode)
print(df)

# Initialize X & Y
X = df.drop("species", axis=1)
y = df["species"]
print(X, y)
# Build Random Forest model
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()
clf.fit(X, y)


# Saving model
import pickle
pickle.dump(clf, open("penguin.pkl", "wb"))
