import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from ucimlrepo import fetch_ucirepo 

dry_bean_dataset = fetch_ucirepo(id=602) 
# Targets are the beans names, we convert it to 1D array for sklearn
X = dry_bean_dataset.data.features
y = dry_bean_dataset.data.targets
y = y.values.ravel()

le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Dictionary to compare labels and their encoded values
label_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
print(f"   Label Mapping: {label_mapping}")

#Stratify to maintain even split (bombay was onl 10%) 
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.25, random_state=42, stratify=y_encoded)

scaler = StandardScaler()

# we are normalizing the training data
# if we do not do this, we would be introducing test set to training data, which is data leakage => ovefitting!
# scaler stores the stats after using fit, so we can just use transform afterwards.
scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

columns = X.columns
X_train_df = pd.DataFrame(X_train_scaled, columns=columns)
X_test_df = pd.DataFrame(X_test_scaled, columns=columns)
y_train_df = pd.DataFrame(y_train, columns=['Class'])
y_test_df = pd.DataFrame(y_test, columns=['Class'])

X_train_df.to_csv('X_train.csv', index=False)
X_test_df.to_csv('X_test.csv', index=False)
y_train_df.to_csv('y_train.csv', index=False)
y_test_df.to_csv('y_test.csv', index=False)

