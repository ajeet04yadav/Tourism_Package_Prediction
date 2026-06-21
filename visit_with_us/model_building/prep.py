
# for data manipulation
import pandas as pd
import sklearn

# for creating a folder
import os

# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split

# for converting text data into numerical representation
from sklearn.preprocessing import LabelEncoder

# for hugging face space authentication to upload files
from huggingface_hub import HfApi

# Initialize Hugging Face API
api = HfApi(token=os.getenv("HF_TOKEN"))

# Load dataset directly from Hugging Face Dataset Repository
DATASET_PATH = "hf://datasets/ajeet04yadav/Tourism-Package-Prediction/tourism.csv"

df = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully.")

# Remove unnecessary column
df.drop(columns=["CustomerID"], inplace=True)

# Handle missing values
for col in df.select_dtypes(include=["int64", "float64"]).columns:
    if df[col].isnull().sum() > 0:
        df[col] = df[col].fillna(df[col].median())

for col in df.select_dtypes(include=["object"]).columns:
    if df[col].isnull().sum() > 0:
        df[col] = df[col].fillna(df[col].mode()[0])

# Encode categorical columns
categorical_cols = df.select_dtypes(include=["object"]).columns

label_encoder = LabelEncoder()

for col in categorical_cols:
    df[col] = label_encoder.fit_transform(df[col])

target_col = "ProdTaken"

# Split into X (features) and y (target)
X = df.drop(columns=[target_col])
y = df[target_col]

# Perform train-test split
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Save locally
Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

# Upload files back to Hugging Face Dataset Repository
files = ["Xtrain.csv", "Xtest.csv", "ytrain.csv", "ytest.csv"]

for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path,
        repo_id="ajeet04yadav/Tourism-Package-Prediction",
        repo_type="dataset",
    )

print("Train and test datasets uploaded successfully.")
