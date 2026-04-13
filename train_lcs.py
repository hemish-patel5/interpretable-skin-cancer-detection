import pandas as pd
import numpy as np
from skExSTraCS import ExSTraCS
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# 1. Load the data you created in the extraction step
df = pd.read_csv('capstone_features.csv')

# 2. Prepare the data
# X = all the feature columns (LBP, HOG, DWT)
# y = the 'target' column (1 or 0)
X = df.drop(['image_id', 'target'], axis=1).values
y = df['target'].values

# 3. Split into Training (80%) and Testing (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Initialize the eLCS Model
# We start with 1000 iterations for a quick "Proof of Concept" for Friday
model = ExSTraCS(learning_iterations=1000, 
                 N=500, # Max number of rules in the population
                 track_accuracy_while_fit=True)

print("Starting LCS Training Loop...")

# 5. Train the model
model.fit(X_train, y_train)

# 6. Get the results for Slide 14
predictions = model.predict(X_test)
print("\n--- PERFORMANCE REPORT ---")
print(classification_report(y_test, predictions))

# 7. Export the "Interpretable" Rules
# This is the 'Outcome' of your project!
model.export_iteration_tracking_data("lcs_rules.csv")
print("\nSuccess! Results generated and rules exported to 'lcs_rules.csv'.")