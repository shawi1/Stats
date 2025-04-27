import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

url = 'https://web.stanford.edu/class/archive/cs/cs109/cs109.1166/stuff/titanic.csv'
df = pd.read_csv(url)

print("\nDataset Information:")
print(df.info())

print("\n📊 First 50 Records:")
print(df.head(50))

df['FamilySize'] = df['Siblings/Spouses Aboard'] + df['Parents/Children Aboard'] + 1

survival_by_family = df.groupby('FamilySize')['Survived'].mean().reset_index()
print("\n📊 Survival Rate by Family Size:")
print(survival_by_family)

plt.figure(figsize=(7,4))
plt.bar(survival_by_family['FamilySize'], survival_by_family['Survived'], color='purple')
plt.title('Survival Rate by Family Size')
plt.xlabel('Family Size')
plt.ylabel('Survival Rate')
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

survival_by_class = df.groupby('Pclass')['Survived'].mean().reset_index()
print("\n📊 Survival Rate by Passenger Class:")
print(survival_by_class)

plt.figure(figsize=(6,4))
plt.bar(survival_by_class['Pclass'], survival_by_class['Survived'], color='orange')
plt.title('Survival Rate by Passenger Class')
plt.xlabel('Passenger Class')
plt.ylabel('Survival Rate')
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

df = df.dropna(subset=['Survived', 'FamilySize'])

# Prepare X and y
X = df[['FamilySize']]  # Predictor must be 2D
y = df['Survived']      

model = LinearRegression()
model.fit(X, y)

# Coefficients
intercept = model.intercept_
slope = model.coef_[0]

print("\n📈 Linear Regression Equation:")
print(f"Survival = {intercept:.4f} + ({slope:.4f}) * Family Size")

# Scatter plot + Regression Line
plt.figure(figsize=(7,5))
plt.scatter(df['FamilySize'], df['Survived'], color='blue', alpha=0.5, label="Actual Data")

# Create points for regression line
x_vals = np.linspace(df['FamilySize'].min(), df['FamilySize'].max(), 100)
y_vals = intercept + slope * x_vals
plt.plot(x_vals, y_vals, color='red', label="Regression Line")

plt.title('Linear Regression: Survival vs Family Size')
plt.xlabel('Family Size')
plt.ylabel('Survival Probability')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()