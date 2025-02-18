import pandas as pd
import matplotlib.pyplot as plt

url = 'https://web.stanford.edu/class/archive/cs/cs109/cs109.1166/stuff/titanic.csv'
df = pd.read_csv(url)

print("\nDataset Information:")
print(df.info())

print("\n📊 First 50 Records:")
print(df.head(50))

# family size column
df['FamilySize'] = df['Siblings/Spouses Aboard'] + df['Parents/Children Aboard'] + 1

# survival rates by family size
survival_by_family = df.groupby('FamilySize')['Survived'].mean().reset_index()
print("\n📊 Survival Rate by Family Size:")
print(survival_by_family)

# histogram creation for q1
plt.figure(figsize=(7,4))
plt.bar(survival_by_family['FamilySize'], survival_by_family['Survived'], color='purple')
plt.title('Survival Rate by Family Size')
plt.xlabel('Family Size')
plt.ylabel('Survival Rate')
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

# survival rates by passenger class
survival_by_class = df.groupby('Pclass')['Survived'].mean().reset_index()
print("\n📊 Survival Rate by Passenger Class:")
print(survival_by_class)

# histogram created for q2
plt.figure(figsize=(6,4))
plt.bar(survival_by_class['Pclass'], survival_by_class['Survived'], color='orange')
plt.title('Survival Rate by Passenger Class')
plt.xlabel('Passenger Class')
plt.ylabel('Survival Rate')
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()
