import pandas as pd
import matplotlib.pyplot as plt

# Create a sample dataset
data = pd.DataFrame({
    "StudyHours": [1, 2, 3, 4, 5, 6],
    "Marks": [35, 42, 50, 58, 68, 75]
})

# Display dataset
print("Dataset:")
print(data)

# Calculate correlation
print("\nCorrelation Matrix:")
print(data.corr(numeric_only=True))

# Draw scatter plot
plt.scatter(data["StudyHours"], data["Marks"])

plt.xlabel("Study Hours")
plt.ylabel("Marks")
plt.title("Association Between Study Hours and Marks")
plt.grid(True)

plt.show()