import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Dataset (Hours, Attendance, Sleep)
X = np.array([
    [2, 60, 6],
    [3, 65, 7],
    [4, 70, 6],
    [5, 75, 7],
    [6, 80, 8],
    [7, 85, 7],
    [8, 90, 8]
])

# Scores
y = np.array([40, 45, 55, 60, 70, 75, 85])

# Train model
model = LinearRegression()
model.fit(X, y)

# User input
hours = float(input("Enter study hours: "))
attendance = float(input("Enter attendance (%): "))
sleep = float(input("Enter sleep hours: "))

# Prediction
prediction = model.predict([[hours, attendance, sleep]])
print("\nPredicted Score:", round(prediction[0], 2))

# Visualization (only study hours vs score for simplicity)
plt.scatter(X[:, 0], y)
plt.plot(X[:, 0], model.predict(X),)
plt.xlabel("Study Hours")
plt.ylabel("Score")
plt.title("Study Hours vs Score (Multi-factor model)")
plt.show()