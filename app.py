
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error, accuracy_score

# Generate random student data
np.random.seed(10)
n = 100

hours = np.random.randint(1, 9, n)
attendance = np.random.randint(50, 101, n)
previous = np.random.randint(30, 91, n)

noise = np.random.normal(0, 5, n)
marks = np.clip(5*hours + 0.25*attendance +
                 0.35*previous + noise, 0, 100)

X = np.column_stack((hours, attendance, previous))
y = marks
passed = (marks >= 40).astype(int)

# Split data
X_train, X_test, y_train, y_test, p_train, p_test = train_test_split(
    X, y, passed, test_size=0.2, random_state=42
)

# Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
pred = lr.predict(X_test)

print("Linear Regression MAE:",
      round(mean_absolute_error(y_test, pred), 2))

# Logistic Regression
log_model = make_pipeline(
    StandardScaler(), LogisticRegression(random_state=10)
)
log_model.fit(X_train, p_train)

print("Logistic Regression Accuracy:",
      round(accuracy_score(p_test, log_model.predict(X_test))*100, 2), "%")

# Support Vector Machine
svm_model = make_pipeline(
    StandardScaler(), SVC(kernel="linear")
)
svm_model.fit(X_train, p_train)

print("SVM Accuracy:",
      round(accuracy_score(p_test, svm_model.predict(X_test))*100, 2), "%")

# K-Means Clustering
kmeans = KMeans(n_clusters=3, random_state=10, n_init=10)
kmeans.fit(X)

print("K-Means: Students divided into",
      len(np.unique(kmeans.labels_)), "groups")

# Predict marks for a new student
new_student = np.array([[5, 80, 65]])

predicted_marks = lr.predict(new_student)[0]
pass_result = log_model.predict(new_student)[0]
svm_result = svm_model.predict(new_student)[0]
group = kmeans.predict(new_student)[0] + 1

print("\nNew Student Details:")
print("Study Hours: 5")
print("Attendance: 80%")
print("Previous Marks: 65")

print("Predicted Marks:", round(predicted_marks, 2), "/ 100")
print("Logistic Regression Result:",
      "Pass" if pass_result == 1 else "Fail")
print("SVM Result:", "Pass" if svm_result == 1 else "Fail")
print("K-Means Student Group:", group)
