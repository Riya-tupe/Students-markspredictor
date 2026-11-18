import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error, accuracy_score

st.set_page_config(page_title="Student Marks Predictor",
                   page_icon="📚")

st.title("📚 Student Marks Predictor")
st.write("Predict student performance using Machine Learning.")

# Generate random data
np.random.seed(10)
n = 100

hours = np.random.randint(1, 9, n)
attendance = np.random.randint(50, 101, n)
previous = np.random.randint(30, 91, n)

marks = np.clip(5*hours + 0.25*attendance +
                0.35*previous +
                np.random.normal(0, 5, n), 0, 100)

X = np.column_stack((hours, attendance, previous))
y = marks
passed = (marks >= 40).astype(int)

X_train, X_test, y_train, y_test, p_train, p_test = train_test_split(
    X, y, passed, test_size=0.2, random_state=42
)

# Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
pred = lr.predict(X_test)

# Logistic Regression
log = make_pipeline(StandardScaler(),
                    LogisticRegression())
log.fit(X_train, p_train)

# SVM
svm = make_pipeline(StandardScaler(),
                    SVC(kernel="linear"))
svm.fit(X_train, p_train)

# K-Means
km = KMeans(n_clusters=3, random_state=10, n_init=10)
km.fit(X)

st.subheader("Enter Student Details")

h = st.slider("Study Hours per Day", 1, 8, 5)
a = st.slider("Attendance (%)", 50, 100, 80)
p = st.slider("Previous Marks", 30, 90, 65)

if st.button("Predict Marks"):
    student = np.array([[h, a, p]])

    result = lr.predict(student)[0]
    logistic = log.predict(student)[0]
    support = svm.predict(student)[0]
    group = km.predict(student)[0] + 1

    st.subheader("Prediction Results")

    st.metric("Predicted Marks", str(round(result, 2)) + " / 100")

    if logistic == 1:
        st.success("Logistic Regression: PASS")
    else:
        st.error("Logistic Regression: FAIL")

    st.write("SVM Result:",
             "PASS" if support == 1 else "FAIL")

    st.write("Student Group:", group)

    st.progress(int(np.clip(result, 0, 100)))

st.divider()

st.subheader("Model Performance")

c1, c2 = st.columns(2)

c1.metric("Linear Regression MAE",
          round(mean_absolute_error(y_test, pred), 2))

c2.metric("Logistic Accuracy",
          str(round(accuracy_score(
              p_test, log.predict(X_test))*100, 2)) + "%")

st.write("SVM Accuracy:",
         round(accuracy_score(
             p_test, svm.predict(X_test))*100, 2), "%")

st.subheader("Random Student Dataset")

df = pd.DataFrame({
    "Study Hours": hours,
    "Attendance": attendance,
    "Previous Marks": previous,
    "Final Marks": np.round(marks, 2)
})

st.dataframe(df.head(10), use_container_width=True)
  
