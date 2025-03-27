import streamlit as st
from streamlit_utilities import read_csv, read_image, read_model
import numpy as np

st.title("page_B5")



st.title("Reading CSV file")
try:
    df = read_csv('customer_engagement.csv')
    st.write("DataFrame from customer_engagement.csv:")
    st.dataframe(df)
except FileNotFoundError as e:
    st.error(str(e))

st.title("Display png image")
try:
    image = read_image('apple.png')
    st.write("Image from apple.png:")
    st.image(image)
except FileNotFoundError as e:
    st.error(str(e))



# read the saved model (.pkl file)
model = read_model('logistic_regression_model.pkl')

st.title("Dummmy Logistic Regression Model Predictor")
st.write("""
Adjust the sliders to set feature values and see the model's prediction in real-time.
The model was trained on randomly generated data with 5 features.
""")

# Create sliders for each feature in the sidebar
st.sidebar.header("Feature Controls")
sliders = []
for i in range(5):
    # Using normal distribution range (-3 to 3) since data was randomly generated
    slider = st.sidebar.slider(
        label=f"Feature {i + 1}",
        min_value=-3.0,
        max_value=3.0,
        value=0.0,
        step=0.1,
        key=f"feature_{i}"
    )
    sliders.append(slider)

# Convert slider values to numpy array for prediction
input_data = np.array(sliders).reshape(1, -1)

# Make prediction
prediction = model.predict(input_data)[0]
probabilities = model.predict_proba(input_data)[0]

# Display results
st.subheader("Prediction Results")

# Show prediction with color (red for class 0, green for class 1)
if prediction == 1:
    st.success(f"Predicted Class: {prediction} (Positive)")
else:
    st.error(f"Predicted Class: {prediction} (Negative)")

# Show probability bar chart
st.write("Class Probabilities:")
st.bar_chart({
    "Class 0": probabilities[0],
    "Class 1": probabilities[1]
})

# Show raw probabilities
st.write(f"Probability of Class 0: {probabilities[0]:.4f}")
st.write(f"Probability of Class 1: {probabilities[1]:.4f}")

# Show the input values
st.subheader("Current Input Values")
for i, value in enumerate(sliders):
    st.write(f"Feature {i + 1}: {value:.2f}")
