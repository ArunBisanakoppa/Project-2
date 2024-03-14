import streamlit as st
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

# Load Naive Bayes Model
model = joblib.load('naive_bayes_model.pkl')

# Load the TfidfVectorizer used during training
word_vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Load LabelEncoder used during training
lb = joblib.load('label_encoder.pkl')

# Set Streamlit App Title and Icon
st.set_page_config(page_title='Resume Role Prediction', page_icon=':clipboard:')

# Streamlit UI Header
st.title('Resume Role Prediction')
st.write('Enter your resume text and click the "Predict Role" button.')

# User Input TextArea
resume_text = st.text_area('Resume Text:', height=200)

# Prediction Button with Custom Styling
if st.button('Predict Role', key='predict_button', help='Click to predict the role'):
    # Check if text is entered
    if resume_text:
        # Vectorize the input using the same vocabulary
        input_features = word_vectorizer.transform([resume_text])

        # Predict using the model
        prediction_code = model.predict(input_features)[0]

# Map the predicted code back to the original role name
        predicted_role = lb.inverse_transform([prediction_code])[0]

        # Display Prediction Result with Styling
        st.success(f'Predicted Role: {predicted_role}')

