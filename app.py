# import streamlit as st
# import pandas as pd
# from joblib import load
# import pickle
# import numpy as np
# import nltk
# import nltk
# import nltk
# nltk.data.path.append('C:/Users/mehul/AppData/Roaming/nltk_data')  # Or the path where you downloaded the resources


# from nltk.tokenize import word_tokenize

# # Load models
# with open("models/disease_model.pkl", "rb") as model_file:
#     model = pickle.load(model_file)

# with open("models/symptom_list.pkl", "rb") as symptom_file:
#     symptoms = pickle.load(symptom_file)

# # Load symptom description dataset
# symptom_data = pd.read_csv('data/symptom_Description.csv')

# # Convert symptoms to lowercase for case-insensitive matching
# symptoms = [symptom.lower() for symptom in symptoms]

# # Symptom extraction function
# def extract_symptoms(paragraph):
#     tokens = word_tokenize(paragraph.lower())
#     matched_keywords = set(tokens) & set(symptoms)
#     return list(matched_keywords)

# # Prediction function
# def predict_disease(input_symptoms):
#     input_data = [1 if symptom in input_symptoms else 0 for symptom in symptoms]
#     input_data = np.array(input_data).reshape(1, -1)
#     predicted_disease = model.predict(input_data)[0]
    
#     # Get description from the dataset
#     description = symptom_data[symptom_data['Disease'].str.lower() == predicted_disease.lower()]['Description'].values
#     description = description[0] if len(description) > 0 else "Description not available."

#     return predicted_disease, description

# # Auth
# def login():
#     st.title("Login Page")
#     username = st.text_input("Username")
#     password = st.text_input("Password", type="password")
#     if st.button("Login"):
#         if username == "user" and password == "user":
#             return True
#         else:
#             st.error("Invalid username or password")
#     return False

# # Details
# def user_details():
#     st.title("User Details")
#     name = st.text_input("Name")
#     age = st.number_input("Age", min_value=0, max_value=120, step=1)
#     medical_history = st.text_area("Medical History")
#     if st.button("Save Details"):
#         st.success("Details saved successfully!")
#         return name, age, medical_history
#     return None, None, None

# # Chatbot Page
# def chatbot_interface():
#     st.title("CB Chatbot - Disease Prediction")
#     st.write("Enter your symptoms separated by commas (e.g., 'fever, cough, headache') or paste a paragraph describing your symptoms.")

#     # User Input
#     user_input = st.text_area("Your Symptoms or Description:")

#     if user_input:
#         # Check if input is comma-separated symptoms or a paragraph
#         if ',' in user_input:
#             input_symptoms = [symptom.strip().lower() for symptom in user_input.split(",")]
#         else:
#             input_symptoms = extract_symptoms(user_input)

#         valid_symptoms = [symptom for symptom in input_symptoms if symptom in symptoms]

#         if not valid_symptoms:
#             st.error("No recognizable symptoms found. Please try again or use valid symptoms.")
#         else:
#             # Predict disease
#             disease, description = predict_disease(valid_symptoms)
#             st.success(f"You might have: {disease}.")
#             st.info(f"Description: {description}")

#     st.sidebar.title("Help")
#     if st.sidebar.button("Show Symptoms List"):
#         st.sidebar.write(", ".join(symptoms))

# # Main
# def main():
#     st.sidebar.title("Navigation")
#     app_mode = st.sidebar.selectbox("Choose a section", ["Login", "User Details", "Chatbot"])

#     if app_mode == "Login":
#         st.title("Login Page")
#         username = st.text_input("Username")
#         password = st.text_input("Password", type="password")
#         if st.button("Login"):
#             if username == "user" and password == "user":
#                 st.session_state["authenticated"] = True
#                 st.success("Login Successful!")
#             elif username == "mehul" and password == "mehul":
#                 st.session_state["authenticated"] = True
#                 st.success("Login Successful!")
#             else:
#                 st.error("Invalid username or password")

#     if app_mode == "User Details":
#         if "authenticated" not in st.session_state or not st.session_state["authenticated"]:
#             st.warning("Please log in first!")
#         else:
#             user_details()

#     if app_mode == "Chatbot":
#         if "authenticated" not in st.session_state or not st.session_state["authenticated"]:
#             st.warning("Please log in first!")
#         else:
#             chatbot_interface()

# if __name__ == "__main__":
#     main()
# ==================================================================================================================
# 2nd Iteration of working code
# ==================================================================================================================
# import streamlit as st
# import pandas as pd
# from joblib import load
# import pickle
# import numpy as np
# import nltk
# from nltk.tokenize import word_tokenize
# from fpdf import FPDF
# import base64



# nltk.data.path.append('C:/Users/mehul/AppData/Roaming/nltk_data')  # Adjust path as needed

# #load rf model
# with open("models/disease_model.pkl", "rb") as model_file:
#     model = pickle.load(model_file)

# with open("models/symptom_list.pkl", "rb") as symptom_file:
#     symptoms = pickle.load(symptom_file)

# #load datasets
# symptom_data = pd.read_csv('data/symptom_Description.csv')
# precautions_data = pd.read_csv('data/symptom_precaution.csv')

# symptoms = [symptom.lower() for symptom in symptoms]

# #get symptoms from input and match 
# def extract_symptoms(paragraph):
#     tokens = word_tokenize(paragraph.lower())
#     matched_keywords = set(tokens) & set(symptoms)
#     return list(matched_keywords)

# def get_precautions(disease):
#     # Filter the row corresponding to the predicted disease
#     disease_precautions = precautions_data[precautions_data['Disease'].str.lower() == disease.lower()]
    
#     if not disease_precautions.empty:
#         # Extract precautions as a list
#         precautions = [
#             disease_precautions.iloc[0]['Precaution_1'],
#             disease_precautions.iloc[0]['Precaution_2'],
#             disease_precautions.iloc[0]['Precaution_3'],
#             disease_precautions.iloc[0]['Precaution_4']
#         ]
#         return [precaution for precaution in precautions if pd.notna(precaution)]  # Return non-NaN precautions
#     else:
#         return ["Precaution information not available."]

# #predict
# def predict_disease(input_symptoms):
#     input_data = [1 if symptom in input_symptoms else 0 for symptom in symptoms]
#     input_data = np.array(input_data).reshape(1, -1)
#     predicted_disease = model.predict(input_data)[0]

#     description = symptom_data[symptom_data['Disease'].str.lower() == predicted_disease.lower()]['Description'].values
#     description = description[0] if len(description) > 0 else "Description not available."

#     return predicted_disease, description

# #auth
# def login():
#     st.title("Login Page")
#     username = st.text_input("Username")
#     password = st.text_input("Password", type="password")
#     if st.button("Login"):
#         if username == "user" and password == "user":
#             return True
#         else:
#             st.error("Invalid username or password")
#     return False

# # def user_details():
# #     st.title("User Details")
# #     patient_name = st.text_input("Patient Name:")
# #     age = st.number_input("Age:", min_value=0, max_value=120, step=1)
# #     medical_history = st.text_area("Medical History")
# #     if st.button("Save Details"):
# #         st.success("Details saved successfully!")
# #         return patient_name, age, medical_history
# #     return None, None, None
# def download_pdf(patient_name, age, disease, description, precautions):
#     # Create a PDF instance
#         pdf = FPDF()
#         pdf.add_page()
#         pdf.set_font("Arial", size=12)

#         # Add patient details
#         pdf.cell(200, 10, txt="Disease Prediction Report", ln=True, align='C')
#         pdf.ln(10)  # Add a line break
#         pdf.cell(200, 10, txt=f"Patient Name: {patient_name}", ln=True)
#         pdf.cell(200, 10, txt=f"Age: {age}", ln=True)
#         pdf.ln(10)

#         # Add disease and description
#         pdf.cell(200, 10, txt=f"Predicted Disease: {disease}", ln=True)
#         pdf.ln(5)
#         pdf.multi_cell(0, 10, txt=f"Disease Description: {description}")
#         pdf.ln(10)

#         # Add precautions
#         pdf.cell(200, 10, txt="Precautions:", ln=True)
#         for i, precaution in enumerate(precautions, 1):
#             pdf.cell(200, 10, txt=f"{i}. {precaution}", ln=True)

#         # Save the PDF to a temporary file
#         pdf_file = "Disease_Report.pdf"
#         pdf.output(pdf_file)

#         # Convert the file to bytes for downloading in Streamlit
#         with open(pdf_file, "rb") as f:
#             pdf_data = f.read()

#         # Encode the file as a base64 string
#         b64_pdf = base64.b64encode(pdf_data).decode('utf-8')

#         # Create a download link
#         download_link = f'<a href="data:application/octet-stream;base64,{b64_pdf}" download="{pdf_file}">Click here to download your report</a>'
#         st.markdown(download_link, unsafe_allow_html=True)

# def chatbot_interface():

#     patient_name = 'Mehul'
#     age = 22
#     medical_history = 'owhwhwehq'
#     # if st.button("Save Details"):
#     #     st.success("Details saved successfully!")
#     #     return patient_name, age, medical_history

#     st.title("CB Chatbot - Disease Prediction")
#     st.write("Enter your symptoms separated by commas (e.g., 'fever, cough, headache') or paste a paragraph describing your symptoms.")
    
#     user_input = st.text_area("Your Symptoms or Description:")
#     st.sidebar.title("Help")
#     if st.sidebar.button("Show Symptoms List"):
#         st.sidebar.write(", ".join(symptoms))

#     if user_input:
#         input_symptoms = extract_symptoms(user_input)

#         valid_symptoms = [symptom for symptom in input_symptoms if symptom in symptoms]

#         if not valid_symptoms:
#             st.error("Sorry I wasn\'t able to predict the disease, please use the 'Show Symptoms List' Button under the help tab and try to include them in the")
#         else:
#             #predict disease
#             disease, description = predict_disease(valid_symptoms)
#             st.success(f"You might have: {disease}.")
#             st.info(f"Description: {description}")
            
        
#             precautions = get_precautions(disease)

#             # Display precautions
#             st.write("Precautions to take:")
#             for i, precaution in enumerate(precautions, 1):
#                 st.write(f"{i}. {precaution}")

#             # Generate PDF report
#             if patient_name and age:
#                 st.write("Generate your report:")
#                 download_pdf(patient_name, age, disease, description, precautions)
#             else:
#                 st.warning("Please provide your name and age to generate the report.")
        

    
# def main():
#     st.sidebar.title("Navigation")
#     app_mode = st.sidebar.selectbox("Choose a section", ["Login", "User Details", "Chatbot"])

#     if app_mode == "Login":
#         st.title("Login Page")
#         username = st.text_input("Username")
#         password = st.text_input("Password", type="password")
#         if st.button("Login"):
#             if username == "user" and password == "user":
#                 st.session_state["authenticated"] = True
#                 st.success("Login Successful!")
#             elif username == "mehul" and password == "mehul":
#                 st.session_state["authenticated"] = True
#                 st.success("Login Successful!")
#             else:
#                 st.error("Invalid username or password")

#     # if app_mode == "User Details":
#     #     if "authenticated" not in st.session_state or not st.session_state["authenticated"]:
#     #         st.warning("Please log in first!")
#     #     else:
#     #         user_details()

#     if app_mode == "Chatbot":
#         if "authenticated" not in st.session_state or not st.session_state["authenticated"]:
#             st.warning("Please log in first!")
#         else:
#             chatbot_interface()

# if __name__ == "__main__":
#     main()
# ======================================================================================================
# 3rd Iteration for working code (with fuzzy word comparision)
# ======================================================================================================

import streamlit as st
import pandas as pd
from joblib import load
import pickle
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from fpdf import FPDF
import base64
from thefuzz import fuzz
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

#load models
@st.cache_resource
def load_models():
    with open("models/disease_model.pkl", "rb") as model_file:
        model = pickle.load(model_file)
    with open("models/symptom_list.pkl", "rb") as symptom_file:
        symptoms = pickle.load(symptom_file)
    return model, [symptom.lower() for symptom in symptoms]

@st.cache_data
def load_datasets():
    symptom_data = pd.read_csv('data/symptom_Description.csv')
    precautions_data = pd.read_csv('data/symptom_precaution.csv')
    return symptom_data, precautions_data

#symptom matching
def extract_symptoms(paragraph, symptoms, similarity_threshold=0.85):
    #tokenization
    paragraph = paragraph.lower()
    tokens = word_tokenize(paragraph)
    matched_symptoms = set()
    max_symptom_length = max(len(symptom.split('_')) for symptom in symptoms)
    
    for i in range(len(tokens)):
        for window_size in range(1, max_symptom_length + 1):
            if i + window_size > len(tokens):
                break
            window = ' '.join(tokens[i:i + window_size])
            
            #word to word
            clean_window = window.replace(' ', '_')
            if clean_window in symptoms:
                matched_symptoms.add(clean_window)
                continue
            
            #fuzzy matching
            for symptom in symptoms:
                clean_symptom = symptom.replace('_', ' ')
                
                #similarity score
                similarity = fuzz.ratio(window, clean_symptom)
                if similarity >= similarity_threshold * 100:
                    matched_symptoms.add(symptom)
                    continue
                
                #ratio for taking strings
                partial_similarity = fuzz.partial_ratio(window, clean_symptom)
                if partial_similarity >= similarity_threshold * 100:
                    matched_symptoms.add(symptom)
    
    return list(matched_symptoms)

def get_precautions(disease, precautions_data):
    disease_precautions = precautions_data[precautions_data['Disease'].str.lower() == disease.lower()]
    
    if not disease_precautions.empty:
        precautions = [
            disease_precautions.iloc[0]['Precaution_1'],
            disease_precautions.iloc[0]['Precaution_2'],
            disease_precautions.iloc[0]['Precaution_3'],
            disease_precautions.iloc[0]['Precaution_4']
        ]
        return [precaution for precaution in precautions if pd.notna(precaution)]
    return ["Precaution information not available."]

def predict_disease(input_symptoms, model, all_symptoms, symptom_data):
    input_data = [1 if symptom in input_symptoms else 0 for symptom in all_symptoms]
    input_data = np.array(input_data).reshape(1, -1)
    predicted_disease = model.predict(input_data)[0]

    description = symptom_data[symptom_data['Disease'].str.lower() == predicted_disease.lower()]['Description'].values
    description = description[0] if len(description) > 0 else "Description not available."

    return predicted_disease, description

def download_pdf(patient_name, age, disease, description, precautions):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    #put content into pdf , gpt (help)
    pdf.cell(200, 10, txt="Disease Prediction Report", ln=True, align='C')
    pdf.ln(10)
    pdf.cell(200, 10, txt=f"Patient Name: {patient_name}", ln=True)
    pdf.cell(200, 10, txt=f"Age: {age}", ln=True)
    pdf.ln(10)
    pdf.cell(200, 10, txt=f"Predicted Disease: {disease}", ln=True)
    pdf.ln(5)
    pdf.multi_cell(0, 10, txt=f"Disease Description: {description}")
    pdf.ln(10)
    pdf.cell(200, 10, txt="Precautions:", ln=True)
    for i, precaution in enumerate(precautions, 1):
        pdf.cell(200, 10, txt=f"{i}. {precaution}", ln=True)

    #download pdf
    pdf_file = "Disease_Report.pdf"
    pdf.output(pdf_file)
    
    with open(pdf_file, "rb") as f:
        pdf_data = f.read()
    b64_pdf = base64.b64encode(pdf_data).decode('utf-8')
    href = f'<a href="data:application/octet-stream;base64,{b64_pdf}" download="{pdf_file}">Click here to download your report</a>'
    return href

def main():
    st.set_page_config(page_title="Disease Prediction System", layout="wide")
    if 'authenticated' not in st.session_state:
        st.session_state['authenticated'] = False
        
    #load models
    try:
        model, symptoms = load_models()
        symptom_data, precautions_data = load_datasets()
    except Exception as e:
        st.error(f"Error loading models or data: {str(e)}")
        return

    #navbar
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.selectbox("Choose a section", ["Login", "Disease Prediction"])

    if app_mode == "Login":
        st.title("Login Page")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            if (username == "user" and password == "user") or (username == "mehul" and password == "mehul"):
                st.session_state["authenticated"] = True
                st.success("Login Successful!")
            else:
                st.error("Invalid username or password")

    elif app_mode == "Disease Prediction":
        if not st.session_state["authenticated"]:
            st.warning("Please log in first!")
            return
            
        st.title("Disease Prediction System")
        
        #form-details
        col1, col2 = st.columns(2)
        with col1:
            patient_name = st.text_input("Patient Name:")
        with col2:
            age = st.number_input("Age:", min_value=0, max_value=120, step=1)
            
        st.write("Enter your symptoms separated by commas or describe them in a paragraph:")
        user_input = st.text_area("Your Symptoms:")
        
        #help
        st.sidebar.title("Available Symptoms")
        if st.sidebar.checkbox("Show Symptoms List"):
            st.sidebar.write(", ".join(symptoms))

        if user_input:
            input_symptoms = extract_symptoms(user_input, symptoms)

            if not input_symptoms:
                st.error("No matching symptoms found. Please check the symptoms list and try again.")
            else:
                st.write("Identified symptoms:", ", ".join(input_symptoms))
                
                disease, description = predict_disease(input_symptoms, model, symptoms, symptom_data)
                precautions = get_precautions(disease, precautions_data)

                st.success(f"Predicted Disease: {disease}")
                st.info(f"Description: {description}")
                
                st.write("Precautions:")
                for i, precaution in enumerate(precautions, 1):
                    st.write(f"{i}. {precaution}")

                if patient_name and age:
                    st.write("Generate your report:")
                    href = download_pdf(patient_name, age, disease, description, precautions)
                    st.markdown(href, unsafe_allow_html=True)
                else:
                    st.warning("Please provide patient name and age to generate the report.")

if __name__ == "__main__":
    main()