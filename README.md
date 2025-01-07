This is the health bot 

Usage- To Identify diseases based on symptoms provided by the user, this involves NLP

Stuff Used - Python , NLTK , Streamlit , Fuzzywuzzy , scikit and a few more machine learning tools

To Run :
 1. Run model.py , this will create pickle files in the models folder for you , namely disease_model.pkl and symptom_list.pkl
 python model.py
 2. Run app.py for the Streamlit front end,
 Streamlit run app.py

 the main model is disease_model.pkl , symptom_list.pkl is just for the list of symptoms
