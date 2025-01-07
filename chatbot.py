import pickle
from chatterbot import ChatBot
from chatterbot.trainers import ChatterBotCorpusTrainer

chatbot = ChatBot('Disease Predictor')
trainer = ChatterBotCorpusTrainer(chatbot)
trainer.train('chatterbot.corpus.english')

def predict_disease(symptoms):
    with open('models/disease_model.pkl', 'rb') as f:
        model, label_encoder = pickle.load(f)

    input_data = [0] * len(model.feature_importances_)
    for symptom in symptoms:
        if symptom in input_data:
            input_data[input_data.index(symptom)] = 1

    prediction = model.predict([input_data])
    disease = label_encoder.inverse_transform(prediction)
    return disease[0]

while True:
    user_input = input("Enter your symptoms (comma-separated): ")
    if user_input.lower() == 'exit':
        break

    symptoms = user_input.split(',')
    result = predict_disease(symptoms)
    print(f"Predicted Disease: {result}")
