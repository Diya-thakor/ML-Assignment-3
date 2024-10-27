import streamlit as st
import torch
import torch.nn as nn
import os
from collections import Counter

# Load the vocabulary from your checkpoint
def load_vocab(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    return checkpoint['vocab']

# Define the model structure
class WordPredictorModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim=64, hidden_dim=1024, context_size=3, activation='relu'):
        super(WordPredictorModel, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.fc1 = nn.Linear(context_size * embedding_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, vocab_size)
        
        # Choose activation function
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()

    def forward(self, x):
        x = self.embeddings(x)  
        x = x.view(x.size(0), -1)  
        x = self.activation(self.fc1(x))  
        x = self.fc2(x)  
        return x

# Function to predict the next word
def predict_next_word(model, context):
    model.eval()
    with torch.no_grad():
        context_tensor = torch.tensor(context, dtype=torch.long).unsqueeze(0)  # Add batch dimension
        output = model(context_tensor)
        predicted_idx = output.argmax(dim=1).item()  # Get the index of the predicted word
    return predicted_idx

# Streamlit App
st.title("Word Predictor App")

# User input for model selection
embedding_size = st.selectbox("Select Embedding Size:", [64, 128])
context_length = st.selectbox("Select Context Length:", [5, 10, 15])
activation_function = st.selectbox("Select Activation Function:", ["ReLU", "Tanh"])

# Load vocabulary
vocab = load_vocab('checkpoints/best_model_1.pth')  # Adjust the path as necessary
vocab_size = len(vocab)

# Load the corresponding model
model = WordPredictorModel(vocab_size, embedding_size, hidden_dim=1024, context_size=context_length, activation=activation_function)
model.load_state_dict(torch.load(f'checkpoints/best_model_1.pth')['model_state_dict'])  # Adjust the path as necessary

# User input for context
user_input = st.text_input("Enter context (words separated by spaces):")
if st.button("Predict Next Word"):
    if user_input:
        # Convert user input to indices
        context = [vocab[word] for word in user_input.split() if word in vocab]
        if len(context) < context_length:
            st.error(f"Please enter at least {context_length} words.")
        else:
            context = context[-context_length:]  # Take the last 'context_length' words
            predicted_idx = predict_next_word(model, context)
            predicted_word = [word for word, idx in vocab.items() if idx == predicted_idx]
            st.success(f"Predicted Next Word: {predicted_word[0] if predicted_word else 'Not Found'}")
    else:
        st.error("Please enter some context.")