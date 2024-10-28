import streamlit as st
import torch
import json
import torch.nn as nn

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Step 1: Load the vocabulary
def load_vocab(vocab_path):
    with open(vocab_path, 'r') as f:
        vocab = json.load(f)
    return vocab

# Load vocabulary
vocab = load_vocab('vocab.json')  # Adjust this path as necessary
word2idx = vocab
idx2word = {idx: word for word, idx in word2idx.items()}
vocab_size = len(vocab)

# Step 2: Define the model
class NextWordPredictor(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, block_size, activation_fn="ReLU"):
        super(NextWordPredictor, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.fc1 = nn.Linear(embedding_dim * block_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, vocab_size)
        self.block_size = block_size

        # Set activation function based on parameter
        if activation_fn == "ReLU":
            self.activation = nn.ReLU()
        elif activation_fn == "Tanh":
            self.activation = nn.Tanh()
        else:
            raise ValueError("Unsupported activation function selected. Choose 'ReLU' or 'Tanh'.")

    def forward(self, x):
        x = self.embedding(x)  # Shape: (batch_size, block_size, embedding_dim)

        # Flatten only if x's dimensions are compatible with fc1
        if x.size(1) == self.block_size:
            x = x.view(x.size(0), -1)  # Flatten to (batch_size, block_size * embedding_dim)
        else:
            raise RuntimeError("Input size mismatch: ensure block_size matches during prediction.")

        x = self.activation(self.fc1(x))  # Apply activation function after the first linear layer
        out = self.fc2(x)  # Output logits for each vocabulary token
        return out

# Step 3: Streamlit UI for User Inputs
st.title("Next Word Predictor")
st.write("Select the parameters for the model:")

# User-selectable parameters for the model
embedding_dim = st.selectbox("Select Embedding Size", [64, 128])
block_size = st.selectbox("Select Context Length (Block Size)", [3, 6, 10]) 
activation_fn = st.selectbox("Select Activation Function", ["ReLU", "Tanh"])
hidden_dim = 1024  # This can be fixed, but feel free to add it as an option

# Seed text and number of words to predict
seed_text = st.text_input("Seed Text", "once upon a time")
num_words = st.number_input("Number of Words to Predict", min_value=1, max_value=10, value=1)

# Model selection logic
model_paths = {
    (3, 64, 'ReLU'): 'best_model_0.pth',
    (6, 64, 'ReLU'): 'best_model_1.pth',
    (10, 64, 'ReLU'): 'best_model_2.pth',
    (3, 128, 'ReLU'): 'best_model_3.pth',
    (6, 128, 'ReLU'): 'best_model_4.pth',
    (10, 128, 'ReLU'): 'best_model_5.pth',
    (3, 64, 'Tanh'): 'best_model_6.pth',
    (6, 64, 'Tanh'): 'best_model_7.pth',
    (10, 64, 'Tanh'): 'best_model_8.pth',
    (3, 128, 'Tanh'): 'best_model_9.pth',
    (6, 128, 'Tanh'): 'best_model_10.pth',
    (10, 128, 'Tanh'): 'best_model_11.pth'
}

selected_model_path = model_paths[(block_size, embedding_dim, activation_fn)]

# Step 4: Initialize and load the model with selected parameters
model = NextWordPredictor(vocab_size, embedding_dim, hidden_dim, block_size, activation_fn)
model.load_state_dict(torch.load(selected_model_path, map_location=device))  # Adjust path if necessary
model.to(device)
model.eval()  # Set the model to evaluation mode

# Step 5: Prediction function
def predict_next_word(model, seed_text, num_words=1, block_size=5):
    model.eval()
    words = seed_text.lower().split()
    
    for _ in range(num_words):
        input_seq = [word2idx.get(word, 0) for word in words[-block_size:]]
        input_seq = torch.tensor(input_seq, dtype=torch.long).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(input_seq)
            predicted_idx = output.argmax(dim=1).item()
        
        next_word = idx2word.get(predicted_idx, "<UNK>")
        words.append(next_word)
    
    return ' '.join(words)

# Prediction button
if st.button("Predict"):
    generated_text = predict_next_word(model, seed_text, num_words, block_size)
    st.write("Generated Text: ", generated_text)
