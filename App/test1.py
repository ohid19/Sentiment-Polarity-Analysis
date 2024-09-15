import tkinter as tk
from tkinter import ttk
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import numpy as np
import re
import nltk
from nltk.corpus import stopwords

# Download stopwords and punkt (tokenizer)
nltk.download('stopwords')
nltk.download('punkt')

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load model and tokenizer
model_name = "ohid19/dissertation_bert"  # repository name
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name)
model = model.to(device)
model.eval()

# Define label mapping
label_mapping = {0: 'negative', 1: 'neutral', 2: 'positive'}

def preprocess_sentence(sentence):
    # Truncate sentence to 300 characters
    sentence = sentence[:300]

    # Replace HTML line breaks with space
    sentence = re.sub(r"<br\s*/?>", " ", sentence)

    # Replace non-alphabetic characters (except apostrophes) with space
    sentence = re.sub(r"[^a-zA-Z']", " ", sentence)

    # Convert to lowercase
    sentence = sentence.lower()

    # Tokenize (split by spaces)
    tokens = sentence.split()

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]

    # Join tokens back into a single string
    cleaned_sentence = ' '.join(tokens)

    return cleaned_sentence

def predict_single_string(text, tokenizer, model, device):
    # Tokenize the input text
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=128,
        padding='max_length',
        return_attention_mask=True,
        return_tensors='pt'
    )

    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    # Make prediction
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        logits = logits.detach().cpu().numpy()

    # Convert logits to predicted label
    predicted_label = np.argmax(logits, axis=1)

    return label_mapping[predicted_label[0]]

def on_predict():
    user_input = entry.get()
    cleaned_input = preprocess_sentence(user_input)
    result = predict_single_string(cleaned_input, tokenizer, model, device)
    result_textbox.delete(1.0, tk.END)
    result_textbox.insert(tk.END, f"Predicted Sentiment: {result}")

# Tkinter UI setup
root = tk.Tk()
root.title("Sentiment Prediction App")
root.geometry("500x300")
root.resizable(False, False)

# Style
style = ttk.Style()
style.theme_use("clam")
style.configure('TLabel', font=('Helvetica', 11), padding=10)
style.configure('TEntry', font=('Helvetica', 11))
style.configure('TButton', font=('Helvetica', 11), padding=5)
style.configure('TFrame', background='#f0f0f0')
style.configure('TText', font=('Helvetica', 11))

# Main Frame
main_frame = ttk.Frame(root)
main_frame.pack(pady=20, padx=20)

# Title Label
title_label = ttk.Label(main_frame, text="Sentiment Prediction", font=("Helvetica", 16, 'bold'))
title_label.grid(row=0, column=0, columnspan=2, pady=10)

# Input Label and Entry
input_label = ttk.Label(main_frame, text="Enter Text:")
input_label.grid(row=1, column=0, sticky=tk.W, pady=10)
entry = ttk.Entry(main_frame, width=50)
entry.grid(row=1, column=1, padx=10, pady=10)

# Predict Button
predict_button = ttk.Button(main_frame, text="Predict Sentiment", command=on_predict)
predict_button.grid(row=2, column=0, columnspan=2, pady=10)

# Result Textbox
result_textbox = tk.Text(main_frame, height=5, width=50, font=("Helvetica", 11), wrap=tk.WORD, relief=tk.GROOVE, borderwidth=2)
result_textbox.grid(row=3, column=0, columnspan=2, pady=10)

# Add padding between widgets
for widget in main_frame.winfo_children():
    widget.grid_configure(padx=5, pady=5)

# Start Tkinter main loop
root.mainloop()
