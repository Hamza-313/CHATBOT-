import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# --- Load your dataset ---
data = open(r"C:\Users\RC\OneDrive\Documents\Chat.txt", "r", encoding="utf-8").read().split("\n")

pairs = [line.split("\t") for line in data if "\t" in line]
inputs, outputs = zip(*pairs)

# Add start/end tokens to target sentences
outputs = ["<START> " + text + " <END>" for text in outputs]

# --- Tokenize ---
tokenizer = Tokenizer(filters='')
tokenizer.fit_on_texts(list(inputs) + list(outputs))
vocab_size = len(tokenizer.word_index) + 1

input_seq = tokenizer.texts_to_sequences(inputs)
output_seq = tokenizer.texts_to_sequences(outputs)

max_len_input = max(len(s) for s in input_seq)
max_len_output = max(len(s) for s in output_seq)

input_seq = pad_sequences(input_seq, maxlen=max_len_input, padding='post')
output_seq = pad_sequences(output_seq, maxlen=max_len_output, padding='post')

# --- Model setup ---
latent_dim = 128

# Encoder
encoder_inputs = Input(shape=(max_len_input,))
enc_emb = Embedding(vocab_size, 64)(encoder_inputs)
encoder_lstm = LSTM(latent_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder_lstm(enc_emb)
encoder_states = [state_h, state_c]

# Decoder
decoder_inputs = Input(shape=(max_len_output,))
dec_emb_layer = Embedding(vocab_size, 64)
dec_emb = dec_emb_layer(decoder_inputs)
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(dec_emb, initial_state=encoder_states)
decoder_dense = Dense(vocab_size, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

# Prepare decoder target data (shifted output)
decoder_target_data = np.zeros_like(output_seq)
decoder_target_data[:, 0:-1] = output_seq[:, 1:]

# --- Train ---
model.fit([input_seq, output_seq], np.expand_dims(decoder_target_data, -1),
          batch_size=8, epochs=500, verbose=1)

# --- Inference setup ---
encoder_model = Model(encoder_inputs, encoder_states)

decoder_state_input_h = Input(shape=(latent_dim,))
decoder_state_input_c = Input(shape=(latent_dim,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

dec_emb2 = dec_emb_layer(decoder_inputs)
decoder_outputs2, state_h2, state_c2 = decoder_lstm(
    dec_emb2, initial_state=decoder_states_inputs)
decoder_outputs2 = decoder_dense(decoder_outputs2)
decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs2, state_h2, state_c2])

index_word = {i: w for w, i in tokenizer.word_index.items()}

# --- Chat function ---
def decode_sequence(input_text):
    seq = tokenizer.texts_to_sequences([input_text])
    seq = pad_sequences(seq, maxlen=max_len_input, padding='post')
    states_value = encoder_model.predict(seq)

    target_seq = np.array([[tokenizer.word_index["<start>"]]])
    stop_condition = False
    decoded_sentence = ''

    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_word = index_word.get(sampled_token_index, '')

        if sampled_word == '<end>' or len(decoded_sentence.split()) > max_len_output:
            stop_condition = True
        else:
            decoded_sentence += ' ' + sampled_word

        target_seq = np.array([[sampled_token_index]])
        states_value = [h, c]

    return decoded_sentence.strip()

def chat():
    print("🤖 Chatbot ready! Type 'quit' to stop.")
    while True:
        msg = input("You: ")
        if msg.lower() == 'quit':
            break
        response = decode_sequence(msg)
        print("Bot:", response)



import pickle
import os

# === SAVE MODEL AND TOKENIZER ===
save_path = r"D:\Deep Learning"

print("🔄 Saving trained models and tokenizer...")

# Create folder if not exists
os.makedirs(save_path, exist_ok=True)

# Save models
model.save(os.path.join(save_path, "chatbot_model.h5"))
encoder_model.save(os.path.join(save_path, "encoder_model.h5"))
decoder_model.save(os.path.join(save_path, "decoder_model.h5"))

# Save tokenizer and parameters
data_info = {
    "tokenizer": tokenizer,
    "index_word": index_word,
    "max_len_input": max_len_input,
    "max_len_output": max_len_output,
    "vocab_size": vocab_size,
    "latent_dim": latent_dim
}

pkl_path = os.path.join(save_path, "chatbot_data.pkl")
with open(pkl_path, "wb") as f:
    pickle.dump(data_info, f)

print(f"✅ All model files saved successfully at:\n{save_path}")
chat() 
