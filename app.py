import os
import numpy as np
import pickle
import streamlit as st
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


DATA_PATH = r"C:\Users\RC\OneDrive\Documents\Chat.txt"
SAVE_DIR = r"D:\Deep Learning"
os.makedirs(SAVE_DIR, exist_ok=True)

MODEL_PATH = os.path.join(SAVE_DIR, "chatbot_model.h5")
ENCODER_PATH = os.path.join(SAVE_DIR, "encoder_model.h5")
DECODER_PATH = os.path.join(SAVE_DIR, "decoder_model.h5")
PKL_PATH = os.path.join(SAVE_DIR, "chatbot_data.pkl")

LATENT_DIM = 128
EMBED_DIM = 64
EPOCHS = 100  # can increase to 500 after testing



def load_dataset():
    data = open(DATA_PATH, "r", encoding="utf-8").read().split("\n")
    pairs = [line.split("\t") for line in data if "\t" in line]
    inputs, outputs = zip(*pairs)

    outputs = ["<start> " + text + " <end>" for text in outputs]

    tokenizer = Tokenizer(filters='')
    tokenizer.fit_on_texts(list(inputs) + list(outputs))
    vocab_size = len(tokenizer.word_index) + 1

    input_seq = tokenizer.texts_to_sequences(inputs)
    output_seq = tokenizer.texts_to_sequences(outputs)

    max_len_input = max(len(s) for s in input_seq)
    max_len_output = max(len(s) for s in output_seq)

    input_seq = pad_sequences(input_seq, maxlen=max_len_input, padding='post')
    output_seq = pad_sequences(output_seq, maxlen=max_len_output, padding='post')

    decoder_target_data = np.zeros_like(output_seq)
    decoder_target_data[:, 0:-1] = output_seq[:, 1:]

    return (input_seq, output_seq, decoder_target_data,
            tokenizer, vocab_size, max_len_input, max_len_output)



def build_model(vocab_size, max_len_input, max_len_output):
    encoder_inputs = Input(shape=(max_len_input,))
    enc_emb = Embedding(vocab_size, EMBED_DIM)(encoder_inputs)
    encoder_lstm = LSTM(LATENT_DIM, return_state=True)
    _, state_h, state_c = encoder_lstm(enc_emb)
    encoder_states = [state_h, state_c]

    decoder_inputs = Input(shape=(max_len_output,))
    dec_emb_layer = Embedding(vocab_size, EMBED_DIM)
    dec_emb = dec_emb_layer(decoder_inputs)
    decoder_lstm = LSTM(LATENT_DIM, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(dec_emb, initial_state=encoder_states)
    decoder_dense = Dense(vocab_size, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)

    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
    return model, encoder_inputs, encoder_states, dec_emb_layer, decoder_lstm, decoder_dense

def train_and_save():
    input_seq, output_seq, decoder_target_data, tokenizer, vocab_size, max_len_input, max_len_output = load_dataset()
    model, encoder_inputs, encoder_states, dec_emb_layer, decoder_lstm, decoder_dense = build_model(
        vocab_size, max_len_input, max_len_output)

    st.info("🧠 Training the chatbot model... (this may take time)")
    model.fit([input_seq, output_seq], np.expand_dims(decoder_target_data, -1),
              batch_size=8, epochs=EPOCHS, verbose=1)

    # Save models
    model.save(MODEL_PATH)

    encoder_model = Model(encoder_inputs, encoder_states)
    encoder_model.save(ENCODER_PATH)

    decoder_state_input_h = Input(shape=(LATENT_DIM,))
    decoder_state_input_c = Input(shape=(LATENT_DIM,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

    decoder_inputs = Input(shape=(None,))
    dec_emb2 = dec_emb_layer(decoder_inputs)
    decoder_outputs2, state_h2, state_c2 = decoder_lstm(dec_emb2, initial_state=decoder_states_inputs)
    decoder_outputs2 = decoder_dense(decoder_outputs2)
    decoder_model = Model([decoder_inputs] + decoder_states_inputs,
                          [decoder_outputs2, state_h2, state_c2])
    decoder_model.save(DECODER_PATH)

    index_word = {i: w for w, i in tokenizer.word_index.items()}

    data_info = {
        "tokenizer": tokenizer,
        "index_word": index_word,
        "max_len_input": max_len_input,
        "max_len_output": max_len_output,
        "vocab_size": vocab_size,
        "latent_dim": LATENT_DIM
    }
    with open(PKL_PATH, "wb") as f:
        pickle.dump(data_info, f)

    st.success("✅ Model and tokenizer saved successfully!")



def load_models():
    encoder_model = load_model(ENCODER_PATH)
    decoder_model = load_model(DECODER_PATH)
    with open(PKL_PATH, "rb") as f:
        data_info = pickle.load(f)
    return encoder_model, decoder_model, data_info



def decode_sequence(input_text, encoder_model, decoder_model, data_info):
    tokenizer = data_info["tokenizer"]
    index_word = data_info["index_word"]
    max_len_input = data_info["max_len_input"]
    max_len_output = data_info["max_len_output"]

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

        if not sampled_word or sampled_word == '<end>' or len(decoded_sentence.split()) > max_len_output:
            stop_condition = True
        else:
            decoded_sentence += ' ' + sampled_word

        target_seq = np.array([[sampled_token_index]])
        states_value = [h, c]

    return decoded_sentence.strip()

st.title("💬 Seq2Seq Chatbot (LSTM)")
st.markdown("### Train or Chat with your custom model")

if st.button("🚀 Train Chatbot"):
    train_and_save()

if os.path.exists(ENCODER_PATH) and os.path.exists(DECODER_PATH) and os.path.exists(PKL_PATH):
    encoder_model, decoder_model, data_info = load_models()
    st.success("✅ Model loaded! You can now chat.")
    user_input = st.text_input("You:", "")
    if user_input:
        reply = decode_sequence(user_input, encoder_model, decoder_model, data_info)
        st.write(f"**Bot:** {reply}")
else:
    st.warning("⚠️ No saved model found. Please train it first.")

