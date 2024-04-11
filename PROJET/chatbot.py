# =================================================================
#                               LIBS
# =================================================================
import re
import os
import json
import random
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from keras_preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, load_model
from keras.layers import Input, Embedding, LSTM, LayerNormalization, Dense, Dropout
import streamlit as st

# =================================================================
#                           CONFIG PAGE
# =================================================================

st.set_page_config(page_title="MENTAL HEALTH CHATBOT")

# =================================================================
#                            FUNCTIONS
# =================================================================

def load_dataset():
    """
    LOAD DATASET FROM JSON FILE
    """
    with open('me.json', 'r') as f:
        data = json.load(f)

    dic = {"tag":[], "patterns":[], "responses":[]}
    for example in data['intents']:
        for pattern in example['patterns']:
            dic['patterns'].append(pattern)
            dic['tag'].append(example['tag'])
            dic['responses'].append(example['responses'])

    df = pd.DataFrame.from_dict(dic)    

    return df

def tokenize_dataset(dataset):
    tokenizer = Tokenizer(lower=True, split=' ')
    tokenizer.fit_on_texts(dataset['patterns'])

    ptrn2seq = tokenizer.texts_to_sequences(dataset['patterns'])
    X = pad_sequences(ptrn2seq, padding='post')

    lbl_enc = LabelEncoder()
    y = lbl_enc.fit_transform(dataset['tag'])

    return X,y,tokenizer,lbl_enc

def make_model(vocab_size,X,y):
    model = Sequential()
    model.add(Input(shape=(X.shape[1],)))
    model.add(Embedding(input_dim=vocab_size+1, output_dim=100, mask_zero=True))
    model.add(LSTM(32, return_sequences=True))
    model.add(LayerNormalization())
    model.add(LSTM(32, return_sequences=True))
    model.add(LayerNormalization())
    model.add(LSTM(32))
    model.add(LayerNormalization())
    model.add(Dense(128, activation="relu"))
    model.add(LayerNormalization())
    model.add(Dropout(0.2))
    model.add(Dense(128, activation="relu"))
    model.add(LayerNormalization())
    model.add(Dropout(0.2))
    model.add(Dense(len(np.unique(y)), activation="softmax"))
    model.compile(optimizer='adam', loss="sparse_categorical_crossentropy", metrics=['accuracy'])
    model.summary()
    return model

def train_model(model,X,y,epochs):
    model_history = model.fit(
        x=X,
        y=y,
        batch_size=10,
        epochs=epochs
    )
    return model_history

def model_responce(query,tokenizer,model,lbl_enc,dataframe): 
    text = []
    txt = re.sub('[^a-zA-Z\']', ' ', query)
    txt = txt.lower()
    txt = txt.split()
    txt = " ".join(txt)
    text.append(txt)  
    x_test = tokenizer.texts_to_sequences(text)
    x_test = np.array(x_test).squeeze()
    x_test = pad_sequences([x_test], padding='post', maxlen=X.shape[1])
    y_pred = model.predict(x_test)
    y_pred = y_pred.argmax()
    tag = lbl_enc.inverse_transform([y_pred])[0]
    responses = dataframe[dataframe['tag'] == tag]['responses'].values[0]
    return random.choice(responses)

def reload_model():
    model = load_model('test.keras')

    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)

    with open('label_encoder.pickle', 'rb') as handle:
        lbl_enc = pickle.load(handle)

    return model, tokenizer, lbl_enc

def save_model(model,tokenizer, lbl_enc):
    model.save('test.keras')

    with open('tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open('label_encoder.pickle', 'wb') as handle:
        pickle.dump(lbl_enc, handle, protocol=pickle.HIGHEST_PROTOCOL)

# =================================================================
#                               SIDEBAR
# =================================================================

if st.sidebar.button("LOAD DATA", key="load", use_container_width=True):
    st.session_state['dataset'] = load_dataset() 

st.sidebar.divider()

epochs = st.sidebar.number_input("EPOCHS",1,500, value = 10)
vocab_size = st.sidebar.number_input("VOCAB SIZE",100,10000, value = 1000)

if st.sidebar.button("TRAIN MODEL", key="train", use_container_width=True):
    X,y,tokenizer,labl_enc = tokenize_dataset(st.session_state['dataset'])
    model = make_model(vocab_size,X,y)
    history = train_model(model,X,y,epochs)

    st.session_state['model'] = model
    st.session_state['tokenizer'] = tokenizer
    st.session_state['labl_enc'] = labl_enc

    st.sidebar.title('Model Accuracy')
    st.sidebar.write('Accuracy over epochs')

    # Afficher le graphique
    fig, ax = plt.subplots()
    ax.plot(history.history['accuracy'])
    ax.set(xlabel='Epoch', ylabel='Accuracy',
        title='Model accuracy over epochs')
    ax.grid()
    st.sidebar.pyplot(fig)

st.sidebar.divider()

if st.sidebar.button("SAVE MODEL", key="save", use_container_width=True):
    save_model(st.session_state['model'],st.session_state['tokenizer'], st.session_state['labl_enc'])

# =================================================================
#                               PAGE
# =================================================================

st.title("💬 MENTAL HEALTH CHATBOT")
st.caption("🚀 Make with Numpy, Pandas, Scikit-Learn, Tensorflow")

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

for msg in st.session_state.messages:
    if msg["role"] == "assistant" : 
        avatar = "🤖"
    else : 
        avatar="🧑‍💻"
    st.chat_message(msg["role"], avatar = avatar).write(msg["content"])

if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user", avatar="🧑‍💻").write(prompt)

    dataset = load_dataset()
    X,y,tokenizer,labl_enc = tokenize_dataset(dataset)
    model, tokenizer, lbl_enc = reload_model()
    response = model_responce(prompt,tokenizer,model,labl_enc,dataset)
    
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.chat_message("assistant", avatar="🤖").write(response)