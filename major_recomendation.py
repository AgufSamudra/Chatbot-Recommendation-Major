import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import pickle
from sentence_transformers import SentenceTransformer
from langchain_google_genai import ChatGoogleGenerativeAI

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import requests
from bs4 import BeautifulSoup
import json
from keras.models import load_model

import sys
print(sys.executable)


@st.cache_resource(show_spinner="Downloading Model...")
def load_models():
    sentence_embed_model = SentenceTransformer("firqaaa/indo-sentence-bert-base")
    print("MODEL TRANSFORMER: indo-sentence-bert-base ------------> READY")
    return sentence_embed_model


def predict_major(input_text: str, model_path="model_rnn_bert.h5", embed_model_name=None, top_n=2):

    with open('BERT_Embedding_Dataset.pkl', 'rb') as f:
        _, _, label_encoder = pickle.load(f)
    
    model = load_model(model_path)
    
    # Embed the input text
    embedded_text = embed_model_name.encode([input_text])
    
    # Reshape embedded_text to match the model's input shape
    embedded_text = np.reshape(embedded_text, (1, 1, -1))
    
    # Predict the probabilities
    pred = model.predict(embedded_text)[0]
    
    # Get the indices of the top_n predictions
    top_indices = pred.argsort()[-top_n:][::-1]
    
    # Decode the top_n labels
    labels = label_encoder.inverse_transform(top_indices)
    
    return labels

def get_content_major(url):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    # Send a GET request to fetch the raw HTML content
    response = requests.get(url, headers=headers)
    response.raise_for_status()  # Check if the request was successful

    # Parse the content of the request with BeautifulSoup
    soup = BeautifulSoup(response.text, 'html.parser')

    # Find all titles with class "title text-left"
    titles = soup.find_all('div', class_='title text-left')

    # Find all contents with class "content"
    contents = soup.find_all('div', class_='content')

    combined_text = ""
    # Combine the text of all titles and contents
    for title, content in zip(titles, contents):
        title_text = title.get_text(strip=True)
        content_text = content.get_text(strip=True)
        combined_text += f"{title_text}\n\n{content_text}\n\n"
        
    return combined_text


def continue_convertation_major(input_prediction, user_input):
    df = pd.read_csv("jurusan_url.csv")
    
    # filter dataset url
    filtered_df_url = df[df['nama_jurusan'].isin(input_prediction)]
    
    # create summary column and fill it with the respective content
    summaries = []
    for index, row in filtered_df_url.iterrows():
        url = row['url']
        summary = get_content_major(url)
        summaries.append(summary)
    
    filtered_df_url['summary'] = summaries
    
    # encode summary with tfidf
    filtered_df_url["summary"] = filtered_df_url["summary"].str.lower()
    
    # vectorizer = TfidfVectorizer()
    # tfidf_matrix = vectorizer.fit_transform(filtered_df_url["summary"])
    
    # user_tfidf = vectorizer.transform([user_input])
    # cosine_similarities = cosine_similarity(user_tfidf, tfidf_matrix).flatten()
    
    # most_similar_index = cosine_similarities.argmax()
    
    # filtered = filtered_df_url.reset_index(drop=True).loc[most_similar_index, "nama_jurusan"] # nama jurusan hasil cosine
    # filter = filtered_df_url[filtered_df_url["nama_jurusan"] == filtered]
    
    
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key="AIzaSyAjVjTEYhTrCwIL_-Q0rBCV7HAOrqTuLd8")
    llm = llm.invoke(f"""
                     [CONTEXT]
                     {filtered_df_url["summary"].values}
                     [/CONTEXT]
                     
                     ==============================
                     
                     [QUESTION]
                     {user_input}
                     [/QUESTION]
                     
                     ==============================
                     
                     Please jawab pertanyaan user terkait apa yang di tanya. refrensi adalah pada Context. jika tidak ada di context jangan jawab 'saya tidak menemukan informasi demikian .. .. ..'
                     Cukup jawab, kalau kamu hanya menjawab berdasarkan informasi context saja. Jika tidak bisa silahkan klik tombol new chat untuk rekomendasi jurusan lainnya.
                     
                     Dan pastikan jawaban mu hanya maksimal di 1500 Character saja.
                     
                     """)
    
    st.session_state.messages.append({"role": "assistant", "content": llm.content})
    
    return llm.content

def major_recomendation():
    sentence_embed_model = load_models()
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
        
    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
    intro = "Hallo, kamu ingin tau Jurusan yang cocok untuk kamu? Saya akan bantu! Pertama tolong beritau saya **Minat dan Bakat** mu. Silahkan ceritakan semuanya kepada saya! 😊"
    if not any(msg["content"] == intro for msg in st.session_state.messages):
        st.session_state.messages.append({"role": "assistant", "content": intro})
        st.chat_message("assistant").markdown(intro)
        
    prompt = st.chat_input("Ketik disni!")
    if prompt:
        
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").markdown(prompt)
        
        if len(st.session_state.messages) == 2: 
            final_message = f"Baiklah saya akan merekomendasikan jurusan untukmu! 😊"
            st.session_state.messages.append({"role": "assistant", "content": final_message})
            st.chat_message("assistant").markdown(final_message)
            
            with st.spinner("Mohon Tunggu..."):
                user_contents = [message['content'] for message in st.session_state.messages if message['role'] == 'user']

                input_join = ' '.join(user_contents)
                global predict
                predict = predict_major(input_join, embed_model_name=sentence_embed_model)
                llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key="AIzaSyAjVjTEYhTrCwIL_-Q0rBCV7HAOrqTuLd8")
                
                llm = llm.invoke(f"""
                                 
                [SYSTEM]
                Anda adalah seorang yang dapat merekomendasikan kepada user terkait jurusan yang ingin di pilih.
                [/SYSTEM]
                
                ===========================================
                                 
                [CONVERTATION]
                {st.session_state.messages}
                [/CONVERTATION]
                                 
                ===========================================
                                 
                [PREDIKSI]
                {predict}
                [/PREDIKSI]

                ===========================================
                
                [SUMMARY]
                Tolong beri tahu user bahwa rekomendasi dari apa yang dia ceritakan adalah sesuai prediksi di atas! Dan berikan penjelasan bagaimana dia bisa memanfaatkan ilmu pengetahuan kedua jurusan tersebut untuk mencapai tujuannya.
                Dan terakhir, beri kalimat untuk memilih 2 jurusan tersebut sesuai prioritas utamanya untuk mencapai tujuan.
                [/SUMMARY]
                
                [END] Mohon tampilkan jawabanmu saja tanpa menulis ulang apa yang diketik di atas [/END]

                """)
                
                st.chat_message("assistant").markdown(llm.content)
                st.session_state.messages.append({"role": "assistant", "content": llm.content})
                st.chat_message("assistant").markdown("Jika kamu ingin mengetahui secara detail terkait kedua jurusan tersebut, kamu bisa lanjut untuk bertanya kepada saya. Atau tekan tombol `New Chat` untuk memulai rekomendasi kembali")
            
        elif len(st.session_state.messages) >= 4:
            # Mengambil 3 data terbaru dari session state messages
            last_three_messages = st.session_state.messages[-3:]

            # Mengkonversi data tersebut ke dalam format string JSON
            json_string = json.dumps(last_three_messages)
            
            st.chat_message("assistant").markdown(continue_convertation_major(predict, json_string))
                
            
