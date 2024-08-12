import streamlit as st
import pickle
from tensorflow.keras.models import load_model
import requests
from konlpy.tag import Kkma
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import tempfile
import re

# Kkma 인스턴스 전역에서 생성
kkma = Kkma()

@st.cache_resource
def download_model():
    model_url = "https://aikingsejong.s3.ap-northeast-2.amazonaws.com/chatbot_model.h5"
    response = requests.get(model_url)
    response.raise_for_status()  # HTTP 오류가 발생하면 예외를 발생시킵니다.
    with tempfile.NamedTemporaryFile(delete=False, suffix='.h5') as tmp_file:
        tmp_file.write(response.content)
        tmp_file.flush()
        return tmp_file.name

@st.cache_resource
def load_data():
    with open('organized_data.pickle', 'rb') as f:
        organized_data = pickle.load(f)

    with open('tokenizer.pickle', 'rb') as f:
        tokenizer = pickle.load(f)

    with open('word_map.pickle', 'rb') as f:
        word_map = pickle.load(f)

    model_file = download_model()
    try:
        lstm_model = load_model(model_file)
    except Exception as e:
        st.error(f"모델을 로드하는 데 실패했습니다: {e}")
        return None, None, None, None
    return organized_data, tokenizer, word_map, lstm_model

organized_data, tokenizer, word_map, lstm_model = load_data()

if not all([organized_data, tokenizer, word_map, lstm_model]):
    st.stop()

st.title('AI 세종대왕과 대화하기')

user_question = st.text_input('질문을 입력하세요: ')

def preprocess_text(text):
    tokens = kkma.morphs(text)
    return ' '.join(tokens)

def find_similar_answer(question, data, vectorizer, lstm_model, tokenizer, threshold=1.0):
    question = preprocess_text(question)
    question_vec = vectorizer.transform([question])
    question_seq = tokenizer.texts_to_sequences([question])
    question_seq = pad_sequences(question_seq, maxlen=lstm_model.input_shape[1])

    max_sim = -1
    most_similar_answer = "그 질문은 이해할 수 없다네. 가엽고 딱한 자로다!"

    for qa in data:
        q = qa['Q']
        q_vec = vectorizer.transform([preprocess_text(q)])
        sim = cosine_similarity(question_vec, q_vec)[0][0]

        q_seq = tokenizer.texts_to_sequences([preprocess_text(q)])
        q_seq = pad_sequences(q_seq, maxlen=question_seq.shape[1])

        lstm_sim = lstm_model.predict([question_seq, q_seq])
        avg_sim = (sim + lstm_sim.mean()) / 2

        if avg_sim > max_sim:
            max_sim = avg_sim
            most_similar_answer = qa['A'] if max_sim >= threshold else most_similar_answer

    return most_similar_answer

def replace_words(text, word_map):
    for key, value in word_map.items():
        text = text.replace(key, value)
    return text

def add_dane_suffix(text):
    sentences = re.split(r'([.?!])', text)
    new_sentences = []

    for i in range(0, len(sentences) - 1, 2):
        sentence = sentences[i].strip()
        punctuation = sentences[i + 1]
        if sentence:
            if sentence.endswith(('다', '까', '니', '라', '냐', '는가', '나요')):
                sentence = sentence[:-1]
            new_sentences.append(sentence + '다네' + punctuation)

    if len(sentences) % 2 != 0:
        sentence = sentences[-1].strip()
        if sentence:
            if sentence.endswith(('다', '까', '니', '라', '냐', '는가', '나요')):
                sentence = sentence[:-1]
            new_sentences.append(sentence + '다네')

    return ' '.join(new_sentences)

if user_question:
    answer = find_similar_answer(user_question, organized_data, tokenizer, lstm_model, tokenizer)

    if answer:
        transformed_answer = replace_words(answer, word_map)
        final_answer = add_dane_suffix(transformed_answer)
        st.write("답변:", final_answer)
    else:
        st.write("유사한 답변을 찾을 수 없습니다.")