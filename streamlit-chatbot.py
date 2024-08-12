import streamlit as st
import pickle
from tensorflow.keras.models import load_model
import requests
from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize
from konlpy.tag import Okt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import matplotlib.pyplot as plt
import re
import pickle
import streamlit as st


# 모델 및 데이터 로드-캐시를 사용해서 중복 로드를 막는다.
@st.cache_resource
def load_data():
    with open('organized_data.pickle', 'rb') as f:
        organized_data = pickle.load(f)

    with open('tokenizer.pickle', 'rb') as f:
        tokenizer = pickle.load(f)

    with open('word_map.pickle', 'rb') as f:
        word_map = pickle.load(f)

    lstm_model = load_model('chatbot_model.keras')
    return organized_data, tokenizer, word_map, lstm_model

organized_data, tokenizer, word_map, lstm_model = load_data()



# Streamlit 앱 코드
st.title('AI 세종대왕과 대화하기')

# 사용자 입력 받기
user_question = st.text_input('질문을 입력하세요: ')

# 데이터 전처리 및 형태소 분석을 위한 함수 정의. komlpy를 사용한다.
def preprocess_text(text):
    kkma = Kkma() # Kkma를 초기화
    tokens = kkma.morphs(text)  # 형태소 분석 및 토큰화
    return ' '.join(tokens) #파이썬에서 문자열을 결합하기 위한 코드이다. 리스트의 문자열을 하나의 문자열로 결합하되, 각 요소 사이에 공백을 삽입하는 것이다.



#코사인 유사도를 이용해서 질문과 유사한 데이터를 q배열에서 찾는 함수이다.
def find_similar_answer(question, data, vectorizer, lstm_model, tokenizer,threshold=1.0): #여기서 threshold는 한계점이라는 뜻이며, 정확도가 일정 수준 이하이면 질문을 이해할 수 없다는 답변을 출력한다.
    okt = Okt() #konlpy의 okt 사용
    question = preprocess_text(question) #질문을  okt를 사용해서 전처리한다. 
    question_vec = vectorizer.transform([question]) #질문을 TF-IDF 벡터화한다. 이를 통해 유사도를 계산할 준비를 한다. 
    question_seq = tokenizer.texts_to_sequences([question])# 질문을 시퀀스로 변환한다.
    question_seq = pad_sequences(question_seq, maxlen=model.input_shape[1])# 패딩 처리.

    max_sim = -1 
    most_similar_answer = None #가장 유사한 답변을 담을 변수를 초기화한다. 

    for i, qa in enumerate(data): #qa 배열을 data에 받는다. data는 q도 있고, a도 있기 때문에, enumerate을 통해 받는다. 
        q = qa['Q'] #qa배열에서 현재의 질문을 추출한다. 
        q_vec = vectorizer.transform([preprocess_text(q)])  # 질문을 전처리한 뒤, TF-IDF 벡터화한다.
        sim = cosine_similarity(question_vec, q_vec)[0][0] #입력 질문과 데이터 질문 사이의 코사인 유사도를 계산한다. 

        q_seq = tokenizer.texts_to_sequences([preprocess_text(q)])  # 질문 전처리한뒤, 시퀀스를 패딩한다.(데이터에 특정 값을 채워서 데이터의 크기를 조정하는 것)
        q_seq = pad_sequences(q_seq, maxlen=question_seq.shape[1]) #시퀀스를 패딩한다. shape[1]은 시퀀스의 길이를 나타낸다.

        lstm_sim = model.predict([question_seq, q_seq])  # LSTM을 통해 입력 질문 시퀀스와 데이터 질문 시퀀스 간의 유사도를 예측한다.
        avg_sim = (sim + lstm_sim.mean()) / 2  # lstm_sim의 평균 값을 사용
        print(avg_sim)#입력 질문 시퀀스와 데이터 질문 시퀀스 간의 유사도를 예측한 값을 프린트한다.

        # numpy 배열로 비교
        if np.any(avg_sim > max_sim): #np.any는 조건을 만족하는지 확인한다.
            max_sim = avg_sim #지금 유사도가 기존의 max_sim보다 크면 값을 새로 업데이트 한다.
            most_similar_answer = qa['A'] if max_sim >= threshold else "그 질문은 이해할 수 없다네. 가엽고 딱한 자로다!" #이에 따라 가장 유사한 답변을 업데이트 한다.

    return most_similar_answer

# 어휘 변환 함수
def replace_words(text, word_map):
    for key, value in word_map.items():
        text = text.replace(key, value)
    return text

def add_dane_suffix(text):
    # 문장 끝의 종결어미를 찾아 '다네'로 변경
    sentences = re.split(r'([.?!])', text)  # 문장을 구분하는 정규 표현식
    new_sentences = []
    
    for i in range(0, len(sentences) - 1, 2):
        sentence = sentences[i].strip()
        punctuation = sentences[i + 1]
        if sentence:  # 문장이 비어있지 않다면
            if sentence.endswith(('다', '까', '니', '라', '냐', '는가', '나요')):
                sentence = sentence[:-1]  # 마지막 글자를 제거
            new_sentences.append(sentence + '다네' + punctuation)
    
    # 마지막 문장이 구두점으로 끝나지 않은 경우
    if len(sentences) % 2 != 0:
        sentence = sentences[-1].strip()
        if sentence:
            if sentence.endswith(('다', '까', '니', '라', '냐', '는가', '나요')):
                sentence = sentence[:-1]  # 마지막 글자를 제거
            new_sentences.append(sentence + '다네')
    
    return ' '.join(new_sentences)

# 유사한 답변 찾기
if user_question:
    answer = find_similar_answer(user_question, organized_data, tokenizer, lstm_model, tokenizer)

    # answer가 None이 아닌지 확인
    if answer is not None:
        # 어휘 변환
        transformed_answer = replace_words(answer, word_map)
        final_answer = add_dane_suffix(transformed_answer)
        st.write("답변:", final_answer)
    else:
        st.write("유사한 답변을 찾을 수 없습니다.")