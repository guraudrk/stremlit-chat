import streamlit as st
import pickle
from tensorflow.keras.models import load_model
import requests
from nltk.tokenize import word_tokenize
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import tempfile
import re

# nltk 다운로드 (필요시 실행)
import nltk
nltk.download('punkt')

# CSS 파일 로드 함수
def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# CSS 파일 적용
load_css('style.css')

# S3 URL로 모델 다운로드
@st.cache_resource
def download_model():
    model_url = "https://aikingsejong.s3.ap-northeast-2.amazonaws.com/chatbot_model.h5"
    response = requests.get(model_url)
    response.raise_for_status()
    with tempfile.NamedTemporaryFile(delete=False, suffix='.h5') as tmp_file:
        tmp_file.write(response.content)
        tmp_file.flush()
        return tmp_file.name

# 데이터 로드
@st.cache_resource
def load_data():
    try:
        with open('organized_data.pickle', 'rb') as f:
            organized_data = pickle.load(f)

        with open('tokenizer.pickle', 'rb') as f:
            tokenizer = pickle.load(f)

        with open('word_map.pickle', 'rb') as f:
            word_map = pickle.load(f)

        with open('vectorizer.pickle', 'rb') as f:  # Vectorizer 로드
            vectorizer = pickle.load(f)

        model_file = download_model()
        lstm_model = load_model(model_file)
        return organized_data, tokenizer, word_map, vectorizer, lstm_model

    except Exception as e:
        st.error(f"데이터를 로드하는 데 실패했습니다: {e}")
        return None, None, None, None, None

organized_data, tokenizer, word_map, vectorizer, lstm_model = load_data()

if not all([organized_data, tokenizer, word_map, vectorizer, lstm_model]):
    st.stop()

# 상단에 세종대왕 이미지 추가
st.markdown("""
    <div style='text-align: center;'>
        <img src='https://raw.githubusercontent.com/guraudrk/stremlit-chat/main/sejong.jpg' width='200'>
    </div>
""", unsafe_allow_html=True)

st.title('AI 세종대왕과 대화하기')

# 상단 우측에 예시 질문 버튼 추가
if st.button('예시 질문 보기'):
    st.session_state.show_examples = not st.session_state.get('show_examples', False)

# 사용자 질문 입력
user_question = st.text_input(
    '질문을 입력하세요:', 
    '원활한 질문을 위해서는 우측 상단의 버튼으로 예시 질문을 확인해주세요'
)

# 대화 내역을 저장할 리스트
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

def preprocess_text(text):
    tokens = word_tokenize(text)
    return ' '.join(tokens)

def find_similar_answer(question, data, vectorizer, lstm_model, tokenizer, threshold=0.3):
    question = preprocess_text(question)
    question_vec = vectorizer.transform([question])
    question_seq = tokenizer.texts_to_sequences([question])
    question_seq = pad_sequences(question_seq, maxlen=lstm_model.input_shape[1])

    max_sim = -1
    most_similar_answer = "적절한 답변을 찾지 못했습니다. 다른 질문을 해보세요."

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
            if max_sim >= threshold:
                most_similar_answer = qa['A']

    return most_similar_answer

def replace_words(text, word_map):
    for key, value in word_map.items():
        text = text.replace(key, value)
    return text

def add_dane_suffix(text):
    sentences = re.split(r'([.?!])', text)
    new_sentences = []

    # 종결 어미들을 하나의 정규식 패턴으로 정의
    ending_pattern = re.compile(r'(다|까|니|라|냐|는가|나요)$')

    for i in range(0, len(sentences) - 1, 2):
        sentence = sentences[i].strip()
        punctuation = sentences[i + 1]
        if sentence:
            # 정규식으로 종결 어미 제거
            sentence = ending_pattern.sub('', sentence)
            new_sentences.append(sentence + '다네' + punctuation)

    if len(sentences) % 2 != 0:
        sentence = sentences[-1].strip()
        if sentence:
            sentence = ending_pattern.sub('', sentence)
            new_sentences.append(sentence + '다네')

    return ' '.join(new_sentences)

# 대화 기록을 말풍선 스타일로 출력
def display_chat_history(chat_history):
    for i, (user, bot) in enumerate(chat_history):
        st.markdown(f"""
        <div class='chat-bubble user'>
            <strong>사용자:</strong> {user}
        </div>
        <div class='chat-bubble bot'>
            <strong>세종대왕:</strong> {bot}
        </div>
        """, unsafe_allow_html=True)

# 예시 질문 목록 팝업을 출력하는 함수
def show_example_popup():
    st.markdown("""
        <div id="example-popup" class="popup">
            <div class="popup-content">
                <span class="close-btn" onclick="document.getElementById('example-popup').style.display='none'">&times;</span>
                <h2>예시 질문</h2>
                <ul>
                    {% for qa in organized_data %}
                        <li>{{ qa['Q'] }}</li>
                    {% endfor %}
                </ul>
            </div>
        </div>
        <script>
            if (document.getElementById('example-popup')) {
                document.getElementById('example-popup').style.display = 'flex';
            }
        </script>
    """, unsafe_allow_html=True)

# 타이핑 애니메이션 함수
def type_text():
    st.markdown("""
        <style>
        #intro-text {
            opacity: 1;
            transition: opacity 1s ease;
        }
        </style>
        <p id="intro-text">원활한 질문을 위해서는 우측 상단의 버튼으로 예시 질문을 확인해주세요</p>
        <script>
            let text = "원활한 질문을 위해서는 우측 상단의 버튼으로 예시 질문을 확인해주세요";
            let i = 0;
            let speed = 50;
            let introTextElement = document.getElementById('intro-text');
            function typeWriter() {
                if (i < text.length) {
                    introTextElement.innerHTML += text.charAt(i);
                    i++;
                    setTimeout(typeWriter, speed);
                } else {
                    setTimeout(() => { introTextElement.style.opacity = 0; }, 3000);
                }
            }
            typeWriter();
        </script>
    """, unsafe_allow_html=True)

if user_question and user_question != '원활한 질문을 위해서는 우측 상단의 버튼으로 예시 질문을 확인해주세요':
    answer = find_similar_answer(user_question, organized_data, vectorizer, lstm_model, tokenizer)

    # "적절한 답변을 찾지 못했습니다" 메시지에 '다네' 접미사를 붙이지 않음
    if answer == "적절한 답변을 찾지 못했습니다. 다른 질문을 해보세요.":
        st.session_state.chat_history.append((user_question, answer))
    else:
        transformed_answer = replace_words(answer, word_map)
        final_answer = add_dane_suffix(transformed_answer)
        st.session_state.chat_history.append((user_question, final_answer))

# 대화 내역 출력
display_chat_history(st.session_state.chat_history)

# 예시 질문 목록이 표시될 때 팝업 창을 표시
if st.session_state.get('show_examples', False):
    show_example_popup()
else:
    type_text()