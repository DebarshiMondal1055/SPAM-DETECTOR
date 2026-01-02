import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

STOPWORDS = set(stopwords.words('english'))


st.markdown("""
<style>
/* App background */
.stApp {
    background: linear-gradient(135deg, #f5f7fa, #c3cfe2);
}

/* Text INPUT styling */
.stTextInput input {
    padding: 14px !important;
    font-size: 16px !important;
    border-radius: 12px !important;
    background-color: #ffffff !important;
    color: #333333 !important;
}

/* Text AREA styling */
.stTextArea textarea {
    padding-left: 15px !important;
    padding-top: 8px !important;
    font-size: 16px !important;
    border-radius: 12px !important;
    background-color: #ffffff !important;
    color: #333333 !important;
}

/* Predict button */
div.stButton > button {
    background-color: #4CAF50;
    color: white;
    padding: 10px 26px;
    border-radius: 10px;
    font-size: 16px;
    border: none;
}

div.stButton > button:hover {
    background-color: #45a049;
    color: white;
}
</style>
""", unsafe_allow_html=True)



ps=PorterStemmer()

tfidf=pickle.load(open('vectorizer.pkl','rb'))
model=pickle.load(open('model.pkl','rb'))


st.title("Spam Detection Classifier")

input_sms=st.text_area("Enter the message",height=200)

if st.button('Predict'):
    #Preprocess
    def transform_text(text):
        text=text.lower()
        text=nltk.word_tokenize(text)
        
        y=[]
        for i in text:
            if i.isalnum():
                y.append(i)
        
        text=y[:]
        y.clear()
        
        for i in text:
            if i not in STOPWORDS and i not in string.punctuation:
                y.append(i)
        
        text=y[:]
        y.clear()
        
        for i in text:
            y.append(ps.stem(i))
        
        return " ".join(y)
    
    transformed_sms=transform_text(input_sms)
    #vectorize
    vector_input=tfidf.transform([transformed_sms])
    #predict
    result=model.predict(vector_input)[0]
    #display
    if result==1:
        st.header("Spam")
    else:
        st.header("Not Spam")

