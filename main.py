import re
from typing import Dict, List
import unicodedata
from dataclasses import dataclass
import streamlit as st
from joblib import load
from sfx import Tokenizer

@dataclass
class Config:
    tf_idf: str = "./checkpoint/tf_idf.jbl"
    pretrained_model_path: str = "./checkpoint/model.jbl"
    t_path: str = "./checkpoint/sep_sfx.jbl"
    
config = Config()

LABEL_ARCHIVE_MAP: Dict[int, str] = {
    0: 'Chinh tri Xa hoi',
    1: 'Doi song',
    2: 'Khoa hoc',
    3: 'Kinh doanh',
    4: 'Phap luat',
    5: 'Suc khoe',
    6: 'The gioi',
    7: 'The thao',
    8: 'Van hoa',
    9: 'Vi tinh'
}

LABEL_ARCHIVE: List[str] = ['Chinh tri Xa hoi', 'Doi song', 'Khoa hoc', 'Kinh doanh',
                            'Phap luat', 'Suc khoe', 'The gioi', 'The thao', 'Van hoa','Vi tinh']

def clean_text(text: str):
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"[^a-zA-Z0-9àÀảẢãÃáÁạẠăĂằẰẳẲẵẴắẮặẶâÂầẦẩẨẫẪấẤậẬđĐèÈẻẺẽẼéÉẹẸêÊềỀểỂễỄếẾệỆìÌỉỈĩĨíÍịỊòÒỏỎõÕóÓọỌôÔồỒổỔỗỖốỐộỘơƠờỜởỞỡỠớỚợỢùÙủỦũŨúÚụỤưƯừỪửỬữỮứỨựỰỳỲỷỶỹỸýÝỵỴ0-9]", " ", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip().lower()

tokenizer = Tokenizer(ckpt_path=config.t_path)
vectorizer = load(config.tf_idf)
model = load(config.pretrained_model_path)

def segment_text(text: str):
    text = clean_text(text)
    
    return [' '.join(i) for j in tokenizer.segment([text]) for i in j]

def classify_text(text: str):
    text = segment_text(text)
    feature = vectorizer.transform(text)
    
    outputs = model.predict_proba(feature)[0]
    
    proba = outputs.round(3)
    return {'label': LABEL_ARCHIVE_MAP[proba.argmax()], "class probability": {k:v for k, v in zip(LABEL_ARCHIVE, proba)}}
    

def main():
    st.title("Vietnamese New Classification")
    custom_css = '''
    <style>
        div.css-1om1ktf.e1y61itm0 {
          width: 800px;
        }
        textarea.st-cl {
          height: 400px;
        }
    </style>
    '''
    st.markdown(custom_css, unsafe_allow_html=True)
    user_input = st.text_area(
        label="abc",
        placeholder="Type your text here:",
        height=600,
        label_visibility='hidden'
    )
    # text_box = st.text_area(
    #     label="abc", height=800, placeholder="Input text here", label_visibility='hidden')
    st.button(label="Classify")
    
    if user_input != "":
        outputs = classify_text(user_input)
        st.json(outputs)
        # st.write(outputs)

if __name__=='__main__':
    main()