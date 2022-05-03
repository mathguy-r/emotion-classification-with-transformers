import streamlit as st
from inference import predict
import time
import json
import os


if __name__ == "__main__":
    st.title("Emotion Detection with transformers")
    input_text = st.text_input('Input text here')
    model_dir = r'C:\Workplace\finetuned-models\emotion-classification\distilbert-finetuned\saved'
    if st.button('Proceed'):
        output_label = predict(input_text,model_dir)
        st.write(f'Your input expresses {output_label}')
    
