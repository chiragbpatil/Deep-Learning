import streamlit as st
import pandas as pd
import analyze_sentence

st.title("Enter text to analyze sentiment")
title = st.text_area('',
                      'I really like the new design of your website!',height=20)

if st.button('Analyze',type='primary'):
    if title:
        prediction = analyze_sentence.predict_sentiment(title)
        st.write("#")
        st.write(prediction)
    else:
        st.warning('Input is Empty', icon="⚠️")
