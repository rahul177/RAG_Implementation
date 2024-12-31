## Integrate our code with OPENAI API

import os
from constants import openai_key
from langchain.llms import OpenAI
import streamlit as st

os.environ["OPENAI_API_KEY"] = openai_key

# Streamlit framework

st.title("Langchain with OpenAI Demo")
input_text = st.text_input("Search the topic you want")

llm = OpenAI(
    temperature=0.8
)
if input_text:
    st.write(llm(input_text))


