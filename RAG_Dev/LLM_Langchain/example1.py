## Integrate our code with OPENAI API

import os
from constants import openai_key
from langchain.llms import OpenAI
import streamlit as st
"""Used for Prompting"""
from langchain import PromptTemplate
"""Used for Running with Prompt Template"""
from langchain.chains import LLMChain

"""SimpleSequentialChain is used for running prompt sequence 
but problem with SimpleSequentialChain is that it will output only last prompt result.
So, instead of SimpleSequentialChain, SequentialChain is used"""
from langchain.chains import SimpleSequentialChain
from langchain.chains import SequentialChain

"""Stroring Conversation"""
from langchain.memory import ConversationBufferMemory

os.environ["OPENAI_API_KEY"] = openai_key

# Streamlit framework

st.title("Celebrity Search Results")
input_text = st.text_input("Search the topic you want")

# Memory

person_memory = ConversationBufferMemory(input_key='name', memory_key='chat_history')
dob_memory = ConversationBufferMemory(input_key='person', memory_key='chat_history')
descr_memory = ConversationBufferMemory(input_key='dob', memory_key='descr_history')

# Prompt Template

first_input_prompt = PromptTemplate(
    input_variables=['name'],
    template="Tell me about Celebrity {name}"
)

llm = OpenAI(
    temperature=0.8
)

chain1 = LLMChain(
    llm = llm,
    prompt=first_input_prompt,
    verbose=True,
    output_key='person',
    memory=person_memory
    )

# Prompt Template

second_input_prompt = PromptTemplate(
    input_variables=['person'],
    template="When was {person} born"
)
chain2 = LLMChain(
    llm = llm,
    prompt=second_input_prompt,
    verbose=True,
    output_key='dob',
    memory=dob_memory
    )

# Prompt Template

third_input_prompt = PromptTemplate(
    input_variables=['dob'],
    template="Mention 5 major events happened around {dob} in the world"
)
chain3 = LLMChain(
    llm = llm,
    prompt=third_input_prompt,
    verbose=True,
    output_key='description',
    memory=descr_memory
    )

# parent_chain = SimpleSequentialChain(chains=[chain1, chain2], verbose=True)

parent_chain = SequentialChain(chains=[chain1,chain2, chain3], verbose=True, output_variables=['person','dob', 'description'])

if input_text:
    "Used when SimpleSequentialChain"
    #st.write(parent_chain.run(input_text))
    "Used when SequentialChain"
    st.write(parent_chain({'name':input_text}))

    with st.expander('Person Name'):
        st.info(person_memory.buffer)
    with st.expander('Major Events'):
        st.info(descr_memory.buffer)


