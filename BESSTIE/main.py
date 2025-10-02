import streamlit as st 
import pandas as pd
from huggingface_hub import login
from datasets import load_dataset
from sklearn.model_selection import train_test_split
import os

# HuggingFace
HF_TOKEN = "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"

login(token=HF_TOKEN)


ds = load_dataset("unswnlporg/BESSTIE")

train_df = ds["train"].to_pandas()
valid_df = ds["validation"].to_pandas()

train_df, test_df = train_test_split(
    train_df,
    test_size=1500,   
    random_state=42,
    shuffle=True
)


#######################

if "page" not in st.session_state:
    st.session_state.page = "Landing Page"

st.sidebar.title("Navigation")
st.session_state.page = st.sidebar.radio("Go to:", ["Landing Page", "BESSTIE benchmark","Mistral Model"], 
                                         index=0 if st.session_state.page == "Landing Page" else 1)

if st.session_state.page == "Landing Page":
    st.title('Sentiment and Sarcasm Classification')
    option = st.selectbox("Select an English variant",("Australian","British","Indian"))

    variety_map = {
        "Australian": "en-AU",
        "British": "en-UK",
        "Indian": "en-IN"
    }

    selected_variety = variety_map.get(option, "en-AU")

    df_reddit = train_df[
        (train_df['source'] == 'Reddit') &
        (train_df['variety'] == selected_variety)
    ]

    st.text("Reddit filtered data")
    st.write(df_reddit)


    df_google = train_df[
        (train_df['source'] == 'Google') &
        (train_df['variety'] == selected_variety)
    ]

    st.text("Google Places filtered data")
    st.write(df_google)

    st.info(
        """
        📘 **Note:**  
        The above tables show *manually annotated examples* of sarcasm and sentiments.  
        These annotations serve as the **ground truth**.  
        They will be compared against predictions made on a non-annotated dataset,  
        where **9 different LLM models** will attempt to detect sarcasm and sentiment  
        across three English varieties: **Australian, British, and Indian**.
        """
    )

    if st.button("BESSTIE"):
        st.session_state.page = "BESSTIE benchmark"
        st.rerun()


elif st.session_state.page == "BESSTIE benchmark":
    st.title("BESSTIE benchmark")
    st.info(
        """
        Here we compare the **predictions from 9 LLM models**  
        against the manually annotated ground truth.
        """
    )
    st.text("This page will show model outputs and evaluation metrics.")
    
    if st.button("Mistral Model"):
        st.session_state.page = "Mistral Model"
        st.rerun()

elif st.session_state.page == "Mistral Model":
    st.text("This page will show MISTRAL model")


