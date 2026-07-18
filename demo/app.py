from __future__ import annotations

import streamlit as st

from jevalops.inference.rule_based_backend import RuleBasedBackend

st.set_page_config(page_title="JEvalOps", layout="wide")
st.title("JEvalOps")

backend = RuleBasedBackend()
prompt = st.text_area(
    "Prompt",
    "指示:\n取引先への丁寧な依頼文に書き換えてください。\n\n入力:\n資料見たら返事ください\n\n回答:",
    height=220,
)
col1, col2 = st.columns([1, 1])
with col1:
    max_tokens = st.slider("Max tokens", 16, 512, 256)
with col2:
    temperature = st.slider("Temperature", 0.0, 1.0, 0.0)

if st.button("Generate"):
    result = backend.generate(prompt, max_tokens=max_tokens, temperature=temperature)
    st.subheader("Output")
    st.write(result.text)
    st.subheader("Efficiency")
    st.json(result.to_dict())
