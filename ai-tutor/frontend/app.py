import streamlit as st
import requests

API_URL = "http://localhost:8000/api/ask"

st.set_page_config(page_title="AI Tutor", page_icon="🎓")

st.title("🎓 AI Tutor")
