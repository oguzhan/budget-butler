# pages/chat.py

import streamlit as st
import streamlit_shadcn_ui as ui
from modules.ollama import query_ollama
import time

st.set_page_config(page_title="BudgetButler Chat", page_icon="💼", layout="centered")

# Add logo to sidebar
st.sidebar.image("butler_logo.jpeg", use_column_width=True)

st.title("BudgetButler AI Advisor")


def chat_message(message, is_user=False):
    with st.chat_message("user" if is_user else "assistant"):
        message_placeholder = st.empty()
        full_response = ""
        for chunk in message.split():
            full_response += chunk + " "
            time.sleep(0.05)
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)


pre_prepared_queries = [
    "How can I improve my savings rate?",
    "What are my top spending categories?",
    "How does my spending compare to last month?",
    "What's my projected savings for the next 3 months?",
    "Suggest a budget plan based on my spending habits.",
]

selected_query = st.radio(
    "Select a question or type your own:", pre_prepared_queries + ["Custom question"]
)

if selected_query == "Custom question":
    user_query = st.text_input("Type your question here:")
else:
    user_query = selected_query

if st.button("Ask"):
    if "df" in st.session_state:
        df = st.session_state.df
        context = f"Total spent: €{-df[df['amount'] < 0]['amount'].sum():.2f}\n"
        context += f"Total income: €{df[df['amount'] > 0]['amount'].sum():.2f}\n"
        context += (
            f"Date range: {df['date'].min().date()} to {df['date'].max().date()}\n"
        )
        top_expenses = (
            df[df["amount"] < 0].groupby("description")["amount"].sum().nsmallest(5)
        )
        context += f"Top 5 expense categories: {', '.join(f'{cat} (€{-amt:.2f})' for cat, amt in top_expenses.items())}"

        chat_message(user_query, is_user=True)
        response = query_ollama(
            f"Context: {context}\n\nUser question: {user_query}\n\nProvide a helpful and insightful answer based on the given financial data and the user's question."
        )
        chat_message(response)
    else:
        st.error("Please upload and process your financial data first.")

if st.button("Back to Dashboard"):
    st.switch_page("budgetbutler.py")
