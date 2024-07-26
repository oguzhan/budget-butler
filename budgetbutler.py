import streamlit as st
import streamlit_shadcn_ui as ui
from streamlit_extras.switch_page_button import switch_page
from modules.pdf import PDFHelper
from modules.config import Config
from modules.ollama import query_ollama
import pandas as pd
import plotly.express as px
import tempfile
import os
import time
import locale


def preprocess_amount(amount_str):
    # Remove any leading or trailing whitespace and the euro sign
    amount_str = amount_str.strip().replace("€", "")

    # Handle negative amounts
    multiplier = -1 if amount_str.startswith("-") else 1
    amount_str = amount_str.lstrip("-+")

    # Replace comma with dot for decimal point
    amount_str = amount_str.replace(",", ".")

    try:
        return multiplier * float(amount_str)
    except ValueError:
        print(f"Warning: Could not convert '{amount_str}' to float. Setting to 0.")
        return 0.0


def process_files(uploaded_files):
    pdf_helper = PDFHelper(model_name=Config.MODEL)
    all_transactions = []
    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            transactions = pdf_helper.extract_transactions(tmp_file.name)
            all_transactions.extend(transactions)
            os.unlink(tmp_file.name)
    return all_transactions


def prepare_dataframe(transactions):
    df = pd.DataFrame(transactions)
    df["date"] = pd.to_datetime(df["date"])
    df["amount"] = df["amount"].apply(preprocess_amount)
    print(df[["date", "amount", "description"]].head())
    return df


def calculate_financial_overview(df):
    total_spent = -df[df["amount"] < 0]["amount"].sum()
    total_income = df[df["amount"] > 0]["amount"].sum()
    num_months = len(df["date"].dt.to_period("M").unique())
    burn_rate = total_spent / num_months if num_months > 0 else 0
    savings_rate = (
        ((total_income - total_spent) / total_income * 100) if total_income > 0 else 0
    )
    return total_spent, total_income, burn_rate, savings_rate


def display_financial_overview(total_spent, total_income, burn_rate):
    st.header("Financial Overview", divider="rainbow")
    cols = st.columns(3)
    with cols[0]:
        ui.metric_card(
            title="Total Spent",
            content=f"€{total_spent:.2f}",
            description="Total expenses",
            key="card1",
        )
    with cols[1]:
        ui.metric_card(
            title="Total Income",
            content=f"€{total_income:.2f}",
            description="Total earnings",
            key="card2",
        )
    with cols[2]:
        ui.metric_card(
            title="Monthly Burn Rate",
            content=f"€{burn_rate:.2f}",
            description="Average monthly spending",
            key="card3",
        )


def find_subscriptions(df):
    prompt = f"""
    Analyze the following transaction data and identify potential subscriptions or recurring payments:
    {df.to_string()}
    
    List the likely subscriptions with their names and amounts. Format the response as a Python list of dictionaries.
    """
    response = query_ollama(prompt)
    try:
        subscriptions = eval(response)
        return pd.DataFrame(subscriptions)
    except:
        st.toast(
            "Error processing subscriptions. Showing all transactions instead.",
            icon="⚠️",
        )
        return df


def display_subscriptions(subscriptions):
    st.header("Potential Subscriptions", divider="rainbow")
    if not subscriptions.empty:
        st.dataframe(subscriptions, use_container_width=True)
    else:
        st.write("No subscriptions found.")


def display_spending_trend(df):
    st.header("Spending Over Time", divider="rainbow")
    daily_spending = (
        df[df["amount"] < 0].groupby("date")["amount"].sum().abs().reset_index()
    )
    if not daily_spending.empty:
        fig = px.line(
            daily_spending, x="date", y="amount", title="Daily Spending Trend"
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.write("No spending data available over time.")


def chat_message(message, is_user=False):
    if is_user:
        st.write(f"User: {message}")
    else:
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            for chunk in message.split():
                full_response += chunk + " "
                time.sleep(0.05)
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)


def handle_user_query(df, total_spent, total_income, burn_rate, savings_rate):
    st.header("Ask me anything about your finances", divider="rainbow")
    queries = [
        "What if I invested €500 in S&P500 on Jan 1, 2022?",
        "How much did I spend on groceries last month?",
        "What's my average daily spending?",
        "Which category do I spend the most on?",
        "How can I improve my savings rate?",
    ]
    selected_query = st.selectbox(
        "Select a question or type your own:", queries + ["Custom question"]
    )
    user_query = (
        st.text_input("Type your question here:")
        if selected_query == "Custom question"
        else selected_query
    )

    if st.button("Ask"):
        chat_message(user_query, is_user=True)
        context = f"Total spent: €{total_spent:.2f}\nTotal income: €{total_income:.2f}\nMonthly burn rate: €{burn_rate:.2f}\nCurrent savings rate: {savings_rate:.2f}%"
        response = query_ollama(
            f"Context: {context}\n\nUser question: {user_query}\n\nPlease provide a helpful and insightful answer based on the given financial data and the user's question."
        )
        chat_message(response)


def main():
    st.set_page_config(page_title="BudgetButler", page_icon="💼", layout="centered")
    st.sidebar.image("butler_logo.jpeg", use_column_width=True)

    if "processed" not in st.session_state:
        st.session_state.processed = False

    st.title("BudgetButler")

    if not st.session_state.processed:
        uploaded_files = st.file_uploader(
            "Upload transaction PDF files", accept_multiple_files=True, type="pdf"
        )

        if uploaded_files:
            st.toast(f"Processing {len(uploaded_files)} files...", icon="📊")
            transactions = process_files(uploaded_files)
            if transactions:
                df = prepare_dataframe(transactions)
                # Add this debug print
                st.write("Sample of processed transactions:")
                st.write(df[["date", "amount", "description"]].head())
                st.session_state.df = df
                st.session_state.processed = True
                st.rerun()
            else:
                st.toast(
                    "No transactions found in the PDF files. Please try again.",
                    icon="⚠️",
                )
    else:
        df = st.session_state.df
        total_spent, total_income, burn_rate, savings_rate = (
            calculate_financial_overview(df)
        )

        col1, col2 = st.columns(2)
        with col1:
            ui.metric_card(
                title="Total Spent",
                content=f"€{total_spent:.2f}",
                description="Total expenses",
                key="card1",
            )
            ui.metric_card(
                title="Monthly Burn Rate",
                content=f"€{burn_rate:.2f}",
                description="Average monthly spending",
                key="card3",
            )
        with col2:
            ui.metric_card(
                title="Total Income",
                content=f"€{total_income:.2f}",
                description="Total earnings",
                key="card2",
            )
            ui.metric_card(
                title="Savings Rate",
                content=f"{savings_rate:.2f}%",
                description="Current savings percentage",
                key="card4",
            )

        st.subheader("Spending Trend")
        display_spending_trend(df)

        st.subheader("Top Expenses")
        display_top_expenses(df)

        st.subheader("Income vs Expenses")
        display_income_vs_expenses(df)

        if st.button("Get AI Advice"):
            switch_page("chat")

        if st.button("Upload New Files"):
            st.session_state.processed = False
            st.rerun()


def display_top_expenses(df):
    top_expenses = (
        df[df["amount"] < 0]
        .groupby("description")["amount"]
        .sum()
        .abs()
        .nlargest(5)
        .reset_index()
    )
    fig = px.bar(top_expenses, x="description", y="amount", title="Top 5 Expenses")
    st.plotly_chart(fig, use_container_width=True)


def display_income_vs_expenses(df):
    monthly_summary = (
        df.groupby(df["date"].dt.to_period("M"))
        .agg({"amount": lambda x: (x[x > 0].sum(), -x[x < 0].sum())})
        .reset_index()
    )
    monthly_summary["date"] = monthly_summary["date"].dt.to_timestamp()
    monthly_summary[["income", "expenses"]] = pd.DataFrame(
        monthly_summary["amount"].tolist(), index=monthly_summary.index
    )
    fig = px.bar(
        monthly_summary,
        x="date",
        y=["income", "expenses"],
        title="Monthly Income vs Expenses",
    )
    st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
