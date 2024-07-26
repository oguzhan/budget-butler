# Budget Butler

Budget Butler is your personal finance advisor, helping you understand and optimize your spending habits. This Streamlit-based application processes PDF bank statements, provides financial insights, and allows you to ask questions about your finances using natural language.

## Features

- PDF bank statement processing
- Financial overview dashboard
- Subscription detection
- Spending trend visualization
- AI-powered financial advice and Q&A

## Prerequisites

- Python 3.8+
- Poetry (for dependency management)
- Ollama (for local AI model inference)

## Setup

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/budget-butler.git
   cd budget-butler
   ```

2. Install dependencies using Poetry:
   ```
   poetry install
   ```

3. Set up Ollama:
   - Install Ollama from [https://ollama.ai/](https://ollama.ai/)
   - Pull the required model:
     ```
     ollama pull llama3
     ```

4. Set up environment variables:
   Create a `.env` file in the project root and add the following:
   ```
   OLLAMA_API_BASE_URL=http://localhost:11434/api
   MODEL=llama3
   ```

## Running the Application

1. Activate the Poetry environment:
   ```
   poetry shell
   ```

2. Run the Streamlit app:
   ```
   streamlit run budgetbutler.py
   ```

3. Open your web browser and navigate to `http://localhost:8501`

## Usage

1. Upload your PDF bank statements when prompted.
2. View your financial overview, including total spent, income, and monthly burn rate.
3. Check the list of potential subscriptions detected from your transactions.
4. Analyze your spending trend over time.
5. Ask questions about your finances using the chatbot interface.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
