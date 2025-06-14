

This document provides comprehensive documentation for the Hugging Face Agent Course final project. It includes an overview of the project, its architecture, the agent's workflow, and instructions for setting up and running the project.

## Table of Contents
1.  [Project Overview](#project-overview)
2.  [System Architecture](#system-architecture)
3.  [Agent Workflow](#agent-workflow)
4.  [Setup and Installation](#setup-and-installation)
5.  [Usage](#usage)
6.  [Evaluation](#evaluation)
7.  [Tools Used](#tools-used)
8.  [References](#references)




## 1. Project Overview

This project is the final assignment for the Hugging Face Agent Course, designed to evaluate the participant's ability to build and deploy an AI agent capable of answering complex, real-world questions. The agent's performance is benchmarked against a subset of the GAIA (General AI Assistant) benchmark, with a target score of 30% or higher for course completion.

The GAIA benchmark, introduced in the paper "GAIA: A Benchmark for General AI Assistants," features 466 challenging questions that require multi-step reasoning, multimodal understanding, web browsing, and proficient tool use. These questions are conceptually simple for humans but prove difficult for current AI systems, highlighting the limitations of standalone Large Language Models (LLMs) and emphasizing the need for agent-based systems.

The project involves interacting with a provided API to fetch questions and submit answers for scoring. The API exposes several routes, including `/questions` to retrieve the full list of evaluation questions, `/random-question` for a single question, `/files/{task_id}` to download associated files, and `/submit` to submit agent answers and update the leaderboard. The submission process requires a Hugging Face username, a link to the agent's code, and a list of answers for each `task_id`.

The provided codebase serves as a basic template, which participants are encouraged to modify and enhance to develop a more robust and effective agent. The evaluation is based on an exact match comparison of the submitted answers to the ground truth.




## 2. System Architecture

The system architecture of this Hugging Face Agent project is designed to facilitate the interaction between a user interface, the core agent logic, external APIs, and various tools and data stores. The main components and their interactions are illustrated in the Components below:

**Components:**

*   **User Interface (Gradio):** This is the primary interface for users to interact with the agent. It allows users to initiate the evaluation process, view the status of the agent, and see the results of the questions and answers.

*   **BasicAgent (Python application):** This component acts as the main application orchestrator. It handles user authentication (Hugging Face login), fetches questions from the external API, instantiates and runs the LangGraph Agent, and submits the agent's answers back to the external API for scoring.

*   **LangGraph Agent (core logic):** This is the brain of the system, responsible for processing questions and generating answers. It leverages the LangGraph framework to define a state machine that orchestrates the use of various tools and the Supabase Vector Store to answer complex queries.

*   **External API (questions and submission):** This API, provided by the Hugging Face Agent Course, serves as the source of evaluation questions and the endpoint for submitting answers. It is crucial for the benchmarking process.

*   **Tools:** The LangGraph Agent utilizes a suite of tools to gather information and perform calculations. These include:
    *   **Wikipedia Search:** For retrieving information from Wikipedia.
    *   **Web Search (Tavily):** For general web searches to find relevant information.
    *   **Arxiv Search:** For searching academic papers on Arxiv.
    *   **Calculator Tools:** For performing arithmetic operations (multiply, add, subtract, divide, modulus).

*   **Supabase Vector Store (question retrieval):** This component acts as a knowledge base, storing embeddings of previously encountered questions or relevant documents. The agent can query this vector store to find similar questions or retrieve contextual information, aiding in answering new questions.

**Interactions:**

1.  The **User Interface** initiates the process by triggering the **BasicAgent** to run the evaluation.
2.  The **BasicAgent** communicates with the **External API** to fetch questions.
3.  The **BasicAgent** passes the questions to the **LangGraph Agent** for processing.
4.  The **LangGraph Agent** dynamically decides whether to use its **Tools** (Wikipedia, Web Search, Arxiv, Calculator) or query the **Supabase Vector Store** based on the nature of the question.
5.  The **Tools** and **Supabase Vector Store** provide information back to the **LangGraph Agent**.
6.  Once the **LangGraph Agent** formulates an answer, it returns it to the **BasicAgent**.
7.  Finally, the **BasicAgent** submits the generated answers to the **External API** for scoring.




## 3. Agent Workflow

The agent's workflow describes the sequence of operations performed by the LangGraph Agent to process a question and arrive at an answer. The workflow is designed to be dynamic, allowing the agent to choose the most appropriate tools or knowledge sources based on the input question. The general workflow is depicted below:


![Agent Workflow Diagram](https://github.com/user-attachments/assets/886e1584-85a5-4e91-9815-7ce911accf75)


**Workflow Steps:**

1.  **Fetch Questions from External API:** The `app.py` script, through the `run_and_submit_all` function, initiates the process by making a GET request to the external API to retrieve a list of questions for evaluation.

2.  **Agent Processes Question (LangGraph Agent):** Each fetched question is then passed to the `BasicAgent` instance, which in turn invokes the `build_graph` function (from `agent.py`). This function initializes the LangGraph agent, which is the core decision-making and processing unit.

3.  **Agent Uses Tools or Supabase Vector Store:** This is a crucial decision point in the workflow. The LangGraph Agent, based on the nature of the question and its internal reasoning, determines whether to:
    *   **Utilize Tools:** If the question requires external information retrieval or calculations, the agent will select and use one or more of the available tools. These include `wiki_search` (Wikipedia), `web_search` (Tavily), `arvix_search` (Arxiv), and the arithmetic tools (`multiply`, `add`, `subtract`, `divide`, `modulus`).
    *   **Query Supabase Vector Store:** If the question is similar to previously encountered questions or if relevant contextual information is stored, the agent can query the Supabase Vector Store to retrieve similar documents or answers. This acts as a form of memory or knowledge base for the agent.

4.  **Agent Generates Answer:** After gathering necessary information from tools or the vector store, the LangGraph Agent processes this information and formulates a coherent answer to the original question. The LLM (Google Gemini, Groq, or Hugging Face endpoint) plays a central role in synthesizing the information and generating the final response.

5.  **Submit Answer to External API:** Once an answer is generated for a given `task_id`, the `BasicAgent` collects this answer. After processing all questions, it compiles all the answers into a payload and sends a POST request to the external API's `/submit` endpoint. This submission includes the user's Hugging Face username, a link to the agent's code, and the list of submitted answers.

This iterative process ensures that the agent can handle a variety of question types by intelligently leveraging its available resources.




## 4. Setup and Installation

To set up and run this project, follow these steps:

### Prerequisites

*   Python 3.9+
*   `pip` (Python package installer)
*   Hugging Face account (for submission to the leaderboard)
*   Supabase project with a `documents` table and `match_documents_langchain` function (for the vector store)
*   API keys for Google Gemini, Groq, or Hugging Face Inference API (depending on your chosen LLM provider)
*   Tavily API key (for web search)

### Installation Steps

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd <your-repository-name>
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up environment variables:**
    Create a `.env` file in the root directory of your project and add the following environment variables. Replace the placeholder values with your actual API keys and credentials.

    ```dotenv
    # Hugging Face
    HF_TOKEN="your_huggingface_token"

    # Supabase
    SUPABASE_URL="your_supabase_url"
    SUPABASE_SERVICE_KEY="your_supabase_service_key"

    # LLM Provider (choose one)
    # For Google Gemini
    GOOGLE_API_KEY="your_google_api_key"
    # For Groq
    GROQ_API_KEY="your_groq_api_key"
    # For Hugging Face Inference API
    HF_INFERENCE_API_KEY="your_huggingface_inference_api_key"

    # Tavily
    TAVILY_API_KEY="your_tavily_api_key"
    ```

    **Note:** Ensure that `system_prompt.txt` is present in the same directory as `agent.py`, as it contains the system prompt for the agent.




## 5. Usage

To run the application and evaluate your agent, execute the `app.py` file:

```bash
python app.py
```

This will launch a Gradio interface in your web browser. Follow the instructions on the Gradio interface:

1.  **Log in to Hugging Face:** Use the provided button to log in with your Hugging Face account. Your username will be used for submission to the leaderboard.
2.  **Run Evaluation & Submit All Answers:** Click this button to initiate the evaluation process. The application will:
    *   Fetch questions from the external API.
    *   Run your `BasicAgent` (which utilizes the `LangGraph Agent` and its tools) on each question.
    *   Collect the answers.
    *   Submit your answers to the Hugging Face scoring API.

Results, including the submission status and a table of questions with your agent's submitted answers, will be displayed in the Gradio interface.





## 6. Evaluation

The primary goal of this project is to achieve a score of 30% or higher on a subset of the GAIA benchmark. The evaluation process is handled by an external API provided by the Hugging Face Agent Course. Here's how it works:

*   **Question Fetching:** The `app.py` script fetches a set of 20 questions, extracted from Level 1 of the GAIA validation set. These questions are chosen for their manageable complexity, requiring fewer than 5 steps and minimal tool usage.

*   **Answer Submission:** Your agent's generated answers are submitted to the `/submit` endpoint of the external API. The submission payload includes your Hugging Face username, a link to your agent's code (for verification), and a list of `task_id` and `submitted_answer` pairs.

*   **Scoring:** The API evaluates your submitted answers against ground truth answers using an **exact match** comparison. This means your agent's output must precisely match the expected answer to be considered correct. Therefore, careful prompting and precise answer formatting by your agent are crucial.

*   **Leaderboard:** Upon successful submission, your score is updated on a public leaderboard hosted on Hugging Face. This allows you to track your progress and compare your agent's performance with other participants.

**Important Considerations for Evaluation:**

*   **Exact Match:** Pay close attention to the format and content of the expected answers. Even minor discrepancies (e.g., extra spaces, incorrect capitalization, or additional text) can lead to an incorrect score.
*   **Agent Code Link:** Ensure your Hugging Face Space containing your agent's code is public so that the submission can be verified.
*   **Error Handling:** The `app.py` includes robust error handling for API requests and agent execution. Any errors during question fetching, agent execution, or submission will be reported in the Gradio interface.





## 7. Tools Used

This project leverages a variety of Python libraries and frameworks to build and run the AI agent:

*   **LangChain:** A framework for developing applications powered by language models. It is used for integrating various components like LLMs, tools, and retrievers.
*   **LangGraph:** An extension of LangChain, used for building robust and stateful multi-actor applications with LLMs. It enables the creation of cyclical graphs for more complex agent behaviors.
*   **Gradio:** An open-source Python library that allows you to quickly create customizable UI components for your ML models, APIs, and data science workflows. It is used to build the web interface for the agent evaluation runner.
*   **Hugging Face Transformers:** Used for loading pre-trained models, specifically for `HuggingFaceEmbeddings` to generate embeddings for the Supabase Vector Store.
*   **Supabase:** An open-source Firebase alternative. In this project, it is used as a vector store to store and retrieve document embeddings, enabling the agent to find similar questions or relevant information.
*   **Tavily:** A search API used for general web searches, providing the agent with up-to-date information from the internet.
*   **WikipediaLoader:** A LangChain document loader for fetching content from Wikipedia.
*   **ArxivLoader:** A LangChain document loader for fetching content from Arxiv.
*   **python-dotenv:** For loading environment variables from a `.env` file.
*   **requests:** A popular Python library for making HTTP requests, used for interacting with the external scoring API.
*   **pandas:** A data manipulation and analysis library, used for displaying the results in a structured format within the Gradio interface.





## 8. References

[1] Hugging Face Agent Course: [https://huggingface.co/learn/deep-rl-course/unit4/introduction](https://huggingface.co/learn/deep-rl-course/unit4/introduction)
[2] GAIA: A Benchmark for General AI Assistants: [https://arxiv.org/abs/2311.12983](https://arxiv.org/abs/2311.12983)
[3] LangChain Documentation: [https://python.langchain.com/docs/get_started/introduction](https://python.langchain.com/docs/get_started/introduction)
[4] LangGraph Documentation: [https://langchain-ai.github.io/langgraph/](https://langchain-ai.github.io/langgraph/)
[5] Gradio Documentation: [https://gradio.app/docs/](https://gradio.app/docs/)
[6] Supabase Documentation: [https://supabase.com/docs](https://supabase.com/docs)
[7] Tavily API: [https://tavily.com/](https://tavily.com/)


