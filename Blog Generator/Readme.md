# Deploying Your LangGraph Blog Generator

This guide will walk you through setting up and deploying your LangGraph blog generator application.

## Prerequisites

- Python 3.9+ installed
- pip (Python package installer)
- Git (optional, for version control)

## Step 1: Set Up Your Environment

First, create a project directory and set up a virtual environment:

```bash
# Create project directory
mkdir langgraph-blog-generator
cd langgraph-blog-generator

# Create and activate virtual environment
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

## Step 2: Install Required Packages

Create a `requirements.txt` file with the following content:

```
langchain==0.1.4
langchain-openai==0.0.5
langgraph==0.0.20
pydantic==2.5.2
streamlit==1.28.0
python-dotenv==1.0.0
```

Install the packages:

```bash
pip install -r requirements.txt
```

## Step 3: Configure Environment Variables

Create a `.env` file in your project directory:

```
OPENAI_API_KEY=your_openai_api_key_here
```

Replace `your_openai_api_key_here` with your actual OpenAI API key.

## Step 4: Create the Application File

Create a file named `app.py` and paste the code from the LangGraph Blog Generator code artifact.

## Step 5: Run Locally

Test the application locally:

```bash
streamlit run app.py
```

Your browser should open automatically to `http://localhost:8501` with the blog generator app running.

## Step 6: Deploy to Streamlit Cloud

1. Create a free account on [Streamlit Cloud](https://streamlit.io/cloud)
2. Create a GitHub repository with your code
3. Connect Streamlit Cloud to your GitHub repository
4. Add your `OPENAI_API_KEY` as a secret in the Streamlit Cloud dashboard
5. Deploy the app

Alternatively, you can deploy to other platforms:

## Step 3 : Modified added Streamlit sidebars to configure the openai-api keys.

I will be trying to add the other models so that one can use Groq API keys and use other models also.

![image](https://github.com/user-attachments/assets/ff046b4d-a324-4928-9051-d688c762ce49)

