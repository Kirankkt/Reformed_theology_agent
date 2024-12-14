# app_theology.py

import os
import sys
import streamlit as st
import warnings
warnings.filterwarnings("ignore", category=SyntaxWarning)

# 1. Set environment variables from Streamlit secrets
if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
else:
    st.error("OpenAI API key not found in secrets.")

if "SERPER_API_KEY" in st.secrets:
    os.environ["SERPER_API_KEY"] = st.secrets["SERPER_API_KEY"]
else:
    st.error("Serper Dev API key not found in secrets.")

# 2. Import pysqlite3 if needed (only if you were using a Chroma DB setup, optional)
try:
    import pysqlite3
    sys.modules["sqlite3"] = pysqlite3
except ImportError:
    # If not needed, this can be ignored
    pass

# 3. Import other libraries after environment setup
import logging
import openai
from crewai import Crew, Task, Agent
from crewai_tools import SerperDevTool
from langchain_openai import ChatOpenAI as OpenAI_LLM

# 4. Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler("theology_output.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

def create_theology_crew(user_question: str):
    openai_api_key = os.environ.get('OPENAI_API_KEY')
    serper_api_key = os.environ.get('SERPER_API_KEY')

    if not openai_api_key or not serper_api_key:
        raise ValueError("Missing API keys in environment variables.")

    llm = OpenAI_LLM(
        model="gpt-3.5-turbo",
        temperature=0.0,
        max_tokens=800,
    )

    search_tool = SerperDevTool(api_key=serper_api_key)

    theology_agent = Agent(
        llm=llm,
        role="Reformed Scholastic Theology Assistant",
        goal=(
            "Provide a scholarly, detailed theological response strictly from a classical Reformed scholastic perspective. "
            "Avoid modern evangelical websites. Use only magisterial confessions, classical Reformed scholastics, "
            "and classical resources (like PRDL) for references."
        ),
        backstory=(
            "You are a Reformed scholastic theologian steeped in the works of Calvin, Turretin, Bavinck, Hodge, Warfield, "
            "Vos, and other classical Reformed sources, as well as classical scholasticism including Aquinas. You have access "
            "to Greek/Hebrew lexicons and can provide rigorous, historical, theological responses."
        ),
        allow_delegation=False,
        tools=[search_tool],
        verbose=1,
    )

    theology_task = Task(
        description=(
            f"Provide a comprehensive theological response to the following question:\n\n"
            f"'{user_question}'\n\n"
            "Requirements:\n"
            "- Strictly classical Reformed scholastic perspective.\n"
            "- Include quotes from classical theologians (Calvin, Turretin, Bavinck, Hodge, Warfield).\n"
            "- Provide Greek/Hebrew insights as relevant.\n"
            "- Provide references to classical and magisterial resources (e.g., PRDL).\n"
            "- Avoid modern popular evangelical websites.\n"
        ),
        expected_output="A magisterial, scholastic response from classical Reformed sources.",
        # Instead of None, we set output_file to an empty string
        output_file="",
        agent=theology_agent,
    )

    crew = Crew(
        agents=[theology_agent],
        tasks=[theology_task],
        verbose=1
    )

    return crew


def run_theology_search(user_question: str):
    """
    Run the theological search/response generation process and return the text result.
    """
    try:
        logging.info("Initiating theological query")
        crew = create_theology_crew(user_question)
        results = crew.kickoff()
        logging.info("Theological query completed")

        return results
    except Exception as e:
        logging.error(f"Error during theological search: {e}", exc_info=True)
        return None

def main():
    # Set Streamlit page config
    st.set_page_config(page_title="Reformed Scholastic Theology Q&A", layout="wide")
    st.title("📜 Reformed Scholastic Theology Q&A")

    st.write("Ask a question about Reformed theology and receive a response grounded in classical Reformed scholasticism.")

    user_question = st.text_input("Your Theological Question:", value="What does the Bible teach about justification by faith?")
    if st.button("Ask"):
        with st.spinner("Consulting classical Reformed scholastic sources..."):
            results = run_theology_search(user_question)
            if results:
                st.success("✅ Response generated!")
                # Check if there's an attribute like 'final' or 'output'
                # This depends on crewai's documentation.
                # Let's assume results is a CrewOutput with a `final` attribute:
                st.write(results.final)
            else:
                st.warning("⚠️ No response generated. Please try again or refine your question.")


if __name__ == "__main__":
    main()
