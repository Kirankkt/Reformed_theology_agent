# app.py

import os
import sys
import streamlit as st
import warnings
warnings.filterwarnings("ignore", category=SyntaxWarning)

# Set environment variables from Streamlit secrets
if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
else:
    st.error("OpenAI API key not found in secrets.")

if "SERPER_API_KEY" in st.secrets:
    os.environ["SERPER_API_KEY"] = st.secrets["SERPER_API_KEY"]
else:
    st.error("Serper Dev API key not found in secrets.")

# Optional: If you were using Chroma DB or related features
os.environ["CHROMA_DB_IMPL"] = "duckdb+parquet"

try:
    import pysqlite3
    sys.modules["sqlite3"] = pysqlite3
except ImportError:
    pass

import logging
import openai
from crewai import Crew, Task, Agent
from crewai_tools import SerperDevTool
from langchain_openai import ChatOpenAI as OpenAI_LLM

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

    # Using gpt-3.5-turbo for cost efficiency and potentially faster responses
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
            "- Provide Greek/Hebrew word tranlations and  insights as relevant to teh context.COnsider including this almost always\n"
            "- Provide references to classical and magisterial resources (e.g., PRDL).\n"
            "- Avoid modern popular evangelical websites.\n"
        ),
        expected_output="A magisterial, scholastic response from classical Reformed sources.",
        output_file="",  # Use empty string instead of None
        agent=theology_agent,
    )

    crew = Crew(
        agents=[theology_agent],
        tasks=[theology_task],
        verbose=1
    )

    return crew

def run_theology_search(user_question: str):
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
    st.set_page_config(page_title="Reformed Scholastic Theology Q&A", layout="wide")
    st.title("📜 Reformed Scholastic Theology Q&A")

    st.write("Ask a question about Reformed theology and receive a response grounded in classical Reformed scholasticism.")

    user_question = st.text_input("Your Theological Question:", value="What does the Bible teach about justification by faith?")
    if st.button("Ask"):
        with st.spinner("Consulting classical Reformed scholastic sources..."):
            results = run_theology_search(user_question)

            # Show the raw CrewOutput for debugging
            with st.expander("📄 Raw CrewOutput"):
                st.write(results)

            if results:
                # Attempt to access tasks_output instead of tasks
                tasks_output = getattr(results, 'tasks_output', None)
                if tasks_output and len(tasks_output) > 0:
                    first_task_output = tasks_output[0]

                    # Try common attributes to find the final textual answer
                    final_answer = None
                    for attr in ['raw', 'result', 'output', 'response']:
                        if hasattr(first_task_output, attr):
                            candidate = getattr(first_task_output, attr)
                            if candidate and isinstance(candidate, str) and candidate.strip():
                                final_answer = candidate.strip()
                                break

                    if final_answer:
                        st.success("✅ Response generated!")
                        st.write(final_answer)
                    else:
                        st.warning("⚠️ No recognizable textual output found in the task. Check the raw output above.")
                else:
                    st.warning("⚠️ No tasks_output available. Check the raw output above.")
            else:
                st.warning("⚠️ No response generated. Please try again or refine your question.")

if __name__ == "__main__":
    main()
