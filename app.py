# app_crewai.py

import os
import sys
import streamlit as st
import warnings
warnings.filterwarnings("ignore", category=SyntaxWarning)

# 2. Set environment variables from Streamlit secrets
if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
else:
    st.error("OpenAI API key not found in secrets.")

if "SERPER_API_KEY" in st.secrets:
    os.environ["SERPER_API_KEY"] = st.secrets["SERPER_API_KEY"]
else:
    st.error("Serper Dev API key not found in secrets.")

# 3. Set Chroma to use DuckDB to avoid sqlite3 dependency
os.environ["CHROMA_DB_IMPL"] = "duckdb+parquet"

# 4. Import pysqlite3 and override the default sqlite3 if available
try:
    import pysqlite3
    sys.modules["sqlite3"] = pysqlite3
except ImportError:
    st.warning("pysqlite3 is not installed. Proceeding without overriding sqlite3.")

# 5. Import other libraries
import logging
import openai
from crewai import Crew, Task, Agent
from crewai_tools import SerperDevTool
from langchain_openai import ChatOpenAI as OpenAI_LLM

# 6. Configure logging
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

    # Refined agent prompt remains the same
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
            "to Greek/Hebrew lexicons (e.g., BDAG for Greek, HALOT for Hebrew) and can provide rigorous, historical, theological responses."
        ),
        allow_delegation=False,
        tools=[search_tool],
        verbose=1,
    )

    # Updated instructions in the theology_task description
    theology_task = Task(
        description=(
            f"Provide a comprehensive theological response to the following question:\n\n"
            f"'{user_question}'\n\n"
            "Requirements:\n"
            "- Strictly classical Reformed scholastic perspective.\n"
            "- Include direct quotes (with citations) from classical Reformed theologians (e.g., Calvin's Institutes, Turretin's Institutes, "
            "  Bavinck's Reformed Dogmatics, Hodge's Systematic Theology, Warfield's works).\n"
            "- Provide Greek/Hebrew lexical insights (e.g., define and explain key Greek terms using BDAG, Hebrew terms using HALOT) as relevant.\n"
            "- Provide references to classical and magisterial resources (e.g., the Westminster Confession of Faith, the Three Forms of Unity) "
            "  and to recognized repositories (e.g., PRDL: https://www.prdl.org/)\n"
            "- Avoid modern popular evangelical websites (no Desiring God, no Ligonier, etc.).\n"
            "- Emphasize the original languages for key doctrinal terms.\n"
            "- Include links to classical texts where possible (e.g., PRDL pages, digitized editions of classical works)."
        ),
        expected_output="A magisterial, scholastic response from classical Reformed sources, with Greek/Hebrew terms and scholarly citations.",
        output_file="",  # keep as empty string to avoid NoneType error
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

    user_question = st.text_input("Your Theological Question:", value="What does supralapsarianism entail in Reformed theology?")
    if st.button("Ask"):
        with st.spinner("Consulting classical Reformed scholastic sources..."):
            results = run_theology_search(user_question)

            with st.expander("📄 Raw CrewOutput"):
                st.write(results)

            if results:
                # Try to get text from results
                final_answer = getattr(results, 'raw', getattr(results, 'result', str(results)))
                if final_answer:
                    st.success("✅ Response generated!")
                    st.write(final_answer)
                else:
                    st.warning("⚠️ Could not find text output in CrewOutput. Check Raw CrewOutput above.")
            else:
                st.warning("⚠️ No response generated. Please try again or refine your question.")

if __name__ == "__main__":
    main()
