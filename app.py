import streamlit as st
import openai
from crewai import Crew, Task, Agent
from crewai_tools import SerperDevTool
from langchain_openai import ChatOpenAI as OpenAI_LLM

# Access the secrets stored in Streamlit's cloud environment
openai.api_key = st.secrets["OPENAI_API_KEY"]
serper_api_key = st.secrets["SERPER_API_KEY"]

# Create the LLM
llm = OpenAI_LLM(
    model="gpt-3.5-turbo", 
    temperature=0.0,
    max_tokens=800,
)

# Tools (now using the secret API key)
search = SerperDevTool(api_key=serper_api_key)

# Create the agent with stricter, classical Reformed scholastic prompt constraints
theology_agent = Agent(
    llm=llm,
    role="Reformed Scholastic Theology Assistant",
    goal=(
        "Provide a scholarly, detailed theological response strictly from a classical Reformed scholastic perspective. "
        "Avoid modern evangelical websites. Use only magisterial confessions, classical Reformed scholastics, "
        "and classical resources (like PRDL) for references."
    ),
    backstory=(
        "You are a Reformed scholastic theologian well-versed in the works of Calvin, Turretin, Bavinck, "
        "the Princeton theologians, and other classical Reformed sources. You have access to Greek/Hebrew lexicons."
    ),
    allow_delegation=False,
    tools=[search],
    verbose=1,
)

# Streamlit UI
st.title("Reformed Scholastic Theology Q&A")
st.write("Ask a question about Reformed theology and receive a response grounded in classical Reformed scholasticism.")

user_question = st.text_input("Your Question:", value="What does the Bible teach about justification by faith?")
ask_button = st.button("Ask")

if ask_button and user_question.strip():
    # Dynamically create the task based on user input
    task = Task(
        description=(
            f"Provide a comprehensive theological response to the question:\n\n'{user_question}'\n\n"
            "Requirements:\n"
            "- Strictly classical Reformed scholastic perspective.\n"
            "- Include quotes from classical theologians (Calvin, Turretin, Bavinck, Hodge, Warfield).\n"
            "- Provide Greek/Hebrew insights as relevant.\n"
            "- Provide references to classical and magisterial resources (e.g., PRDL).\n"
            "- Avoid modern popular evangelical websites (e.g., no Desiring God, Ligonier, etc.).\n"
        ),
        expected_output="A magisterial, scholastic response from classical Reformed sources.",
        output_file=None,
        agent=theology_agent,
    )

    crew = Crew(agents=[theology_agent], tasks=[task], verbose=1)
    results = crew.kickoff()
    
    st.markdown("### Scholastic Response:")
    st.write(results[0])
