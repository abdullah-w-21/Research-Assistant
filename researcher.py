import os
import requests
import json
import streamlit as st
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.tools import Tool


# Load environment variables
load_dotenv()
serper_api_key = os.getenv("SERP_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")


# Initialize Groq LLM (Llama 3.1 70B)
groq_llm = ChatGroq(model_name="llama-3.1-70b-versatile", api_key=groq_api_key)




def google_scholar_search(search_keyword):
   url = "https://google.serper.dev/scholar"
   payload = json.dumps({"q": search_keyword})
   headers = {
       'X-API-KEY': serper_api_key,
       'Content-Type': 'application/json'
   }
   response = requests.post(url, headers=headers, data=payload)
   return response.json()


def create_vector_db(pdf_files):
   documents = []
   for pdf_file in pdf_files:
       loader = PyPDFLoader(pdf_file)
       documents.extend(loader.load())


   text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
   texts = text_splitter.split_documents(documents)


   embeddings = HuggingFaceEmbeddings()
   db = FAISS.from_documents(texts, embeddings)
   return db


# init tools


google_scholar_tool = Tool(
   name="Google Scholar Search",
   func=google_scholar_search,
   description="Searches Google Scholar for academic articles based on the provided keyword."
)


# define agnets


researcher = Agent(
   role='Junior Researcher',
   goal='Conduct thorough research on the given topic using Google Scholar and educational sites',
   backstory='You are a diligent junior researcher with a keen eye for academic sources.',
   llm=groq_llm,
   tools=[google_scholar_tool]
)


senior_researcher = Agent(
   role='Senior Researcher',
   goal='Validate and synthesize research findings, ensuring high-quality output',
   backstory='You are an experienced researcher with expertise in critically analyzing and integrating information.',
   llm=groq_llm
)


research_director = Agent(
   role='Research Director',
   goal='Oversee the entire research process and ensure the final output meets academic standards',
   backstory='You are a seasoned research director with a track record of producing high-impact academic papers.',
   llm=groq_llm
)


# Define tasks


def research_task(topic, keywords, description, pdf_db=None):
   return Task(
       description=f"Research the topic: {topic}. Use keywords: {keywords}. Consider this description: {description}. Use Google Scholar and educational sites. If a PDF database is provided, use it as the primary source.",
       expected_output="A comprehensive research report with relevant academic sources and key findings.",
       agent=researcher
   )


def validate_task():
   return Task(
       description="Review and validate the research findings. Ensure all information is accurate and properly cited. Request revisions from the Junior Researcher if necessary.",
       expected_output="A validation report highlighting the strengths and weaknesses of the research, with suggestions for improvement if needed.",
       agent=senior_researcher
   )


def synthesize_task():
   return Task(
       description="Synthesize the validated research into a coherent final output with proper citations. Ensure the document meets high academic standards.",
       expected_output="A well-structured, comprehensive research paper that synthesizes all the findings and meets high academic standards.",
       agent=research_director
   )


# making crew


def create_research_crew(topic, keywords, description, pdf_db=None):
   tasks = [
       research_task(topic, keywords, description, pdf_db),
       validate_task(),
       synthesize_task()
   ]
   return Crew(
       agents=[researcher, senior_researcher, research_director],
       tasks=tasks,
       process=Process.sequential
   )




# Streamlit frontend


st.title("Advanced Research Assistant")


# Form input for research details
with st.form("research_form"):
   topic = st.text_input("Main Research Topic")
   keywords = st.text_input("Related Keywords (comma-separated)")
   description = st.text_area("Brief Description of Research Objective")
   pdf_files = st.file_uploader("Upload PDF files (optional)", accept_multiple_files=True)
   submit_button = st.form_submit_button(label="Start Research")


if submit_button:
   pdf_db = None
   if pdf_files:
       with st.spinner("Processing PDF files..."):
           pdf_db = create_vector_db(pdf_files)


   research_crew = create_research_crew(topic, keywords, description, pdf_db)


   st.write("Research in progress. Agent conversations will appear below:")


   # conversation_placeholder = st.empty()
   #
   #
   # def process_output(output):
   #     conversation_placeholder.text(output)
   #     return output
   #
   #
   # # Set the process_output function as the callback
   # research_crew.process_output = process_output


   result = research_crew.kickoff()


   st.write("Research completed. Final output:")
   st.write(result)

