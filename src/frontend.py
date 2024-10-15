import os
import tempfile
import pandas as pd
import tiktoken

from graphrag.query.context_builder.entity_extraction import EntityVectorStoreKey
from graphrag.query.indexer_adapters import (
    read_indexer_covariates,
    read_indexer_entities,
    read_indexer_relationships,
    read_indexer_reports,
    read_indexer_text_units,
)
from graphrag.query.input.loaders.dfs import (
    store_entity_semantic_embeddings,
)
from graphrag.query.llm.oai.chat_openai import ChatOpenAI
from graphrag.query.llm.oai.embedding import OpenAIEmbedding
from graphrag.query.llm.oai.typing import OpenaiApiType
from graphrag.query.question_gen.local_gen import LocalQuestionGen
from graphrag.query.structured_search.local_search.mixed_context import (
    LocalSearchMixedContext,
)
from graphrag.query.structured_search.local_search.search import LocalSearch
from graphrag.vector_stores.lancedb import LanceDBVectorStore
from graphrag.query.indexer_adapters import read_indexer_entities, read_indexer_reports
from graphrag.query.llm.oai.chat_openai import ChatOpenAI
from graphrag.query.llm.oai.typing import OpenaiApiType
from graphrag.query.structured_search.global_search.community_context import (
    GlobalCommunityContext,
)
from graphrag.query.structured_search.global_search.search import GlobalSearch
import asyncio
import time
import streamlit as st
import extra_streamlit_components as stx
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.embeddings import OpenAIEmbeddings
# from langchain.vectorstores import FAISS
from langchain.llms import OpenAI

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts.prompt import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import Tuple, List, Optional
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
import os
import shutil
from langchain_openai import ChatOpenAI
from streamlit_tags import st_tags 
import re
# from uiconfig import UiConfig
# from infrence import ChatBot

from dotenv import load_dotenv
load_dotenv()




def extract_complexity_from_output(result):
    # Use a regex pattern to find variations of "Complexity level" or "## Complexity" in the output
    pattern = r"(?:Complexity level|## Complexity)\s*:\s*(High|Medium|Low)"
    match = re.search(pattern, result, re.IGNORECASE)  # Case-insensitive search
    
    if match:
        return match.group(1)  # Return the matched complexity level (High, Medium, Low)
    return None

def extract_tags_from_output(output):
        match = re.search(r"## Tags:\n(.*)", output)
        if match:
            return match.group(1).strip() 
        return "No Tags Found"


def update_output_with_new_tags(result, new_tags):
    
    updated_result = re.sub(r"Tags: (.*)", f"Tags: {', '.join(new_tags)}", result)
    return updated_result


def reset_app():
        st.session_state.clear()

    
        if "temp_dir" in st.session_state:
            shutil.rmtree(st.session_state.temp_dir)
            del st.session_state.temp_dir

    
        st.session_state.step = 0
        
def navigate(step):
        st.session_state.step = step
    
def change_step(new_step):
        st.session_state.step = new_step
        
def save_results_to_file(save_dir):
    result_file_path = os.path.join(save_dir, "result.txt")
    
    with open(result_file_path, "w") as result_file:
        for data in st.session_state.process_data:
            filename = data["filename"]
            result = data["result"]
            result_file.write(f"Filename: {filename}\n")
            result_file.write(result)
            result_file.write("\n\n")  # Add spacing between results
    
    st.success(f"All results saved to {result_file_path}!")
        
def render_navigation_buttons(current_step, max_step):
    # Helper function to render buttons in each step
    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        if current_step > 0:
            st.button("Previous", on_click=lambda: set_step(current_step - 1))

    with col2:
        st.button("Reset", on_click=reset_app)

    with col3:
        if current_step < max_step:
            st.button("Next", on_click=lambda: set_step(current_step + 1))

def calculate_text_area_height(text):
    
    min_height = 150  
    line_height = 20  
    num_lines = len(text.splitlines())  
    return min(max(num_lines * line_height, min_height), 500)   

def save_final_output_to_file(save_dir, process_data):
   
    with open(os.path.join(save_dir, "final_output.txt"), "w") as f:
        for data in process_data:
            f.write(f"Process Name: {data['filename']}\n")
            f.write(f"Tags: {', '.join(data['tags'])}\n")
            f.write("\n")        
            
def main():
    # UiConfig.setup()
    
    if 'step' not in st.session_state:
        st.session_state.step = 0
    if 'process_data' not in st.session_state:
        st.session_state.process_data = []

    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
    llm = ChatOpenAI(model="gpt-4o")
    template = """Analyse the Process map {processmap} clearly and generate a clear, detailed, and accurate description of the entire process flow. Ensure that each step, action, and interaction in the process is explained in detail. Identify the primary process name and assign a suitable, descriptive tag that summarizes or categorizes the overall process.

    Provide the output in the following format:
    ## Summary: 
    ## Tags:
    Teams/Roles involved: Identify the teams or roles involved
                        If no swimlanes, say no info available
    Phases:
    For each phase step by step detailed description of activities.
    If any activity is associated with a specific color code, highlight the meaning of that color.
    If there are multiple paths coming out of an activity, create separate branches in the description.
    If there are $ amount range applicable in any process, there should be an easy way to identify them.
    Decision points: For each decision point provide description
    Complexity level: If no of steps are between 1-5 and decision points are between 1-3 the complexity level would be LOW.
                      If no of steps are between 5-10 and decision points are between 3-10 the complexity level would be MEDIUM.
                      If no of steps are between 10+ and decision points are between 10+ the complexity level would be HIGH.
    NOTE: Consider the threshold provided Both steps and decision points count should be matched while assigning complexity level.
    Answer should be only LOW,MEDIUM or HIGH based on the threshold given,Do not write explanation or any other text.
    Controls: If no controls highlighted say no info available
    Systems involved:

    Ensure that the description is precise, detailed, and covering all details present in the process map.
    Do Not Hallucinate
    """
    prompt = ChatPromptTemplate.from_template(template)
  
    steps = ["Extraction", "Categorization", "Aggregation", "Inference"]
    val = stx.stepper_bar(steps=steps)
    #st.info(f"Phase #{val}")
    val = st.session_state.step
    if "process_data" not in st.session_state:
        st.session_state.process_data = []
    
    save_dir="/Users/tcml/Library/CloudStorage/OneDrive-Capco/Documents/process-rationalization/Dev/app/graph/ragtest/input"
    if val == 0:
        st.subheader("Extraction")
        messages = [
        "Processing...",
        "Hang on! Its Almost there"
            ]
       
        uploaded_files = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)

        if st.button("Get transcription") and uploaded_files:
            st.session_state.process_data = []  
            
            with st.spinner("\n".join(messages)):
                
                with tempfile.TemporaryDirectory() as temp_dir:
                  
                    for uploaded_file in uploaded_files:
                        pdf_path = os.path.join(temp_dir, uploaded_file.name)
                        
                        
                        with open(pdf_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())

                        
                        loader = PyPDFLoader(pdf_path)
                        documents = loader.load()

                        
                        chain = (prompt | llm | StrOutputParser())
                        result = chain.invoke({"processmap": documents})
                        result=result.replace('$', '\$')

                        
                        st.session_state.process_data.append({
                            "filename": uploaded_file.name,
                            "result": result,
                            "editable": False 
                        })

            st.success("All PDFs processed successfully!")
            
       
        if "process_data" in st.session_state and st.session_state.process_data:
            for index, data in enumerate(st.session_state.process_data):
                filename = data["filename"]
                result = data["result"]
                editable_key = f"editable_{index}"

                # Initialize the editable state in session_state if not already present
                if editable_key not in st.session_state:
                    st.session_state[editable_key] = False

                with st.expander(f"{filename} Result", expanded=False):
                    # Create a unique key for each text area and button
                    result_key = f"result_{index}"
                    edit_button_key = f"edit_button_{index}"

                    if st.session_state[editable_key]:
                        height = calculate_text_area_height(result)
                        editable_result = st.text_area(f"Edit result for {filename}", value=result, key=result_key, height=height)
                        if st.button(f"Save {filename} Result", key=f"save_button_{index}"):
                            
                            st.session_state.process_data[index]["result"] = editable_result
                            st.session_state[editable_key] = False  # Exit edit mode
                            st.success(f"{filename} result saved!")
                    else:
                        st.write(result)
                        
                        if st.button(f"Edit {filename} Result", key=edit_button_key):
                            st.session_state[editable_key] = True 
        render_navigation_buttons(val, 3)
    elif val == 1:
        st.subheader("Categorization")

        if st.session_state.process_data:
            process_names = []
            tags = []
            complexity_levels = []
            edited_data = []

            
            for data in st.session_state.process_data:
                filename = data["filename"].replace(".pdf", "")
                result = data["result"]

                # Extract tags from the result
                tags_found = extract_tags_from_output(result) or []  
                if isinstance(tags_found, str):
                    tags_found = tags_found.split(", ")  
                

                complexity = extract_complexity_from_output(result) or "Unknown"
                
                process_names.append(filename)
                tags.append(tags_found)
                complexity_levels.append(complexity)
                edited_data.append({"filename": filename, "result": result, "tags": tags_found, "complexity": complexity})

            df = pd.DataFrame({
                "Process Name": process_names,
                "Tags": tags,
                "Complexity": complexity_levels
            })

            st.write("Process Name, Initial Tags, and Complexity:")
            st.table(df)

            for i, tag_list in enumerate(tags):
                st.write(f"Tags for {process_names[i]}:")
                
                
                editable_tags = st_tags(
                    label='Edit Tags',
                    text='Press enter to add more tags',
                    value=tag_list or [], 
                    suggestions=["Process", "Map", "Workflow"],  
                    key=f"tag_edit_{i}"
                )

                complexity = complexity_levels[i]
                complexity_color = {
                    "HIGH": "red",
                    "MEDIUM": "orange",
                    "LOW": "green"
                }.get(complexity, "gray")  
                
               
                st.markdown(f"<span style='color:{complexity_color}; font-weight:bold;'>Complexity: {complexity}</span>", unsafe_allow_html=True)

                if st.button(f"Save Tags for {process_names[i]}"):
                    
                    existing_tags = tag_list or []  
                    updated_tags = editable_tags  

                    final_tags = set(existing_tags)  
                    final_tags.update(updated_tags)  

                    for tag in existing_tags:
                        if tag not in updated_tags:
                            final_tags.discard(tag)  

                    st.session_state.process_data[i]["tags"] = list(final_tags)

                    original_result = st.session_state.process_data[i]["result"]
                    updated_tags_section = "## Tags:\n" + ", ".join(final_tags)  # Create the updated tags section

                   
                    lines = original_result.splitlines()
                    new_result = []
                    for line in lines:
                        if line.startswith("## Tags:"):
                            new_result.append(updated_tags_section)  
                        else:
                            new_result.append(line)  
                    
                    st.session_state.process_data[i]["result"] = "\n".join(new_result)  
                    
                    st.success("Tags and complexity level saved successfully!")

            if st.button("Generate Final Output"):
                final_output = ""
                for data in st.session_state.process_data:
                    final_output += f"{data['filename']}: {data['result']}\n\n"  
                
                
                save_path = "final_output.txt"
                with open(save_path, "w") as f:
                    f.write(final_output)
                st.success(f"Final output saved to {save_path}")

        render_navigation_buttons(val, 3)
    
    elif val == 2:
        st.subheader("Aggregation")
        
        st.write("### Select a Knowledge Base")
        
        
        knowledge_base_option = st.selectbox(
            "Select a Knowledge Base:",
            options=["Capco-demo", "Demo 2"]
        )

      
        if st.button("Initialize Knowledge Base"):
            
            with st.spinner("Please wait, initializing your Knowledge Base..."):
               
                import time
                time.sleep(3)  
                
                
                st.success("Your Knowledge Base is ready to use!")

        render_navigation_buttons(val, 3)
   
    elif val == 3:
        async def main():
            st.subheader("Inference Phase")
            with st.sidebar:
                render_navigation_buttons(val, 3)
            # chatbot = ChatBot()
            # await chatbot.run()
        if __name__ == "__main__":
            asyncio.run(main())
                

def set_step(new_step):
    st.session_state.step = new_step

if __name__ == "__main__":
    main()
