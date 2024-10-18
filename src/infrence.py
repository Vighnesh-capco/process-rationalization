import os
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
from graphrag.query.input.loaders.dfs import store_entity_semantic_embeddings
from graphrag.query.llm.oai.chat_openai import ChatOpenAI
from graphrag.query.llm.oai.embedding import OpenAIEmbedding
from graphrag.query.llm.oai.typing import OpenaiApiType
from graphrag.query.structured_search.local_search.mixed_context import LocalSearchMixedContext
from graphrag.query.structured_search.local_search.search import LocalSearch
from graphrag.vector_stores.lancedb import LanceDBVectorStore
from graphrag.query.structured_search.global_search.community_context import GlobalCommunityContext
from graphrag.query.structured_search.global_search.search import GlobalSearch
import asyncio
import streamlit as st
from time_tracker import TimeIt
from uiconfig import UiConfig

class ChatBot:
    def __init__(self, input_dir="/Users/tcml/Library/CloudStorage/OneDrive-Capco/Documents/Hackathon-usecase/process-rationalization/src/graph3/ragtest/output/20241014-200506/artifacts", model_name="gpt-4o"):
        os.environ["OPENAI_API_KEY"] = ""
        self.timer = TimeIt()
        self.input_dir = input_dir
        self.llm = ChatOpenAI(
            model=model_name,
            api_type=OpenaiApiType.OpenAI,
            max_retries=20,
        )
        self.token_encoder = tiktoken.get_encoding("cl100k_base")
        self.text_embedder = OpenAIEmbedding(
            api_base=None,
            api_type=OpenaiApiType.OpenAI,
            model="text-embedding-3-small",
            deployment_name="text-embedding-3-small",
            max_retries=20,
        )

    async def local_search(self, query):
        LANCEDB_URI = f"{self.input_dir}/lancedb"
        COMMUNITY_REPORT_TABLE = "create_final_community_reports"
        ENTITY_TABLE = "create_final_nodes"
        ENTITY_EMBEDDING_TABLE = "create_final_entities"
        RELATIONSHIP_TABLE = "create_final_relationships"
        TEXT_UNIT_TABLE = "create_final_text_units"
        COMMUNITY_LEVEL = 2

        # Load data
        entity_df = pd.read_parquet(f"{self.input_dir}/{ENTITY_TABLE}.parquet")
        entity_embedding_df = pd.read_parquet(f"{self.input_dir}/{ENTITY_EMBEDDING_TABLE}.parquet")
        entities = read_indexer_entities(entity_df, entity_embedding_df, COMMUNITY_LEVEL)

        description_embedding_store = LanceDBVectorStore(collection_name="entity_description_embeddings")
        description_embedding_store.connect(db_uri=LANCEDB_URI)
        store_entity_semantic_embeddings(entities=entities, vectorstore=description_embedding_store)

        relationship_df = pd.read_parquet(f"{self.input_dir}/{RELATIONSHIP_TABLE}.parquet")
        relationships = read_indexer_relationships(relationship_df)
        report_df = pd.read_parquet(f"{self.input_dir}/{COMMUNITY_REPORT_TABLE}.parquet")
        reports = read_indexer_reports(report_df, entity_df, COMMUNITY_LEVEL)
        text_unit_df = pd.read_parquet(f"{self.input_dir}/{TEXT_UNIT_TABLE}.parquet")
        text_units = read_indexer_text_units(text_unit_df)

        context_builder = LocalSearchMixedContext(
            community_reports=reports,
            text_units=text_units,
            entities=entities,
            relationships=relationships,
            entity_text_embeddings=description_embedding_store,
            embedding_vectorstore_key=EntityVectorStoreKey.ID,
            text_embedder=self.text_embedder,
            token_encoder=self.token_encoder,
        )

        search_engine = LocalSearch(
            llm=self.llm,
            context_builder=context_builder,
            system_prompt="""You are an assistant that provides answers to the user question based only on the provided context {context_data}.
            Understand the question clearly and generate only the relevant answers.
            If something is not directly involved or related,do not include that information in your response.
            Do Not Hallucinate.
            ---Target response length and format---
                {response_type}
            """,
            token_encoder=self.token_encoder,
            llm_params={"max_tokens":2000},
            context_builder_params={
                "text_unit_prop": 0.5,
                "community_prop": 0.1,
                "conversation_history_max_turns": 5,
                "top_k_mapped_entities": 10,
                "include_entity_rank": True,
                "max_tokens": 12000,
            },
            response_type="multiple paragraphs",
        )

        result = await search_engine.asearch(query)
        await asyncio.sleep(1)  # Simulating async task
        return result.response

    async def global_search(self, query):
        COMMUNITY_REPORT_TABLE = "create_final_community_reports"
        ENTITY_TABLE = "create_final_nodes"
        ENTITY_EMBEDDING_TABLE = "create_final_entities"
        COMMUNITY_LEVEL = 2

        # Load data
        entity_df = pd.read_parquet(f"{self.input_dir}/{ENTITY_TABLE}.parquet")
        report_df = pd.read_parquet(f"{self.input_dir}/{COMMUNITY_REPORT_TABLE}.parquet")
        entity_embedding_df = pd.read_parquet(f"{self.input_dir}/{ENTITY_EMBEDDING_TABLE}.parquet")

        reports = read_indexer_reports(report_df, entity_df, COMMUNITY_LEVEL)
        entities = read_indexer_entities(entity_df, entity_embedding_df, COMMUNITY_LEVEL)

        context_builder = GlobalCommunityContext(
            community_reports=reports,
            entities=entities,
            token_encoder=self.token_encoder,
        )

        search_engine = GlobalSearch(
            llm=self.llm,
            context_builder=context_builder,
            token_encoder=self.token_encoder,
            max_data_tokens=12000,
            map_llm_params={"max_tokens": 1000, "temperature": 0.0},
            reduce_llm_params={"max_tokens": 2000,"temperature": 0.0},
            json_mode=True,
            response_type="multiple paragraphs",
        )

        result = await search_engine.asearch(query)
        await asyncio.sleep(1)
        return result.response

    async def run(self):
        
        st.sidebar.title("Search Options")
        search_type = st.sidebar.radio("Select Search Type", ("Local Search", "Global Search"))
        
        if "messages" not in st.session_state:
            st.session_state.messages = []

        with st.chat_message("assistant", avatar="🤖"):
            st.write("Hi, what questions can I answer?")

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if query := st.chat_input("Type your question here..."):
            st.session_state.messages.append({"role": "user", "content": query})
            with st.chat_message("user", avatar="👥"):
                st.markdown(query)

            with st.chat_message("assistant", avatar="🤖"):
                message_placeholder = st.empty()
                with st.spinner("Thinking..."):
                    self.timer.tick("Chain invoke")
                    if search_type == "Local Search":
                        result = await self.local_search(query)
                        result=result.replace('$', '\$')
                    else:
                        result = await self.global_search(query)
                        result=result.replace('$', '\$')
                    
                    message_placeholder.markdown(result)
                    st.session_state.messages.append({"role": "assistant", "content": result})

if __name__ == "__main__":
    bot = ChatBot()
    asyncio.run(bot.run())
