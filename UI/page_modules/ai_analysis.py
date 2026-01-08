"""
AI-Powered Analysis page - Uses LLM for intelligent data analysis
Supports OpenAI and Groq providers
"""

import streamlit as st
import os
import time
from typing import Optional, Dict, Any
import pandas as pd
import numpy as np

# Check if LangChain is available
LANGCHAIN_AVAILABLE = False
GROQ_AVAILABLE = False
try:
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    LANGCHAIN_AVAILABLE = True
except ImportError:
    pass

try:
    from langchain_groq import ChatGroq
    GROQ_AVAILABLE = True
except ImportError:
    pass

try:
    from langchain_openai import ChatOpenAI
except ImportError:
    ChatOpenAI = None


def render_ai_analysis():
    """Render the AI-powered analysis page"""
    st.markdown("## 🤖 AI Analysis")
    
    if 'result' not in st.session_state:
        st.info("Upload a PDF in the Home tab first.")
        return
    
    result = st.session_state.result
    
    # API Key section
    provider = st.radio("Provider", ["groq", "openai"], horizontal=True, index=0)
    st.session_state.llm_provider = provider
    
    api_key = st.text_input(
        f"{'Groq' if provider == 'groq' else 'OpenAI'} API Key",
        type="password",
        value=st.session_state.get(f'{provider}_api_key', ''),
        help="Groq is free: console.groq.com/keys"
    )
    if api_key:
        st.session_state[f'{provider}_api_key'] = api_key
    
    st.markdown("---")
    
    # Model selection
    if provider == 'groq':
        model = st.selectbox("Model", ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768"])
    else:
        model = st.selectbox("Model", ["gpt-4o-mini", "gpt-4o"])
    
    # Analysis
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Analyze Document", type="primary"):
            if not api_key:
                st.error("Enter API key first")
            else:
                with st.spinner("Analyzing..."):
                    analysis = _perform_analysis(api_key, result, model, provider)
                    st.session_state.ai_analysis = analysis
    
    with col2:
        question = st.text_input("Or ask a question:", placeholder="What are the key trends?")
        if question and api_key:
            with st.spinner("Thinking..."):
                answer = _ask_question(api_key, result, question, provider)
                st.markdown(f"**Answer:** {answer}")
    
    # Show results
    if 'ai_analysis' in st.session_state and st.session_state.ai_analysis:
        st.markdown("### Analysis Results")
        st.markdown(st.session_state.ai_analysis)


def _perform_analysis(api_key: str, result, model: str, provider: str) -> str:
    """Perform LLM analysis"""
    if not LANGCHAIN_AVAILABLE:
        return "Install langchain: pip install langchain-core langchain-groq"
    
    try:
        if provider == "groq":
            llm = ChatGroq(model=model, temperature=0.1, api_key=api_key)
        else:
            llm = ChatOpenAI(model=model, temperature=0.1, api_key=api_key)
        
        data = _prepare_data(result)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a data analyst. Provide concise analysis with key findings."),
            ("human", "Analyze this data:\n{data}\n\nProvide: Summary, Key Findings, Recommendations")
        ])
        
        chain = prompt | llm | StrOutputParser()
        return chain.invoke({"data": data})
    except Exception as e:
        return f"Error: {str(e)}"


def _ask_question(api_key: str, result, question: str, provider: str) -> str:
    """Answer a question about the data"""
    if not LANGCHAIN_AVAILABLE:
        return "Install langchain first"
    
    try:
        if provider == "groq":
            llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.1, api_key=api_key)
        else:
            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1, api_key=api_key)
        
        data = _prepare_data(result)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Answer questions about document data concisely."),
            ("human", "Data:\n{data}\n\nQuestion: {question}")
        ])
        
        chain = prompt | llm | StrOutputParser()
        return chain.invoke({"data": data, "question": question})
    except Exception as e:
        return f"Error: {str(e)}"


def _prepare_data(result) -> str:
    """Prepare data summary for LLM"""
    parts = [f"Tables: {len(result.tables)}, Charts: {len(result.graphs)}"]
    
    for i, table in enumerate(result.tables[:5]):
        if hasattr(table, 'data') and table.data is not None:
            try:
                df = pd.DataFrame(table.data)
                parts.append(f"\nTable {i+1}:\n{df.head(10).to_string()}")
            except:
                pass
    
    for i, graph in enumerate(result.graphs[:5]):
        chart_type = graph.graph_type.name
        parts.append(f"\nChart {i+1}: {chart_type}")
        if hasattr(graph, 'extracted_values') and graph.extracted_values:
            parts.append(f"Values: {graph.extracted_values[:20]}")
    
    return "\n".join(parts)
