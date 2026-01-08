"""
LangChain-based intelligent document extraction pipeline
Provides clean, modular architecture with LLM-powered analysis
"""

from typing import List, Dict, Optional, Any
from dataclasses import dataclass
import os
from pathlib import Path
import json

# LangChain imports - using correct import paths for newer versions
LANGCHAIN_AVAILABLE = False
try:
    from langchain_openai import ChatOpenAI
    from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
    from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
    from langchain_core.runnables import RunnablePassthrough, RunnableParallel
    from langchain_core.messages import HumanMessage, SystemMessage
    LANGCHAIN_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ LangChain not available: {e}. Install: pip install langchain-openai langchain-core")

from .extractors import DocXtract
from .data_structures import ExtractionResult, Table, Graph


@dataclass
class DocumentAnalysis:
    """LLM analysis of document structure"""
    document_type: str
    has_tables: bool
    has_charts: bool
    table_structure: str  # 'bordered', 'unstructured', 'mixed'
    chart_types: List[str]
    recommended_methods: Dict[str, str]
    confidence: float
    reasoning: str


class IntelligentDocumentAnalyzer:
    """
    Uses LLM to analyze document structure and recommend extraction methods
    """
    
    def __init__(self, llm: Optional[Any] = None, api_key: Optional[str] = None):
        """
        Initialize the analyzer
        
        Args:
            llm: LangChain LLM instance (if None, will create ChatOpenAI)
            api_key: OpenAI API key (will check env if None)
        """
        if not LANGCHAIN_AVAILABLE:
            raise ImportError("LangChain is required. Install: pip install langchain langchain-openai")
        
        # Initialize LLM
        if llm is None:
            api_key = api_key or os.getenv('OPENAI_API_KEY')
            if not api_key:
                raise ValueError("OpenAI API key required. Set OPENAI_API_KEY env var or pass api_key parameter")
            self.llm = ChatOpenAI(model="gpt-4", temperature=0, api_key=api_key)
        else:
            self.llm = llm
        
        # Create analysis chain
        self._setup_analysis_chain()
    
    def _setup_analysis_chain(self):
        """Setup the document analysis chain"""
        
        analysis_prompt = PromptTemplate(
            input_variables=["page_count", "has_images", "text_sample"],
            template="""You are an expert document analyzer. Analyze this PDF document and provide extraction recommendations.

Document Information:
- Total Pages: {page_count}
- Contains Images/Charts: {has_images}
- Text Sample: {text_sample}

Analyze the document and provide:
1. Document type (report, invoice, presentation, scientific paper, etc.)
2. Whether it contains tables (true/false)
3. Whether it contains charts/graphs (true/false)
4. Table structure type: 'bordered' (clear grid lines), 'unstructured' (no clear borders), or 'mixed'
5. Types of charts present (bar, line, pie, scatter, or none)
6. Recommended extraction methods for tables and charts
7. Confidence level (0-1)
8. Brief reasoning for your recommendations

Respond in JSON format:
{{
    "document_type": "...",
    "has_tables": true/false,
    "has_charts": true/false,
    "table_structure": "bordered|unstructured|mixed",
    "chart_types": ["bar", "line", ...],
    "recommended_methods": {{
        "table_detector": "line_based|ml_based|text_clustering",
        "table_parser": "grid|stream|unstructured",
        "chart_extractor": "cv_based|ocr_based|hybrid"
    }},
    "confidence": 0.0-1.0,
    "reasoning": "..."
}}"""
        )
        
        self.analysis_chain = (
            analysis_prompt 
            | self.llm 
            | JsonOutputParser()
        )
    
    def analyze(self, pdf_path: str, quick_scan: bool = True) -> DocumentAnalysis:
        """
        Analyze PDF document structure
        
        Args:
            pdf_path: Path to PDF file
            quick_scan: If True, only analyze first few pages
            
        Returns:
            DocumentAnalysis object with recommendations
        """
        import fitz
        
        doc = fitz.open(pdf_path)
        page_count = len(doc)
        
        # Quick scan first 3 pages or all if quick_scan=False
        pages_to_scan = min(3, page_count) if quick_scan else page_count
        
        # Check for images/charts
        has_images = False
        text_sample = ""
        
        for i in range(pages_to_scan):
            page = doc[i]
            
            # Check for images
            if len(page.get_images()) > 0:
                has_images = True
            
            # Extract text sample
            text = page.get_text()
            if text:
                text_sample += text[:500] + "\n"
        
        doc.close()
        
        # Get LLM analysis
        result = self.analysis_chain.invoke({
            "page_count": page_count,
            "has_images": has_images,
            "text_sample": text_sample[:1000]  # Limit text sample
        })
        
        return DocumentAnalysis(**result)


class LangChainExtractionPipeline:
    """
    Complete LangChain-based extraction pipeline with intelligent routing
    """
    
    def __init__(self, llm: Optional[Any] = None, api_key: Optional[str] = None,
                 use_analysis: bool = True):
        """
        Initialize the pipeline
        
        Args:
            llm: LangChain LLM instance
            api_key: OpenAI API key
            use_analysis: Whether to use LLM for document analysis
        """
        self.use_analysis = use_analysis and LANGCHAIN_AVAILABLE
        
        if self.use_analysis:
            self.analyzer = IntelligentDocumentAnalyzer(llm=llm, api_key=api_key)
            self.llm = self.analyzer.llm
            self._setup_validation_chain()
        else:
            self.analyzer = None
            self.llm = None
        
        # Initialize extractors
        self.extractors = {}
    
    def _setup_validation_chain(self):
        """Setup data validation chain"""
        
        validation_prompt = PromptTemplate(
            input_variables=["data_type", "raw_data"],
            template="""You are a data quality expert. Validate and clean this extracted data.

Data Type: {data_type}
Raw Data: {raw_data}

Tasks:
1. Check for obvious errors or inconsistencies
2. Clean and standardize the data
3. Identify any missing or corrupted values
4. Provide confidence score

Respond in JSON:
{{
    "is_valid": true/false,
    "cleaned_data": [...],
    "issues": ["issue1", "issue2"],
    "confidence": 0.0-1.0,
    "suggestions": "..."
}}"""
        )
        
        self.validation_chain = (
            validation_prompt 
            | self.llm 
            | JsonOutputParser()
        )
    
    def extract(self, pdf_path: str, pages: str = 'all',
                validate_data: bool = True) -> Dict[str, Any]:
        """
        Extract data from PDF with intelligent routing
        
        Args:
            pdf_path: Path to PDF file
            pages: Pages to process
            validate_data: Whether to validate with LLM
            
        Returns:
            Dictionary with extraction results and metadata
        """
        result = {
            'pdf_path': pdf_path,
            'analysis': None,
            'extraction': None,
            'validation': None,
            'metadata': {}
        }
        
        # Step 1: Analyze document (if enabled)
        if self.use_analysis:
            print("🔍 Analyzing document structure with LLM...")
            analysis = self.analyzer.analyze(pdf_path)
            result['analysis'] = analysis
            
            # Get recommended methods
            table_detector = analysis.recommended_methods.get('table_detector', 'line_based')
            table_flavor = self._map_flavor(analysis.table_structure)
            
            print(f"  📋 Document type: {analysis.document_type}")
            print(f"  📊 Recommended method: {table_flavor}")
            print(f"  💡 Reasoning: {analysis.reasoning}")
        else:
            # Default to lattice method
            table_flavor = 'lattice'
        
        # Step 2: Extract with recommended methods
        print(f"\n📤 Extracting with flavor: {table_flavor}")
        extractor = self._get_extractor(table_flavor)
        extraction_result = extractor.extract(pdf_path, pages=pages, table_flavor=table_flavor)
        result['extraction'] = extraction_result
        
        # Step 3: Validate data (if enabled and data found)
        if validate_data and self.use_analysis and (extraction_result.tables or extraction_result.graphs):
            print("\n✅ Validating extracted data with LLM...")
            result['validation'] = self._validate_extraction(extraction_result)
        
        # Step 4: Add metadata
        result['metadata'] = {
            'total_tables': len(extraction_result.tables),
            'total_charts': len(extraction_result.graphs),
            'pages_processed': extraction_result.n_pages,
            'extraction_method': table_flavor
        }
        
        return result
    
    def _map_flavor(self, structure: str) -> str:
        """Map structure analysis to extraction flavor"""
        mapping = {
            'bordered': 'lattice',
            'unstructured': 'stream',
            'mixed': 'lattice'  # Start with lattice, fallback handles rest
        }
        return mapping.get(structure, 'lattice')
    
    def _get_extractor(self, flavor: str) -> DocXtract:
        """Get or create extractor for flavor"""
        if flavor not in self.extractors:
            # Configure based on flavor
            if flavor == 'lattice':
                self.extractors[flavor] = DocXtract(
                    use_ml=True,
                    extract_chart_data=True,
                    handle_unstructured=True
                )
            elif flavor == 'stream':
                self.extractors[flavor] = DocXtract(
                    use_ml=False,
                    extract_chart_data=True,
                    handle_unstructured=True
                )
            else:
                self.extractors[flavor] = DocXtract()
        
        return self.extractors[flavor]
    
    def _validate_extraction(self, result: ExtractionResult) -> Dict[str, Any]:
        """Validate extracted data using LLM"""
        validations = {
            'tables': [],
            'charts': []
        }
        
        # Validate tables
        for i, table in enumerate(result.tables[:3]):  # Limit to first 3
            if table.data is not None:
                validation = self.validation_chain.invoke({
                    'data_type': 'table',
                    'raw_data': str(table.data[:5]) if hasattr(table.data, '__getitem__') else str(table.data)
                })
                validations['tables'].append({
                    'table_index': i,
                    'validation': validation
                })
        
        # Validate charts
        for i, graph in enumerate(result.graphs[:3]):  # Limit to first 3
            if hasattr(graph, 'extracted_values') and graph.extracted_values:
                validation = self.validation_chain.invoke({
                    'data_type': f'{graph.graph_type.value} chart',
                    'raw_data': str(graph.extracted_values[:10])
                })
                validations['charts'].append({
                    'chart_index': i,
                    'validation': validation
                })
        
        return validations


class DocumentQueryAgent:
    """
    Conversational agent to query extracted document data
    """
    
    def __init__(self, extraction_result: Dict[str, Any], llm: Optional[Any] = None,
                 api_key: Optional[str] = None):
        """
        Initialize query agent
        
        Args:
            extraction_result: Result from LangChainExtractionPipeline.extract()
            llm: LangChain LLM instance
            api_key: OpenAI API key
        """
        if not LANGCHAIN_AVAILABLE:
            raise ImportError("LangChain required")
        
        self.extraction_result = extraction_result
        
        # Initialize LLM
        if llm is None:
            api_key = api_key or os.getenv('OPENAI_API_KEY')
            self.llm = ChatOpenAI(model="gpt-4", temperature=0, api_key=api_key)
        else:
            self.llm = llm
        
        self._setup_query_chain()
    
    def _setup_query_chain(self):
        """Setup conversational query chain"""
        
        # Prepare context from extraction results
        self.context = self._prepare_context()
        
        query_prompt = PromptTemplate(
            input_variables=["context", "question"],
            template="""You are a helpful assistant analyzing extracted PDF data.

Available Data:
{context}

User Question: {question}

Provide a clear, concise answer based on the extracted data. If the data doesn't contain the answer, say so.
Include specific values and references when possible."""
        )
        
        self.query_chain = (
            {{"context": lambda x: self.context, "question": RunnablePassthrough()}}
            | query_prompt
            | self.llm
            | StrOutputParser()
        )
    
    def _prepare_context(self) -> str:
        """Prepare context from extraction results"""
        context_parts = []
        
        extraction = self.extraction_result.get('extraction')
        if not extraction:
            return "No data available"
        
        # Add tables info
        if extraction.tables:
            context_parts.append(f"Tables found: {len(extraction.tables)}")
            for i, table in enumerate(extraction.tables[:5]):
                if hasattr(table, 'data') and table.data is not None:
                    shape = table.data.shape if hasattr(table.data, 'shape') else (0, 0)
                    context_parts.append(f"  Table {i+1}: {shape[0]} rows × {shape[1]} columns on page {table.page}")
        
        # Add charts info
        if extraction.graphs:
            context_parts.append(f"\nCharts found: {len(extraction.graphs)}")
            for i, graph in enumerate(extraction.graphs[:5]):
                chart_type = graph.graph_type.value.replace('_', ' ').title()
                values_count = len(graph.extracted_values) if hasattr(graph, 'extracted_values') else 0
                context_parts.append(f"  Chart {i+1}: {chart_type} with {values_count} data points on page {graph.page}")
        
        return "\n".join(context_parts)
    
    def query(self, question: str) -> str:
        """
        Query the extracted data
        
        Args:
            question: Natural language question
            
        Returns:
            Answer based on extracted data
        """
        return self.query_chain.invoke(question)
