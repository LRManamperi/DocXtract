"""
Example: Using LangChain-based intelligent extraction pipeline
"""

import os
from docxtract import LangChainExtractionPipeline, DocumentQueryAgent

# Set your OpenAI API key
# os.environ['OPENAI_API_KEY'] = 'your-api-key-here'

def basic_ai_extraction():
    """Basic extraction with AI analysis"""
    
    # Initialize pipeline with AI analysis enabled
    pipeline = LangChainExtractionPipeline(
        use_analysis=True  # LLM will analyze document structure
    )
    
    # Extract from PDF
    result = pipeline.extract(
        'test_combined.pdf',
        pages='all',
        validate_data=True  # LLM will validate extracted data
    )
    
    # Show analysis
    if result['analysis']:
        analysis = result['analysis']
        print(f"\n📊 Document Analysis:")
        print(f"  Type: {analysis.document_type}")
        print(f"  Tables: {analysis.has_tables}")
        print(f"  Charts: {analysis.has_charts}")
        print(f"  Structure: {analysis.table_structure}")
        print(f"  Confidence: {analysis.confidence:.0%}")
        print(f"  Reasoning: {analysis.reasoning}")
    
    # Show extraction results
    extraction = result['extraction']
    print(f"\n📤 Extraction Results:")
    print(f"  Tables: {len(extraction.tables)}")
    print(f"  Charts: {len(extraction.graphs)}")
    
    # Show validation
    if result['validation']:
        print(f"\n✅ Validation:")
        for item in result['validation']['tables']:
            val = item['validation']
            print(f"  Table {item['table_index']}: {val['confidence']:.0%} confidence")
            if val['issues']:
                print(f"    Issues: {val['issues']}")
    
    return result


def conversational_query():
    """Use conversational agent to query data"""
    
    # First extract data
    pipeline = LangChainExtractionPipeline(use_analysis=True)
    result = pipeline.extract('test_combined.pdf')
    
    # Initialize query agent
    agent = DocumentQueryAgent(result)
    
    # Ask questions
    questions = [
        "How many tables were found?",
        "What types of charts are in the document?",
        "What data is in the first table?",
        "Summarize the key findings"
    ]
    
    print("\n💬 Conversational Query Demo:\n")
    for question in questions:
        print(f"Q: {question}")
        answer = agent.query(question)
        print(f"A: {answer}\n")


def batch_processing():
    """Process multiple PDFs with AI"""
    
    pipeline = LangChainExtractionPipeline(use_analysis=True)
    
    pdf_files = [
        'test_charts.pdf',
        'test_tables.pdf',
        'test_combined.pdf'
    ]
    
    results = []
    for pdf_file in pdf_files:
        if os.path.exists(pdf_file):
            print(f"\n📄 Processing: {pdf_file}")
            result = pipeline.extract(pdf_file, validate_data=False)
            
            # Show quick stats
            extraction = result['extraction']
            print(f"  ✓ Tables: {len(extraction.tables)}")
            print(f"  ✓ Charts: {len(extraction.graphs)}")
            
            if result['analysis']:
                print(f"  ✓ Type: {result['analysis'].document_type}")
            
            results.append(result)
    
    return results


def custom_llm_config():
    """Use custom LLM configuration"""
    from langchain_openai import ChatOpenAI
    
    # Use GPT-3.5 for faster/cheaper processing
    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0,
        max_tokens=1000
    )
    
    pipeline = LangChainExtractionPipeline(
        llm=llm,
        use_analysis=True
    )
    
    result = pipeline.extract('test_combined.pdf')
    return result


def analyze_only():
    """Just analyze document without full extraction"""
    from docxtract import IntelligentDocumentAnalyzer
    
    analyzer = IntelligentDocumentAnalyzer()
    
    # Quick analysis
    analysis = analyzer.analyze('test_combined.pdf', quick_scan=True)
    
    print(f"\n🔍 Document Analysis:")
    print(f"  Document Type: {analysis.document_type}")
    print(f"  Has Tables: {'Yes' if analysis.has_tables else 'No'}")
    print(f"  Has Charts: {'Yes' if analysis.has_charts else 'No'}")
    print(f"  Table Structure: {analysis.table_structure}")
    print(f"  Chart Types: {', '.join(analysis.chart_types) if analysis.chart_types else 'None'}")
    print(f"\n  Recommended Methods:")
    for key, value in analysis.recommended_methods.items():
        print(f"    {key}: {value}")
    print(f"\n  AI Reasoning:")
    print(f"    {analysis.reasoning}")
    print(f"\n  Confidence: {analysis.confidence:.0%}")
    
    return analysis


if __name__ == '__main__':
    print("=" * 60)
    print("DocXtract + LangChain Examples")
    print("=" * 60)
    
    # Check API key
    if not os.getenv('OPENAI_API_KEY'):
        print("\n⚠️  Set OPENAI_API_KEY environment variable first!")
        print("Example: export OPENAI_API_KEY='your-key-here'")
    else:
        print("\n1. Basic AI Extraction")
        print("-" * 60)
        # basic_ai_extraction()
        
        print("\n2. Document Analysis Only")
        print("-" * 60)
        # analyze_only()
        
        print("\n3. Conversational Query")
        print("-" * 60)
        # conversational_query()
        
        print("\n4. Batch Processing")
        print("-" * 60)
        # batch_processing()
        
        print("\nUncomment the function calls to run examples!")
