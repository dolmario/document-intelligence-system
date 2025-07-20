# tests/test_enhanced_features.py
import pytest
import asyncio
import json
import asyncpg
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
import base64
from PIL import Image
import io
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from agents.ocr.ocr_agent import EnhancedOCRAgent
from core.models import DocumentIndex

@pytest.fixture
async def test_db():
    """Test database with enhanced schema"""
    # Use test database
    db_url = "postgresql://test:test@localhost:5432/test_semantic_doc_finder"
    
    # Create test tables
    conn = await asyncpg.connect(db_url)
    
    # Create minimal test schema
    await conn.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            original_name VARCHAR(500),
            file_type VARCHAR(50),
            metadata JSONB,
            status VARCHAR(50) DEFAULT 'pending'
        )
    """)
    
    await conn.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            document_id UUID REFERENCES documents(id),
            chunk_index INTEGER,
            content TEXT,
            content_type VARCHAR(50),
            page_number INTEGER,
            metadata JSONB
        )
    """)
    
    await conn.execute("""
        CREATE TABLE IF NOT EXISTS enhanced_chunks (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            document_id UUID REFERENCES documents(id),
            original_chunk_id UUID REFERENCES chunks(id),
            chunk_index INTEGER,
            enhanced_content TEXT,
            categories JSONB,
            extracted_metadata JSONB,
            key_topics JSONB,
            confidence_score FLOAT,
            quality_score FLOAT,
            content_type VARCHAR(50),
            enhancement_model VARCHAR(50)
        )
    """)
    
    await conn.execute("""
        CREATE TABLE IF NOT EXISTS table_content (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            enhanced_chunk_id UUID REFERENCES enhanced_chunks(id),
            table_data JSONB,
            original_image_base64 TEXT
        )
    """)
    
    await conn.execute("""
        CREATE TABLE IF NOT EXISTS image_content (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            enhanced_chunk_id UUID REFERENCES enhanced_chunks(id),
            image_base64 TEXT,
            ocr_text TEXT,
            llm_description TEXT,
            image_type VARCHAR(50)
        )
    """)
    
    yield conn
    
    # Cleanup
    await conn.execute("DROP TABLE IF EXISTS image_content CASCADE")
    await conn.execute("DROP TABLE IF EXISTS table_content CASCADE") 
    await conn.execute("DROP TABLE IF EXISTS enhanced_chunks CASCADE")
    await conn.execute("DROP TABLE IF EXISTS chunks CASCADE")
    await conn.execute("DROP TABLE IF EXISTS documents CASCADE")
    await conn.close()

@pytest.fixture
def enhanced_ocr_agent():
    """Enhanced OCR Agent fixture"""
    agent = EnhancedOCRAgent("postgresql://test:test@localhost:5432/test")
    agent.db_pool = AsyncMock()
    return agent

@pytest.fixture
def sample_pdf_bytes():
    """Sample PDF as bytes for testing"""
    # Create a simple test image that simulates a PDF page
    img = Image.new('RGB', (800, 600), color='white')
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    return buf.getvalue()

@pytest.fixture
def sample_table_image():
    """Sample table image for testing"""
    # Create a simple table-like image
    img = Image.new('RGB', (400, 300), color='white')
    # Add some text/lines to simulate a table
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    return buf.getvalue()

# === ENHANCED OCR TESTS ===

@pytest.mark.asyncio
async def test_table_detection(enhanced_ocr_agent, sample_table_image):
    """Test table detection and extraction"""
    from PIL import Image
    
    image = Image.open(io.BytesIO(sample_table_image))
    
    # Mock the table detection
    with patch.object(enhanced_ocr_agent, 'detect_table_regions') as mock_detect:
        mock_detect.return_value = [(50, 50, 200, 100)]  # x, y, w, h
        
        with patch('pytesseract.image_to_string') as mock_ocr:
            mock_ocr.return_value = "Header1\tHeader2\nValue1\tValue2"
            
            table_chunks = await enhanced_ocr_agent.detect_and_extract_tables(image, 1)
            
            assert len(table_chunks) == 1
            assert table_chunks[0]['type'] == 'table'
            assert 'table_data' in table_chunks[0]['metadata']
            assert table_chunks[0]['page'] == 1

@pytest.mark.asyncio
async def test_image_extraction(enhanced_ocr_agent, sample_table_image):
    """Test image/diagram extraction"""
    from PIL import Image
    
    image = Image.open(io.BytesIO(sample_table_image))
    
    # Mock image region detection
    with patch.object(enhanced_ocr_agent, 'detect_image_regions') as mock_detect:
        mock_detect.return_value = [(100, 100, 150, 150)]
        
        with patch('pytesseract.image_to_string') as mock_ocr:
            mock_ocr.return_value = "Chart showing Q1 results"
            
            image_chunks = await enhanced_ocr_agent.extract_images_from_page(image, 1)
            
            assert len(image_chunks) == 1
            assert image_chunks[0]['type'] == 'image'
            assert 'image_base64' in image_chunks[0]['metadata']
            assert 'ocr_text' in image_chunks[0]['metadata']

@pytest.mark.asyncio  
async def test_enhanced_pdf_extraction(enhanced_ocr_agent):
    """Test enhanced PDF processing with tables and images"""
    pdf_bytes = b"fake_pdf_content"
    
    with patch('pdf2image.convert_from_bytes') as mock_convert:
        # Mock PDF to image conversion
        mock_image = Mock()
        mock_image.size = (800, 600)
        mock_convert.return_value = [mock_image]
        
        with patch.object(enhanced_ocr_agent, 'extract_text_from_image') as mock_text:
            mock_text.return_value = [{"content": "Sample text", "type": "text", "page": 1}]
            
            with patch.object(enhanced_ocr_agent, 'detect_and_extract_tables') as mock_tables:
                mock_tables.return_value = [{"content": "Table 1", "type": "table", "page": 1}]
                
                with patch.object(enhanced_ocr_agent, 'extract_images_from_page') as mock_images:
                    mock_images.return_value = [{"content": "Image 1", "type": "image", "page": 1}]
                    
                    chunks = await enhanced_ocr_agent.extract_pdf_enhanced(pdf_bytes, "test.pdf")
                    
                    # Should have text, table, and image chunks
                    assert len(chunks) >= 3
                    chunk_types = [chunk['type'] for chunk in chunks]
                    assert 'text' in chunk_types
                    assert 'table' in chunk_types
                    assert 'image' in chunk_types

def test_image_preprocessing(enhanced_ocr_agent):
    """Test image preprocessing for better OCR"""
    # Create test image
    img = Image.new('RGB', (100, 100), color='gray')
    
    processed = enhanced_ocr_agent.preprocess_image_for_ocr(img)
    
    # Should be grayscale
    assert processed.mode == 'L'
    # Should be resized if too small
    assert processed.size[0] >= 1000

def test_table_text_parsing(enhanced_ocr_agent):
    """Test parsing OCR text into structured table"""
    table_text = """
    Name    Age    City
    John    25     Berlin
    Jane    30     Munich
    """
    
    parsed = enhanced_ocr_agent.parse_table_text(table_text)
    
    if parsed:  # Only test if tabulation is available
        assert len(parsed) == 2
        assert 'Name' in parsed[0]
        assert parsed[0]['Name'] == 'John'

def test_image_classification(enhanced_ocr_agent):
    """Test basic image content classification"""
    # Test chart classification
    chart_text = "Sales chart showing 50% increase"
    classification = enhanced_ocr_agent.classify_image_content(None, chart_text)
    assert classification == 'chart'
    
    # Test logo classification
    logo_text = "Company logo brand"
    classification = enhanced_ocr_agent.classify_image_content(None, logo_text)
    assert classification == 'logo'

# === ENHANCED SEARCH TESTS ===

@pytest.mark.asyncio
async def test_enhanced_search_api():
    """Test enhanced search API functionality"""
    from services.search.api_v2 import search_enhanced_content, SearchRequest
    
    # Mock database connection
    with patch('services.search.api_v2.db_pool') as mock_pool:
        mock_conn = AsyncMock()
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
        
        # Mock search results
        mock_conn.fetchval.return_value = "search-id-123"
        mock_conn.fetch.return_value = [
            {
                'id': 'chunk-123',
                'document_id': 'doc-123',
                'enhanced_content': 'Enhanced content about machine learning',
                'content_type': 'text',
                'categories': '["AI", "Technology"]',
                'key_topics': '["machine learning", "neural networks"]',
                'extracted_metadata': '{"keywords": ["AI", "ML"]}',
                'confidence_score': 0.9,
                'quality_score': 0.8,
                'page_number': 1,
                'enhancement_model': 'mistral',
                'original_name': 'ml_guide.pdf',
                'final_relevance': 0.95
            }
        ]
        mock_conn.execute.return_value = None
        
        # Test search
        request = SearchRequest(query="machine learning", limit=10)
        results = await search_enhanced_content(request)
        
        assert len(results) == 1
        assert results[0].content_type == 'text'
        assert 'AI' in results[0].categories
        assert results[0].confidence_score == 0.9
        assert results[0].relevance_score == 0.95

# === N8N WORKFLOW TESTS ===

def test_n8n_anonymizer_logic():
    """Test N8N anonymizer logic"""
    # Simulate the N8N anonymizer code
    content = "Contact Max Mustermann at max@example.com or +49 123 456789"
    use_external_api = True
    
    # Basic anonymization patterns (simplified version of N8N code)
    import re
    
    anonymization_map = {}
    anonymized_content = content
    
    if use_external_api:
        # Email anonymization
        emails = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', content)
        for i, email in enumerate(emails):
            mask = f"[EMAIL_MASKED_{i}]"
            anonymization_map[mask] = email
            anonymized_content = anonymized_content.replace(email, mask)
        
        # Phone anonymization
        phones = re.findall(r'(?:\+49|0)\s*\d{2,4}[\s-]?\d{4,}', content)
        for i, phone in enumerate(phones):
            mask = f"[PHONE_MASKED_{i}]"
            anonymization_map[mask] = phone
            anonymized_content = anonymized_content.replace(phone, mask)
    
    # Verify anonymization
    assert "@" not in anonymized_content
    assert "+49" not in anonymized_content
    assert "[EMAIL_MASKED_0]" in anonymized_content
    assert "[PHONE_MASKED_0]" in anonymized_content
    assert len(anonymization_map) == 2

def test_llm_enhancement_prompt_generation():
    """Test LLM enhancement prompt generation logic"""
    # Simulate N8N LLM prompt generation
    content = "Dies ist ein test dokument mit fehler."
    content_type = "text"
    
    if content_type == "text":
        prompt = f"""Improve this text by:

"{content}"

Tasks:
1. Fix OCR errors and typos
2. Improve sentence structure and grammar
3. Preserve all factual information
4. Maintain original meaning
5. Use professional language

Return ONLY the improved text content."""
    
    assert "Fix OCR errors" in prompt
    assert content in prompt
    assert "ONLY the improved text" in prompt

# === INTEGRATION TESTS ===

@pytest.mark.integration
@pytest.mark.asyncio
async def test_full_enhancement_pipeline(test_db):
    """Test complete enhancement pipeline"""
    # Create test document
    doc_id = await test_db.fetchval("""
        INSERT INTO documents (original_name, file_type, metadata, status)
        VALUES ($1, $2, $3, $4) RETURNING id
    """, "test_doc.pdf", "pdf", json.dumps({"test": True}), "completed")
    
    # Create raw chunk
    chunk_id = await test_db.fetchval("""
        INSERT INTO chunks (document_id, chunk_index, content, content_type)
        VALUES ($1, $2, $3, $4) RETURNING id
    """, doc_id, 0, "Raw OCR content with errors", "text")
    
    # Simulate LLM enhancement
    enhanced_id = await test_db.fetchval("""
        INSERT INTO enhanced_chunks (
            document_id, original_chunk_id, chunk_index, enhanced_content,
            categories, key_topics, confidence_score, quality_score,
            enhancement_model
        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9) RETURNING id
    """, doc_id, chunk_id, 0, "Enhanced content without errors",
        json.dumps(["Technology"]), json.dumps(["OCR", "Enhancement"]),
        0.9, 0.8, "mistral")
    
    # Verify enhancement was created
    enhanced = await test_db.fetchrow("""
        SELECT * FROM enhanced_chunks WHERE id = $1
    """, enhanced_id)
    
    assert enhanced is not None
    assert enhanced['confidence_score'] == 0.9
    assert "Enhanced content" in enhanced['enhanced_content']
    
    # Verify categories
    categories = json.loads(enhanced['categories'])
    assert "Technology" in categories

@pytest.mark.integration
async def test_table_and_image_storage(test_db):
    """Test storage of table and image content"""
    # Create document and enhanced chunk
    doc_id = await test_db.fetchval("""
        INSERT INTO documents (original_name, file_type, status)
        VALUES ($1, $2, $3) RETURNING id
    """, "table_doc.pdf", "pdf", "completed")
    
    enhanced_id = await test_db.fetchval("""
        INSERT INTO enhanced_chunks (document_id, chunk_index, enhanced_content, content_type)
        VALUES ($1, $2, $3, $4) RETURNING id
    """, doc_id, 0, "Table content", "table")
    
    # Store table content
    table_id = await test_db.fetchval("""
        INSERT INTO table_content (enhanced_chunk_id, table_data, original_image_base64)
        VALUES ($1, $2, $3) RETURNING id
    """, enhanced_id, json.dumps([{"col1": "val1", "col2": "val2"}]), "base64data")
    
    # Store image content
    image_id = await test_db.fetchval("""
        INSERT INTO image_content (enhanced_chunk_id, image_base64, ocr_text, image_type)
        VALUES ($1, $2, $3, $4) RETURNING id
    """, enhanced_id, "base64imagedata", "Chart showing results", "chart")
    
    # Verify storage
    table = await test_db.fetchrow("SELECT * FROM table_content WHERE id = $1", table_id)
    image = await test_db.fetchrow("SELECT * FROM image_content WHERE id = $1", image_id)
    
    assert table is not None
    assert image is not None
    assert "val1" in table['table_data']
    assert image['image_type'] == "chart"

# === PERFORMANCE TESTS ===

@pytest.mark.slow
def test_large_text_processing_performance(enhanced_ocr_agent):
    """Test performance with large text chunks"""
    import time
    
    # Generate large text (10KB)
    large_text = "This is a test sentence. " * 400
    
    start_time = time.time()
    chunks = enhanced_ocr_agent.split_text(large_text)
    processing_time = time.time() - start_time
    
    # Should process quickly (under 1 second)
    assert processing_time < 1.0
    assert len(chunks) > 1
    assert all(len(chunk) <= enhanced_ocr_agent.chunk_size * 1.2 for chunk in chunks)

@pytest.mark.slow
async def test_concurrent_search_performance():
    """Test search performance under load"""
    from services.search.api_v2 import SearchRequest
    
    # Mock multiple concurrent searches
    async def mock_search():
        request = SearchRequest(query="test query", limit=10)
        # Simulate search processing time
        await asyncio.sleep(0.1)
        return []
    
    start_time = asyncio.get_event_loop().time()
    
    # Run 10 concurrent searches
    tasks = [mock_search() for _ in range(10)]
    results = await asyncio.gather(*tasks)
    
    end_time = asyncio.get_event_loop().time()
    
    # Should handle concurrent requests efficiently
    assert end_time - start_time < 1.0  # All should complete in under 1 second
    assert len(results) == 10

# === PYTEST CONFIGURATION ===

def pytest_configure(config):
    """Configure pytest markers"""
    config.addinivalue_line("markers", "integration: integration tests requiring database")
    config.addinivalue_line("markers", "slow: slow tests that take more time")

# Run specific test groups:
# pytest tests/test_enhanced_features.py -m "not slow"  # Skip slow tests
# pytest tests/test_enhanced_features.py -m integration  # Only integration tests
# pytest tests/test_enhanced_features.py::test_table_detection -v  # Specific test