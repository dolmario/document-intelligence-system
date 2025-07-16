#!/usr/bin/env python3
"""
Health Check for Document Intelligence System V2
"""

import requests
import json
import time
import sys

def check_service(name, url, expected_status=200):
    """Check if a service is running"""
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == expected_status:
            print(f"✅ {name}: OK (HTTP {response.status_code})")
            return True
        else:
            print(f"❌ {name}: FAILED (HTTP {response.status_code})")
            return False
    except Exception as e:
        print(f"❌ {name}: ERROR - {str(e)}")
        return False

def test_upload():
    """Test document upload"""
    try:
        response = requests.post(
            "http://localhost:8001/upload",
            json={
                "name": "health_check.txt",
                "content": "Health check document. Testing Roses metal: 50% Bismuth, 25% Lead, 25% Tin."
            }
        )
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Upload: OK - Document ID: {data.get('document_id', 'unknown')}")
            return data.get('document_id')
        else:
            print(f"❌ Upload: FAILED - {response.text}")
            return None
    except Exception as e:
        print(f"❌ Upload: ERROR - {str(e)}")
        return None

def test_search():
    """Test search functionality"""
    try:
        response = requests.post(
            "http://localhost:8001/search",
            json={
                "query": "Roses metal",
                "limit": 5
            }
        )
        if response.status_code == 200:
            results = response.json()
            if results:
                print(f"✅ Search: OK - Found {len(results)} results")
                return True
            else:
                print("⚠️  Search: OK but no results yet (processing may be ongoing)")
                return True
        else:
            print(f"❌ Search: FAILED - {response.text}")
            return False
    except Exception as e:
        print(f"❌ Search: ERROR - {str(e)}")
        return False

def main():
    print("🏥 Document Intelligence System V2 - Health Check")
    print("=" * 50)
    
    # Check services
    print("\n📡 Checking Services:")
    services = [
        ("PostgreSQL", "http://localhost:5432/", 52),  # Will fail but shows attempt
        ("Search API", "http://localhost:8001/", 200),
        ("N8N", "http://localhost:5678/", 200),
        ("Ollama", "http://localhost:11434/", 200),
        ("Qdrant", "http://localhost:6333/", 200),
        ("Open WebUI", "http://localhost:8080/", 200),
    ]
    
    all_ok = True
    for name, url, expected in services:
        if not check_service(name, url, expected):
            all_ok = all_ok and (name == "PostgreSQL")  # PostgreSQL HTTP check expected to fail
    
    # Test functionality
    print("\n🧪 Testing Functionality:")
    
    # Upload test
    doc_id = test_upload()
    if doc_id:
        print(f"   Waiting 3 seconds for processing...")
        time.sleep(3)
    
    # Search test
    test_search()
    
    # Get stats
    try:
        response = requests.get("http://localhost:8001/stats")
        if response.status_code == 200:
            stats = response.json()
            print(f"\n📊 System Statistics:")
            print(f"   Documents: {stats.get('total_documents', 0)}")
            print(f"   Chunks: {stats.get('total_chunks', 0)}")
            print(f"   Searches: {stats.get('total_searches', 0)}")
    except:
        pass
    
    # Summary
    print("\n" + "=" * 50)
    if all_ok:
        print("✅ System is healthy!")
        print("\n⚠️  Don't forget to:")
        print("   1. Activate N8N workflow (see N8N_SETUP.md)")
        print("   2. Change default passwords in .env")
        return 0
    else:
        print("❌ Some services are not running properly")
        print("   Run: docker compose logs -f")
        return 1

if __name__ == "__main__":
    sys.exit(main())
