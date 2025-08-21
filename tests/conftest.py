"""
Common test fixtures and utilities for the docling-eval test suite.

This module provides reusable test components that follow DRY principles
and can be shared across all test files in the project.
"""

import pytest
from unittest.mock import Mock
from pathlib import Path
from typing import Dict, Any

from docling_eval.datamodels.dataset_record import DatasetRecord
from docling_core.types.io import DocumentStream


class TestUtils:
    """Common test utilities that can be reused across test files."""
    
    @staticmethod
    def create_mock_dataset_record(doc_id: str = "test_doc", **kwargs) -> Mock:
        """
        Create a mock dataset record with common attributes.
        
        Args:
            doc_id: Document ID for the record
            **kwargs: Additional attributes to set on the record
            
        Returns:
            Mock dataset record
        """
        record = Mock(spec=DatasetRecord)
        record.doc_id = doc_id
        record.as_record_dict = Mock(return_value={
            "document_id": doc_id,
            "ground_truth_doc": None,
            "original": None,
        })
        
        # Set any additional attributes
        for key, value in kwargs.items():
            setattr(record, key, value)
            
        return record
    
    @staticmethod
    def create_mock_document_stream(content: bytes = b"test content") -> Mock:
        """
        Create a mock document stream.
        
        Args:
            content: Content to return when reading the stream
            
        Returns:
            Mock document stream
        """
        mock_stream = Mock()
        mock_stream.open.return_value.__enter__ = Mock(return_value=Mock(read=Mock(return_value=content)))
        mock_stream.open.return_value.__exit__ = Mock(return_value=None)
        return mock_stream
    
    @staticmethod
    def create_mock_element(element_type: str, text_content: str) -> Mock:
        """
        Create a mock element with specified type and content.
        
        Args:
            element_type: Type name for the element
            text_content: Text content for the element
            
        Returns:
            Mock element
        """
        mock_element = Mock()
        mock_element.__str__ = Mock(return_value=text_content)
        type(mock_element).__name__ = element_type
        return mock_element
    
    @staticmethod
    def create_mock_document() -> Mock:
        """
        Create a mock document for testing.
        
        Returns:
            Mock document
        """
        return Mock()
    
    @staticmethod
    def create_mock_prediction_provider(provider_type: str, **kwargs) -> Mock:
        """
        Create a mock prediction provider.
        
        Args:
            provider_type: Type of prediction provider
            **kwargs: Additional attributes to set
            
        Returns:
            Mock prediction provider
        """
        provider = Mock()
        provider.prediction_provider_type = provider_type
        provider.prediction_format = "doclingdocument"
        
        # Set any additional attributes
        for key, value in kwargs.items():
            setattr(provider, key, value)
            
        return provider


# Global test utilities instance
test_utils = TestUtils()


@pytest.fixture
def mock_dataset_record():
    """Fixture to create a mock dataset record."""
    return test_utils.create_mock_dataset_record


@pytest.fixture
def mock_document_stream():
    """Fixture to create a mock document stream."""
    return test_utils.create_mock_document_stream


@pytest.fixture
def mock_element():
    """Fixture to create a mock element."""
    return test_utils.create_mock_element


@pytest.fixture
def mock_document():
    """Fixture to create a mock document."""
    return test_utils.create_mock_document


@pytest.fixture
def mock_prediction_provider():
    """Fixture to create a mock prediction provider."""
    return test_utils.create_mock_prediction_provider


@pytest.fixture
def temp_test_dir(tmp_path):
    """Fixture to create a temporary test directory."""
    return tmp_path


@pytest.fixture
def sample_pdf_content():
    """Fixture to provide sample PDF content for testing."""
    # This is a minimal PDF content for testing
    return b"%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog\n/Pages 2 0 R\n>>\nendobj\n2 0 obj\n<<\n/Type /Pages\n/Kids [3 0 R]\n/Count 1\n>>\nendobj\n3 0 obj\n<<\n/Type /Page\n/Parent 2 0 R\n/MediaBox [0 0 612 792]\n/Contents 4 0 R\n>>\nendobj\n4 0 obj\n<<\n/Length 44\n>>\nstream\nBT\n/F1 12 Tf\n72 720 Td\n(Test PDF) Tj\nET\nendstream\nendobj\nxref\n0 5\n0000000000 65535 f \n0000000009 00000 n \n0000000058 00000 n \n0000000115 00000 n \n0000000204 00000 n \ntrailer\n<<\n/Size 5\n/Root 1 0 R\n>>\nstartxref\n297\n%%EOF\n"
