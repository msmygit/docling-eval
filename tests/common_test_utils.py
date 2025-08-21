import pytest
from unittest.mock import Mock
from pathlib import Path
from typing import Dict, Any

from docling_eval.datamodels.dataset_record import DatasetRecord
from docling_core.types.io import DocumentStream


class TestUtils:
    @staticmethod
    def create_mock_dataset_record(doc_id: str = "test_doc", **kwargs) -> Mock:
        record = Mock(spec=DatasetRecord)
        record.doc_id = doc_id
        record.as_record_dict = Mock(return_value={
            "document_id": doc_id,
            "document_filepath": str(None),
            "document_filehash": None,
            "GroundTruthDocument": "{}",  # JSON string representation
            "ground_truth_segmented_pages": "{}",
            "BinaryDocument": None,
            "GroundTruthPageImages": [],
            "GroundTruthPictures": [],
            "mime_type": "application/pdf",
            "modalities": [],
        })
        for key, value in kwargs.items():
            setattr(record, key, value)
        return record
    
    @staticmethod
    def create_mock_document_stream(content: bytes = b"test content") -> Mock:
        mock_stream = Mock()
        # Create a mock BytesIO stream
        mock_bytesio = Mock()
        mock_bytesio.read.return_value = content
        mock_stream.stream = mock_bytesio
        # Also set the name attribute
        mock_stream.name = "test.pdf"
        return mock_stream
    
    @staticmethod
    def create_mock_element(element_type: str, text_content: str) -> Mock:
        mock_element = Mock()
        mock_element.__str__ = Mock(return_value=text_content)
        # Set the type name for isinstance checks
        type(mock_element).__name__ = element_type
        # Create a proper type for isinstance checks
        element_class = type(element_type, (), {})
        mock_element.__class__ = element_class
        return mock_element
    
    @staticmethod
    def create_mock_document() -> Mock:
        return Mock()

test_utils = TestUtils()

@pytest.fixture
def mock_dataset_record():
    return test_utils.create_mock_dataset_record

@pytest.fixture
def mock_document_stream():
    return test_utils.create_mock_document_stream

@pytest.fixture
def mock_element():
    return test_utils.create_mock_element

@pytest.fixture
def mock_document():
    return test_utils.create_mock_document
