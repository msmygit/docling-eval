# Adding Unstructured-OSS Support to Docling-Eval

## Overview

This document provides a detailed step-by-step implementation plan for adding unstructured-oss as a prediction provider to the docling-eval framework. Unstructured-oss is an open-source library for processing and extracting text from various document formats including PDFs and images.

## Prerequisites

- Familiarity with Python and the docling-eval codebase
- Understanding of prediction providers and their interfaces
- Knowledge of unstructured-oss library and its capabilities
- Access to the docling-eval development environment

## Implementation Steps

### Step 1: Add Dependencies

#### 1.1 Update pyproject.toml
Add unstructured-oss as an optional dependency in the `pyproject.toml` file:

```toml
[project.optional-dependencies]
hyperscalers = [
    'azure-ai-documentintelligence (>=1.0.2,<2.0.0)',
    'azure-common (>=1.1.28,<2.0.0)',
    'azure-core (>=1.33.0,<2.0.0)',
    'boto3 (>=1.37.8,<2.0.0)',
    'google-cloud-documentai (>=3.2.0,<4.0.0)',
    'ibm-cos-sdk (>=2.1.40,<3.0.0)',
]
unstructured = [
    'unstructured[pdf,image] (>=0.18.13,<1.0.0)',
]
```

#### 1.2 Install Dependencies
Run the following command to install the new dependency:
```bash
uv add "unstructured[pdf,image]"
```

### Step 2: Update Type Definitions

#### 2.1 Add Provider Type to Enum
In `docling_eval/datamodels/types.py`, add the new provider type to the `PredictionProviderType` enum:

```python
class PredictionProviderType(str, Enum):
    """Types of prediction providers available."""

    DOCLING = "Docling"
    PDF_DOCLING = "PDF_Docling"
    OCR_DOCLING = "OCR_Docling"
    MacOCR_DOCLING = "MacOCR_Docling"
    EasyOCR_DOCLING = "EasyOCR_Docling"

    TABLEFORMER = "TableFormer"
    FILE = "File"
    SMOLDOCLING = "SmolDocling"
    AWS = "AWS"
    AZURE = "Azure"
    GOOGLE = "Google"
    UNSTRUCTURED_OSS = "UnstructuredOSS"  # Add this line
```

### Step 3: Create the Unstructured-OSS Prediction Provider

#### 3.1 Create Provider File
Create a new file `docling_eval/prediction_providers/unstructured_oss_provider.py`:

```python
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set

from docling_core.types.doc import DocItemLabel
from docling_core.types.doc.document import DoclingDocument
from unstructured.partition.auto import partition
from unstructured.documents.elements import (
    Title, 
    NarrativeText, 
    ListItem, 
    Table, 
    Figure,
    PageBreak
)

from docling_eval.datamodels.dataset_record import (
    DatasetRecord,
    DatasetRecordWithPrediction,
)
from docling_eval.datamodels.types import (
    EvaluationModality,
    PredictionFormats,
    PredictionProviderType,
)
from docling_eval.prediction_providers.base_prediction_provider import (
    BasePredictionProvider,
)

_log = logging.getLogger(__name__)


class UnstructuredOSSPredictionProvider(BasePredictionProvider):
    """
    Prediction provider that uses unstructured-oss for document processing.
    
    This provider processes documents using unstructured-oss and converts
    the output to DoclingDocument format for evaluation.
    """

    prediction_provider_type: PredictionProviderType = PredictionProviderType.UNSTRUCTURED_OSS

    prediction_modalities: List[EvaluationModality] = [
        EvaluationModality.LAYOUT,
        EvaluationModality.MARKDOWN_TEXT,
        EvaluationModality.BBOXES_TEXT,
    ]

    def __init__(
        self,
        do_visualization: bool = False,
        ignore_missing_predictions: bool = True,
        true_labels: Optional[Set[DocItemLabel]] = None,
        pred_labels: Optional[Set[DocItemLabel]] = None,
        include_metadata: bool = True,
        strategy: str = "fast",
    ):
        """
        Initialize the Unstructured-OSS prediction provider.

        Args:
            do_visualization: Whether to generate visualizations
            ignore_missing_predictions: Whether to ignore missing predictions
            true_labels: Set of DocItemLabel to use for ground truth visualization
            pred_labels: Set of DocItemLabel to use for prediction visualization
            include_metadata: Whether to include metadata in the output
            strategy: Processing strategy ('fast', 'accurate', 'ocr_only')
        """
        super().__init__(
            do_visualization=do_visualization,
            ignore_missing_predictions=ignore_missing_predictions,
            true_labels=true_labels,
            pred_labels=pred_labels,
        )
        self.include_metadata = include_metadata
        self.strategy = strategy

    @property
    def prediction_format(self) -> PredictionFormats:
        """Get the prediction format."""
        return PredictionFormats.DOCLING_DOCUMENT

    def info(self) -> Dict[str, str]:
        """Get information about the prediction provider."""
        return {
            "provider": "UnstructuredOSS",
            "version": "1.0.0",
            "strategy": self.strategy,
            "include_metadata": str(self.include_metadata),
        }

    def predict(self, record: DatasetRecord) -> DatasetRecordWithPrediction:
        """
        Generate a prediction using unstructured-oss.

        Args:
            record: Input dataset record

        Returns:
            Dataset record with prediction

        Raises:
            RuntimeError: If original document stream is not available
        """
        if record.original is None:
            raise RuntimeError(
                "Stream must be given for unstructured-oss prediction provider to work."
            )

        try:
            # Process the document with unstructured-oss
            elements = self._process_document(record.original)
            
            # Convert unstructured-oss elements to DoclingDocument
            predicted_doc = self._convert_elements_to_docling(elements, record)
            
            # Create the prediction record
            pred_record = self.create_dataset_record_with_prediction(
                record,
                predicted_doc,
                None,  # No original prediction text
            )
            
            return pred_record
            
        except Exception as e:
            _log.error(f"Error processing document {record.doc_id}: {e}")
            if self.ignore_missing_predictions:
                return self.create_dataset_record_with_prediction(
                    record, None, None
                )
            else:
                raise

    def _process_document(self, document_stream) -> List:
        """
        Process document using unstructured-oss.
        
        Args:
            document_stream: Document stream to process
            
        Returns:
            List of unstructured elements
        """
        # Convert stream to temporary file for processing
        with document_stream.open() as f:
            # Use unstructured-oss to partition the document
            elements = partition(
                file=f,
                strategy=self.strategy,
                include_metadata=self.include_metadata,
            )
        
        return elements

    def _convert_elements_to_docling(
        self, elements: List, record: DatasetRecord
    ) -> DoclingDocument:
        """
        Convert unstructured-oss elements to DoclingDocument format.
        
        Args:
            elements: List of unstructured-oss elements
            record: Original dataset record
            
        Returns:
            DoclingDocument with converted elements
        """
        # Initialize DoclingDocument
        doc = DoclingDocument(name=record.doc_id)
        
        # Process elements and convert to Docling format
        for element in elements:
            self._process_element(element, doc)
        
        return doc

    def _process_element(self, element, doc: DoclingDocument):
        """
        Process individual unstructured-oss element and add to DoclingDocument.
        
        Args:
            element: Unstructured-oss element
            doc: DoclingDocument to add elements to
        """
        # Map unstructured-oss element types to Docling labels
        if isinstance(element, Title):
            # Handle title elements
            pass
        elif isinstance(element, NarrativeText):
            # Handle text elements
            pass
        elif isinstance(element, ListItem):
            # Handle list items
            pass
        elif isinstance(element, Table):
            # Handle table elements
            pass
        elif isinstance(element, Figure):
            # Handle figure elements
            pass
        elif isinstance(element, PageBreak):
            # Handle page breaks
            pass
        else:
            # Handle other element types
            pass

        # TODO: Implement detailed conversion logic for each element type
        # This will involve:
        # 1. Extracting text content
        # 2. Converting coordinates if available
        # 3. Mapping to appropriate Docling labels
        # 4. Adding to the appropriate page in the document
```

#### 3.2 Implement Element Conversion Logic
The `_process_element` method needs detailed implementation to convert unstructured-oss elements to Docling format. This involves:

1. **Text Extraction**: Extract text content from elements
2. **Coordinate Mapping**: Convert unstructured-oss coordinates to Docling bounding boxes
3. **Label Mapping**: Map unstructured-oss element types to Docling labels
4. **Page Management**: Handle multi-page documents correctly

#### 3.3 Add Error Handling and Logging
Implement comprehensive error handling for:
- Unsupported file formats
- Processing failures
- Coordinate conversion errors
- Memory issues with large documents

### Step 4: Update CLI Integration

#### 4.1 Add Import to main.py
In `docling_eval/cli/main.py`, add the import for the new provider:

```python
from docling_eval.prediction_providers.unstructured_oss_provider import (
    UnstructuredOSSPredictionProvider,
)
```

#### 4.2 Update Provider Factory Function
In the `get_prediction_provider` function in `docling_eval/cli/main.py`, add the case for the new provider:

```python
elif provider_type == PredictionProviderType.UNSTRUCTURED_OSS:
    return UnstructuredOSSPredictionProvider(
        do_visualization=do_visualization,
        ignore_missing_predictions=True,
        include_metadata=True,
        strategy="fast",  # Can be made configurable via CLI options
    )
```

#### 4.3 Add CLI Options (Optional)
If you want to make unstructured-oss parameters configurable, add new CLI options to the `create_eval` function:

```python
unstructured_strategy: Annotated[
    str,
    typer.Option(
        help="Processing strategy for unstructured-oss (fast, accurate, ocr_only)"
    ),
] = "fast",
unstructured_include_metadata: Annotated[
    bool,
    typer.Option(
        help="Include metadata in unstructured-oss output"
    ),
] = True,
```

### Step 5: Update Provider Registry

#### 5.1 Update Provider Mapping
In `docling_eval/utils/utils.py`, update the provider mapping if it exists:

```python
PROVIDER_MAPPING = {
    PredictionProviderType.DOCLING: DoclingPredictionProvider,
    PredictionProviderType.TABLEFORMER: TableFormerPredictionProvider,
    PredictionProviderType.UNSTRUCTURED_OSS: UnstructuredOSSPredictionProvider,  # Add this
    # ... other providers
}
```

### Step 6: Create Tests

#### 6.1 Create Test File
Create `tests/test_unstructured_oss_provider.py`:

```python
import pytest
from pathlib import Path
from unittest.mock import Mock, patch

from docling_eval.datamodels.types import PredictionProviderType
from docling_eval.prediction_providers.unstructured_oss_provider import (
    UnstructuredOSSPredictionProvider,
)
from docling_eval.datamodels.dataset_record import DatasetRecord


class TestUnstructuredOSSPredictionProvider:
    """Test cases for UnstructuredOSSPredictionProvider."""

    def test_provider_initialization(self):
        """Test provider initialization."""
        provider = UnstructuredOSSPredictionProvider()
        assert provider.prediction_provider_type == PredictionProviderType.UNSTRUCTURED_OSS
        assert provider.prediction_format == PredictionFormats.DOCLING_DOCUMENT

    def test_info_method(self):
        """Test info method returns correct information."""
        provider = UnstructuredOSSPredictionProvider(strategy="accurate")
        info = provider.info()
        assert info["provider"] == "UnstructuredOSS"
        assert info["strategy"] == "accurate"

    @patch('unstructured.partition.auto.partition')
    def test_predict_method(self, mock_partition):
        """Test predict method with mocked unstructured-oss."""
        # Setup mock
        mock_elements = [Mock(), Mock()]
        mock_partition.return_value = mock_elements
        
        # Create provider and test record
        provider = UnstructuredOSSPredictionProvider()
        record = Mock(spec=DatasetRecord)
        record.doc_id = "test_doc"
        record.original = Mock()
        record.original.open.return_value.__enter__ = Mock()
        record.original.open.return_value.__exit__ = Mock()
        
        # Test prediction
        result = provider.predict(record)
        assert result is not None
        mock_partition.assert_called_once()

    def test_predict_without_stream(self):
        """Test predict method raises error when no stream is provided."""
        provider = UnstructuredOSSPredictionProvider()
        record = Mock(spec=DatasetRecord)
        record.original = None
        
        with pytest.raises(RuntimeError, match="Stream must be given"):
            provider.predict(record)

    def test_error_handling(self):
        """Test error handling in predict method."""
        provider = UnstructuredOSSPredictionProvider(ignore_missing_predictions=True)
        record = Mock(spec=DatasetRecord)
        record.doc_id = "test_doc"
        record.original = Mock()
        record.original.open.side_effect = Exception("Processing error")
        
        # Should not raise exception when ignore_missing_predictions=True
        result = provider.predict(record)
        assert result is not None
```

#### 6.2 Add Integration Tests
Create integration tests that use real documents to verify the complete pipeline works correctly.

### Step 7: Update Documentation

#### 7.1 Update README
Add information about the new provider to the main README.md file.

#### 7.2 Create Provider-Specific Documentation
Create documentation explaining how to use the unstructured-oss provider, including:
- Supported file formats
- Configuration options
- Performance characteristics
- Limitations and known issues

### Step 8: Performance Optimization

#### 8.1 Implement Caching
Consider implementing caching for processed documents to improve performance during evaluation.

#### 8.2 Memory Management
Implement proper memory management for large documents, especially for batch processing.

#### 8.3 Parallel Processing
Consider implementing parallel processing for batch operations if unstructured-oss supports it.

### Step 9: Validation and Testing

#### 9.1 Test with Real Documents
Test the implementation with various document types:
- PDFs with different layouts
- Images with text
- Mixed content documents
- Large documents

#### 9.2 Benchmark Performance
Compare performance with other providers:
- Processing speed
- Memory usage
- Accuracy of text extraction
- Quality of layout detection

#### 9.3 Validate Output Format
Ensure the output DoclingDocument format is compatible with existing evaluators:
- Layout evaluator
- Markdown text evaluator
- Bbox text evaluator

### Step 10: Integration Testing

#### 10.1 Test CLI Integration
Test the complete CLI workflow:

```bash
# Create ground truth
docling-eval create-gt --benchmark OmniDocBench --output-dir ./benchmarks/OmniDocBench-gt/

# Create predictions using unstructured-oss
docling-eval create-eval \
  --benchmark OmniDocBench \
  --gt-dir ./benchmarks/OmniDocBench-gt/ \
  --output-dir ./benchmarks/OmniDocBench-unstructured/ \
  --prediction-provider UnstructuredOSS

# Evaluate results
docling-eval evaluate \
  --modality layout \
  --benchmark OmniDocBench \
  --output-dir ./benchmarks/OmniDocBench-unstructured/
```

#### 10.2 Test with Different Modalities
Test the provider with different evaluation modalities:
- Layout evaluation
- Markdown text evaluation
- Bbox text evaluation

## Implementation Notes

### Key Considerations

1. **Coordinate System**: Unstructured-oss may use different coordinate systems than Docling. Ensure proper conversion.

2. **Element Mapping**: Carefully map unstructured-oss element types to Docling labels to ensure accurate evaluation.

3. **Performance**: Unstructured-oss may be slower than other providers. Consider this in the implementation.

4. **Memory Usage**: Large documents may require significant memory. Implement proper memory management.

5. **Error Handling**: Robust error handling is essential for production use.

### Potential Challenges

1. **Format Compatibility**: Ensure unstructured-oss supports all required input formats.

2. **Output Quality**: The quality of unstructured-oss output may vary. Validate against ground truth.

3. **Performance**: Monitor performance and optimize as needed.

4. **Dependencies**: Ensure all unstructured-oss dependencies are properly managed.

### Future Enhancements

1. **Configuration Options**: Add more configuration options for different use cases.

2. **Caching**: Implement caching for improved performance.

3. **Parallel Processing**: Add support for parallel processing of multiple documents.

4. **Custom Models**: Allow integration with custom unstructured-oss models.

## Conclusion

This implementation plan provides a comprehensive roadmap for adding unstructured-oss support to docling-eval. Following these steps will ensure a robust, well-tested, and maintainable implementation that integrates seamlessly with the existing framework.

The key to success is thorough testing and validation, especially ensuring that the output format is compatible with existing evaluators and that the performance meets the requirements of the evaluation framework.
