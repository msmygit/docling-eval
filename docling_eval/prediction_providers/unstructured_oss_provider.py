import logging
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Set, Any

from docling_core.types.doc.base import BoundingBox, CoordOrigin, Size
from docling_core.types.doc.document import (
    DoclingDocument,
    PageItem,
    ProvenanceItem,
)
from docling_core.types.doc.labels import DocItemLabel
from docling_core.types.io import DocumentStream

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

    def _process_document(self, document_stream: DocumentStream) -> List[Any]:
        """
        Process document using unstructured-oss.
        
        Args:
            document_stream: Document stream to process
            
        Returns:
            List of unstructured elements
        """
        try:
            # Import unstructured here to avoid dependency issues
            from unstructured.partition.auto import partition
            
            # Create a temporary file for processing
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_file:
                # Write the document stream to the temporary file
                with document_stream.open() as f:
                    temp_file.write(f.read())
                temp_file_path = temp_file.name
            
            try:
                # Use unstructured-oss to partition the document
                elements = partition(
                    filename=temp_file_path,
                    strategy=self.strategy,
                    include_metadata=self.include_metadata,
                )
                return elements
            finally:
                # Clean up temporary file
                Path(temp_file_path).unlink(missing_ok=True)
                
        except ImportError:
            raise RuntimeError(
                "unstructured-oss is not installed. Install with: uv add 'unstructured[pdf,image]'"
            )
        except Exception as e:
            _log.error(f"Error processing document with unstructured-oss: {e}")
            raise

    def _convert_elements_to_docling(
        self, elements: List[Any], record: DatasetRecord
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
        
        # Create a single page for all elements (simplified approach)
        page_item = PageItem(
            page_no=1,
            size=Size(width=1000.0, height=1000.0),
        )
        doc.pages[1] = page_item
        
        # Process elements and convert to Docling format
        y_offset = 0
        for element in elements:
            self._process_element(element, doc, y_offset)
            y_offset += 50  # Simple spacing between elements
        
        return doc

    def _create_bounding_box(self, y_offset: float) -> BoundingBox:
        """Create a bounding box with simplified positioning."""
        return BoundingBox(
            l=50.0,
            t=y_offset,
            r=950.0,
            b=y_offset + 40.0,
            coord_origin=CoordOrigin.TOPLEFT,
        )
    
    def _create_provenance_item(self, text_content: str, y_offset: float) -> ProvenanceItem:
        """Create a provenance item for the element."""
        bbox = self._create_bounding_box(y_offset)
        return ProvenanceItem(
            page_no=1,
            bbox=bbox,
            charspan=(0, len(text_content)),
        )
    
    def _add_element_to_document(self, element: Any, doc: DoclingDocument, text_content: str, prov: ProvenanceItem) -> None:
        """Add element to document based on its type."""
        try:
            # Import unstructured elements here to avoid dependency issues
            from unstructured.documents.elements import (
                Title, 
                NarrativeText, 
                ListItem, 
                Table, 
                PageBreak,
                Text
            )
            
            # Element type mapping
            element_handlers = {
                Title: lambda: doc.add_title(text=text_content, prov=prov),
                NarrativeText: lambda: doc.add_text(label=DocItemLabel.PARAGRAPH, text=text_content, prov=prov),
                ListItem: lambda: doc.add_list_item(text=text_content, prov=prov),
                Table: lambda: doc.add_text(label=DocItemLabel.TABLE, text=text_content, prov=prov),
                PageBreak: lambda: None,  # Skip page breaks
                Text: lambda: doc.add_text(label=DocItemLabel.TEXT, text=text_content, prov=prov),
            }
            
            # Find the appropriate handler for the element type
            for element_type, handler in element_handlers.items():
                if isinstance(element, element_type):
                    handler()
                    return
            
            # Default handler for unknown types
            doc.add_text(label=DocItemLabel.TEXT, text=text_content, prov=prov)
            
        except ImportError:
            # If unstructured is not available, use element type name for mapping
            element_type_name = type(element).__name__
            
            if element_type_name == "Title":
                doc.add_title(text=text_content, prov=prov)
            elif element_type_name == "NarrativeText":
                doc.add_text(label=DocItemLabel.PARAGRAPH, text=text_content, prov=prov)
            elif element_type_name == "ListItem":
                doc.add_list_item(text=text_content, prov=prov)
            elif element_type_name == "Table":
                doc.add_text(label=DocItemLabel.TABLE, text=text_content, prov=prov)
            elif element_type_name == "PageBreak":
                # Skip page breaks
                pass
            else:
                # Default to text for unknown types
                doc.add_text(label=DocItemLabel.TEXT, text=text_content, prov=prov)
    
    def _process_element(self, element: Any, doc: DoclingDocument, y_offset: float) -> None:
        """
        Process individual unstructured-oss element and add to DoclingDocument.
        
        Args:
            element: Unstructured-oss element
            doc: DoclingDocument to add elements to
            y_offset: Y coordinate offset for positioning
        """
        try:
            # Extract text content
            text_content = str(element)
            
            # Create provenance item
            prov = self._create_provenance_item(text_content, y_offset)
            
            # Add element to document
            self._add_element_to_document(element, doc, text_content, prov)
            
        except Exception as e:
            _log.warning(f"Error processing element {type(element).__name__}: {e}")
