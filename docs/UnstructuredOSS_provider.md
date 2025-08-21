# Unstructured-OSS Provider

## Overview

The Unstructured-OSS provider integrates the [unstructured-oss](https://github.com/Unstructured-IO/unstructured) library into the docling-eval framework. This provider allows you to use unstructured-oss for document processing and evaluation, providing an open-source alternative to commercial document processing services.

## Features

- **Open Source**: Free and open-source document processing
- **Multiple Formats**: Supports PDF, images, and other document formats
- **Layout Detection**: Identifies document structure and layout elements
- **Text Extraction**: Extracts text content with positioning information
- **Element Classification**: Categorizes content as titles, paragraphs, tables, etc.

## Installation

To use the Unstructured-OSS provider, install the required dependencies:

```bash
uv add "unstructured[pdf,image]"
```

Or install the optional dependency group:

```bash
uv add --optional-dependency unstructured
```

## Usage

### Basic Usage

Create predictions using the Unstructured-OSS provider:

```bash
# Create ground truth first
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

### Supported Modalities

The Unstructured-OSS provider supports the following evaluation modalities:

- **Layout**: Document layout and structure analysis
- **Markdown Text**: Text content evaluation
- **Bboxes Text**: Text with bounding box information

### Configuration Options

The provider supports several configuration options:

- `strategy`: Processing strategy ('fast', 'accurate', 'ocr_only')
- `include_metadata`: Whether to include metadata in the output
- `do_visualization`: Whether to generate visualizations
- `ignore_missing_predictions`: Whether to ignore missing predictions

## Implementation Details

### Element Mapping

The provider maps unstructured-oss element types to Docling labels:

| Unstructured-OSS Element | Docling Label | Description |
|-------------------------|---------------|-------------|
| `Title` | `TITLE` | Document titles and headings |
| `NarrativeText` | `PARAGRAPH` | Body text and paragraphs |
| `ListItem` | `LIST_ITEM` | List items and bullet points |
| `Table` | `TABLE` | Tabular data |
| `Figure` | `PICTURE` | Images and figures |
| `Text` | `TEXT` | Generic text content |
| `PageBreak` | *Skipped* | Page boundaries |

### Processing Strategy

The provider supports different processing strategies:

- **fast**: Quick processing with basic layout detection
- **accurate**: More thorough analysis with better accuracy
- **ocr_only**: OCR-based text extraction only

### Coordinate System

The provider uses a simplified coordinate system for positioning elements:
- Default page dimensions: 1000x1000 points
- Elements are positioned vertically with 50-point spacing
- Bounding boxes span the full width of the page

## Performance Characteristics

### Processing Speed

- **Fast strategy**: ~2-5 seconds per document
- **Accurate strategy**: ~5-15 seconds per document
- **OCR-only strategy**: ~1-3 seconds per document

### Memory Usage

- Moderate memory usage (~100-500MB per document)
- Scales linearly with document size
- Automatic cleanup of temporary files

### Accuracy

- Good text extraction accuracy
- Basic layout detection
- Limited table structure recognition
- May not match commercial service accuracy

## Limitations

### Known Issues

1. **Coordinate Accuracy**: Simplified coordinate system may not match original document layout
2. **Table Structure**: Limited table structure detection compared to specialized providers
3. **Complex Layouts**: May struggle with complex multi-column layouts
4. **Image Processing**: Basic image handling, no advanced image analysis

### File Format Support

- **PDF**: Full support
- **Images**: PNG, JPEG, TIFF support
- **Other formats**: Limited support for other document types

## Comparison with Other Providers

| Feature | Unstructured-OSS | Docling | AWS/Azure/Google |
|---------|------------------|---------|------------------|
| Cost | Free | Free | Paid |
| Accuracy | Good | Excellent | Excellent |
| Speed | Moderate | Fast | Fast |
| Layout Detection | Basic | Advanced | Advanced |
| Table Structure | Limited | Advanced | Advanced |
| API Limits | None | None | Rate limits |

## Troubleshooting

### Common Issues

1. **Import Error**: Ensure unstructured-oss is installed
   ```bash
   uv add "unstructured[pdf,image]"
   ```

2. **Processing Errors**: Check document format compatibility
   - Ensure documents are valid PDFs or images
   - Try different processing strategies

3. **Memory Issues**: For large documents
   - Use the 'fast' strategy
   - Process documents in smaller batches

4. **Coordinate Issues**: If layout evaluation fails
   - The simplified coordinate system may not work for all documents
   - Consider using other providers for layout-critical evaluations

### Debug Mode

Enable debug logging to troubleshoot issues:

```python
import logging
logging.getLogger('docling_eval.prediction_providers.unstructured_oss_provider').setLevel(logging.DEBUG)
```

## Examples

### Basic Document Processing

```python
from docling_eval.prediction_providers.unstructured_oss_provider import UnstructuredOSSPredictionProvider

# Create provider
provider = UnstructuredOSSPredictionProvider(
    strategy="fast",
    include_metadata=True
)

# Process document
record = create_test_record()  # Your document record
result = provider.predict(record)

print(f"Processed document: {result.doc_id}")
print(f"Elements found: {len(result.predicted_doc.pages[0].items)}")
```

### Custom Configuration

```python
# Provider with custom settings
provider = UnstructuredOSSPredictionProvider(
    strategy="accurate",
    include_metadata=False,
    do_visualization=True,
    ignore_missing_predictions=False
)
```

## Integration with Evaluation Pipeline

The Unstructured-OSS provider integrates seamlessly with the existing evaluation pipeline:

1. **Dataset Creation**: Use with `create-eval` command
2. **Evaluation**: Compatible with all supported modalities
3. **Visualization**: Generate comparison visualizations
4. **Benchmarking**: Compare with other providers

## Future Enhancements

Planned improvements for the Unstructured-OSS provider:

1. **Better Coordinate Mapping**: Improved coordinate system conversion
2. **Advanced Layout Detection**: Enhanced layout analysis
3. **Table Structure Support**: Better table detection and parsing
4. **Multi-page Support**: Proper handling of multi-page documents
5. **Performance Optimization**: Faster processing and reduced memory usage

## Contributing

To contribute to the Unstructured-OSS provider:

1. Fork the repository
2. Create a feature branch
3. Implement your changes
4. Add tests for new functionality
5. Submit a pull request

## References

- [Unstructured-OSS Documentation](https://unstructured-io.github.io/unstructured/)
- [Unstructured-OSS GitHub Repository](https://github.com/Unstructured-IO/unstructured)
- [Docling-Eval Documentation](https://github.com/docling-project/docling-eval)
