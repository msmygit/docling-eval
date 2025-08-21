"""
Prediction providers module.

This module contains all prediction providers for the docling-eval framework.
"""

from typing import Dict, Type

from docling_eval.datamodels.types import PredictionProviderType
from docling_eval.prediction_providers.base_prediction_provider import BasePredictionProvider
from docling_eval.prediction_providers.docling_provider import DoclingPredictionProvider
from docling_eval.prediction_providers.file_provider import FilePredictionProvider
from docling_eval.prediction_providers.tableformer_provider import TableFormerPredictionProvider
# Import providers conditionally to avoid dependency issues
try:
    from docling_eval.prediction_providers.aws_prediction_provider import AWSTextractPredictionProvider
except ImportError:
    AWSTextractPredictionProvider = None

try:
    from docling_eval.prediction_providers.azure_prediction_provider import AzureDocIntelligencePredictionProvider
except ImportError:
    AzureDocIntelligencePredictionProvider = None

try:
    from docling_eval.prediction_providers.google_prediction_provider import GoogleDocAIPredictionProvider
except ImportError:
    GoogleDocAIPredictionProvider = None

try:
    from docling_eval.prediction_providers.unstructured_oss_provider import UnstructuredOSSPredictionProvider
except ImportError:
    UnstructuredOSSPredictionProvider = None

# Provider registry mapping provider types to their classes
PROVIDER_REGISTRY: Dict[PredictionProviderType, Type[BasePredictionProvider]] = {
    PredictionProviderType.DOCLING: DoclingPredictionProvider,
    PredictionProviderType.PDF_DOCLING: DoclingPredictionProvider,
    PredictionProviderType.OCR_DOCLING: DoclingPredictionProvider,
    PredictionProviderType.MacOCR_DOCLING: DoclingPredictionProvider,
    PredictionProviderType.EasyOCR_DOCLING: DoclingPredictionProvider,
    PredictionProviderType.SMOLDOCLING: DoclingPredictionProvider,
    PredictionProviderType.TABLEFORMER: TableFormerPredictionProvider,
    PredictionProviderType.FILE: FilePredictionProvider,
}

# Add optional providers if available
if AWSTextractPredictionProvider is not None:
    PROVIDER_REGISTRY[PredictionProviderType.AWS] = AWSTextractPredictionProvider

if AzureDocIntelligencePredictionProvider is not None:
    PROVIDER_REGISTRY[PredictionProviderType.AZURE] = AzureDocIntelligencePredictionProvider

if GoogleDocAIPredictionProvider is not None:
    PROVIDER_REGISTRY[PredictionProviderType.GOOGLE] = GoogleDocAIPredictionProvider

if UnstructuredOSSPredictionProvider is not None:
    PROVIDER_REGISTRY[PredictionProviderType.UNSTRUCTURED_OSS] = UnstructuredOSSPredictionProvider


def get_provider_class(provider_type: PredictionProviderType) -> Type[BasePredictionProvider]:
    """
    Get the provider class for a given provider type.
    
    Args:
        provider_type: The type of prediction provider
        
    Returns:
        The provider class
        
    Raises:
        ValueError: If the provider type is not supported
    """
    if provider_type not in PROVIDER_REGISTRY:
        raise ValueError(f"Unsupported prediction provider: {provider_type}")
    
    return PROVIDER_REGISTRY[provider_type]


def get_supported_provider_types() -> list[PredictionProviderType]:
    """
    Get a list of all supported provider types.
    
    Returns:
        List of supported provider types
    """
    return list(PROVIDER_REGISTRY.keys())


def is_provider_supported(provider_type: PredictionProviderType) -> bool:
    """
    Check if a provider type is supported.
    
    Args:
        provider_type: The provider type to check
        
    Returns:
        True if the provider is supported, False otherwise
    """
    return provider_type in PROVIDER_REGISTRY


__all__ = [
    "BasePredictionProvider",
    "DoclingPredictionProvider",
    "FilePredictionProvider",
    "TableFormerPredictionProvider",
    "AWSTextractPredictionProvider",
    "AzureDocIntelligencePredictionProvider",
    "GoogleDocAIPredictionProvider",
    "UnstructuredOSSPredictionProvider",
    "PROVIDER_REGISTRY",
    "get_provider_class",
    "get_supported_provider_types",
    "is_provider_supported",
]
