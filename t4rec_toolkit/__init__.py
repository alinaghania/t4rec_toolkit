"""
T4Rec Toolkit - Bibliothèque d'outils pour les modèles de recommandation T4Rec
"""

# Core components
from .core import (
    BaseTransformer,
    TransformationResult,
    DataValidator,
    ValidationResult,
    T4RecToolkitError,
    DataValidationError,
    TransformationError,
    SchemaError,
    ConfigurationError,
)

# Transformers
from .transformers import (
    SequenceTransformer,
    CategoricalTransformer,
    NumericalTransformer,
)

# Model builders
from .models import (
    ModelRegistry,
    BaseModelBuilder,
    XLNetModelBuilder,
    GPT2ModelBuilder,
    get_available_models,
    create_model,
    registry,
)

# Adapters
from .adapters import T4RecAdapter, DataikuAdapter

__version__ = "0.1.0"

__all__ = [
    # Core
    "BaseTransformer",
    "TransformationResult",
    "DataValidator",
    "ValidationResult",
    "T4RecToolkitError",
    "DataValidationError",
    "TransformationError",
    "SchemaError",
    "ConfigurationError",
    # Transformers
    "SequenceTransformer",
    "CategoricalTransformer",
    "NumericalTransformer",
    # Models
    "ModelRegistry",
    "BaseModelBuilder",
    "XLNetModelBuilder",
    "GPT2ModelBuilder",
    "get_available_models",
    "create_model",
    "registry",
    # Adapters
    "T4RecAdapter",
    "DataikuAdapter",
]
