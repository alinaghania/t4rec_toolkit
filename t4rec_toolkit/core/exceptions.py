# core/exceptions.py
"""
Exceptions personnalisées pour le toolkit T4Rec.

Ce module définit toutes les exceptions spécifiques au domaine
métier de la transformation de données pour les modèles de
recommandation.
"""


class T4RecToolkitError(Exception):
    """Exception de base pour tous les erreurs du toolkit."""
    
    def __init__(self, message: str, details: dict = None):
        """
        Initialise l'exception avec un message et des détails optionnels.
        
        Args:
            message: Message d'erreur principal
            details: Détails supplémentaires sur l'erreur
        """
        super().__init__(message)
        self.message = message
        self.details = details or {}
    
    def __str__(self) -> str:
        """Représentation string de l'erreur."""
        base_msg = self.message
        if self.details:
            details_str = ", ".join(f"{k}={v}" for k, v in self.details.items())
            return f"{base_msg} | Détails: {details_str}"
        return base_msg


class DataValidationError(T4RecToolkitError):
    """Exception levée lors de la validation des données."""
    
    def __init__(self, message: str, column: str = None, validation_type: str = None):
        """
        Initialise l'erreur de validation.
        
        Args:
            message: Message d'erreur
            column: Nom de la colonne problématique (optionnel)
            validation_type: Type de validation qui a échoué (optionnel)
        """
        details = {}
        if column:
            details['column'] = column
        if validation_type:
            details['validation_type'] = validation_type
            
        super().__init__(message, details)
        self.column = column
        self.validation_type = validation_type


class TransformationError(T4RecToolkitError):
    """Exception levée lors de la transformation des données."""
    
    def __init__(self, message: str, transformer_name: str = None, step: str = None):
        """
        Initialise l'erreur de transformation.
        
        Args:
            message: Message d'erreur
            transformer_name: Nom du transformer qui a échoué (optionnel)
            step: Étape de transformation problématique (optionnel)
        """
        details = {}
        if transformer_name:
            details['transformer'] = transformer_name
        if step:
            details['step'] = step
            
        super().__init__(message, details)
        self.transformer_name = transformer_name
        self.step = step


class SchemaError(T4RecToolkitError):
    """Exception levée lors de problèmes de schéma."""
    
    def __init__(self, message: str, schema_field: str = None, expected_type: str = None):
        """
        Initialise l'erreur de schéma.
        
        Args:
            message: Message d'erreur
            schema_field: Champ du schéma problématique (optionnel)
            expected_type: Type attendu (optionnel)
        """
        details = {}
        if schema_field:
            details['field'] = schema_field
        if expected_type:
            details['expected_type'] = expected_type
            
        super().__init__(message, details)
        self.schema_field = schema_field
        self.expected_type = expected_type


class ConfigurationError(T4RecToolkitError):
    """Exception levée lors de problèmes de configuration."""
    
    def __init__(self, message: str, config_key: str = None, valid_values: list = None):
        """
        Initialise l'erreur de configuration.
        
        Args:
            message: Message d'erreur
            config_key: Clé de configuration problématique (optionnel)
            valid_values: Valeurs valides pour cette configuration (optionnel)
        """
        details = {}
        if config_key:
            details['config_key'] = config_key
        if valid_values:
            details['valid_values'] = valid_values
            
        super().__init__(message, details)
        self.config_key = config_key
        self.valid_values = valid_values