# models/base_builder.py
"""
Interface de base pour les builders de modèles.

Ce module définit l'interface commune que tous les builders
de modèles doivent implémenter pour assurer la cohérence
et la compatibilité avec le system de registry.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Tuple
import logging

logger = logging.getLogger(__name__)


class BaseModelBuilder(ABC):
    """
    Classe de base abstraite pour tous les builders de modèles.
    
    Cette classe définit l'interface commune que tous les builders
    doivent implémenter pour créer des modèles T4Rec compatibles.
    """
    
    def __init__(self):
        """Initialise le builder."""
        self.default_config = self.get_default_config()
        self.required_params = self.get_required_parameters()
    
    @abstractmethod
    def get_default_config(self) -> Dict[str, Any]:
        """
        Retourne la configuration par défaut pour cette architecture.
        
        Returns:
            Dictionnaire de configuration par défaut
        """
        pass
    
    @abstractmethod
    def get_required_parameters(self) -> List[str]:
        """
        Retourne la liste des paramètres obligatoires.
        
        Returns:
            Liste des noms de paramètres requis
        """
        pass
    
    @abstractmethod
    def build_transformer_config(self, **config) -> Any:
        """
        Construit la configuration du transformer.
        
        Args:
            **config: Paramètres de configuration
            
        Returns:
            Configuration T4Rec du transformer
        """
        pass
    
    @abstractmethod
    def build_model(self, 
                   schema: Dict[str, Any], 
                   **config) -> Any:
        """
        Construit le modèle complet T4Rec.
        
        Args:
            schema: Schéma T4Rec des données
            **config: Configuration du modèle
            
        Returns:
            Modèle T4Rec configuré et prêt pour l'entraînement
        """
        pass
    
    def validate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Valide et complète la configuration.
        
        Args:
            config: Configuration à valider
            
        Returns:
            Configuration validée et complétée
            
        Raises:
            ValueError: Si des paramètres requis manquent
        """
        # Fusionner avec la config par défaut
        merged_config = self.default_config.copy()
        merged_config.update(config)
        
        # Vérifier les paramètres obligatoires
        missing_params = []
        for param in self.required_params:
            if param not in merged_config or merged_config[param] is None:
                missing_params.append(param)
        
        if missing_params:
            raise ValueError(
                f"Paramètres requis manquants pour {self.__class__.__name__}: {missing_params}"
            )
        
        # Validation spécifique
        validated_config = self._validate_specific_config(merged_config)
        
        return validated_config
    
    def _validate_specific_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validation spécifique à chaque architecture.
        
        Args:
            config: Configuration à valider
            
        Returns:
            Configuration validée
        """
        # Par défaut, pas de validation supplémentaire
        return config
    
    def extract_vocab_sizes(self, schema: Dict[str, Any]) -> Dict[str, int]:
        """
        Extrait les tailles de vocabulaire du schéma.
        
        Args:
            schema: Schéma T4Rec
            
        Returns:
            Dictionnaire des tailles de vocabulaire par feature
        """
        vocab_sizes = {}
        
        if 'feature_specs' in schema:
            for spec in schema['feature_specs']:
                feature_name = spec.get('name')
                vocab_size = spec.get('vocab_size')
                
                if feature_name and vocab_size:
                    vocab_sizes[feature_name] = vocab_size
        
        return vocab_sizes
    
    def calculate_total_vocab_size(self, schema: Dict[str, Any]) -> int:
        """
        Calcule la taille totale du vocabulaire.
        
        Args:
            schema: Schéma T4Rec
            
        Returns:
            Taille maximale du vocabulaire
        """
        vocab_sizes = self.extract_vocab_sizes(schema)
        return max(vocab_sizes.values()) if vocab_sizes else 1000
    
    def get_sequence_length(self, schema: Dict[str, Any]) -> int:
        """
        Extrait la longueur de séquence du schéma.
        
        Args:
            schema: Schéma T4Rec
            
        Returns:
            Longueur de séquence
        """
        return schema.get('sequence_length', 20)
    
    def build_input_module(self, 
                          schema: Dict[str, Any], 
                          d_model: int,
                          max_sequence_length: int,
                          masking: str = "mlm") -> Any:
        """
        Construit le module d'entrée TabularSequenceFeatures.
        
        Args:
            schema: Schéma T4Rec
            d_model: Dimension du modèle
            max_sequence_length: Longueur maximale des séquences
            masking: Type de masking ('mlm', 'clm', 'plm')
            
        Returns:
            Module d'entrée T4Rec
        """
        try:
            import transformers4rec.torch as tr
            
            # Créer un schéma T4Rec compatible
            t4rec_schema = self._convert_to_t4rec_schema(schema)
            
            input_module = tr.TabularSequenceFeatures.from_schema(
                t4rec_schema,
                max_sequence_length=max_sequence_length,
                continuous_projection=d_model,
                aggregation="concat",
                masking=masking
            )
            
            return input_module
            
        except ImportError as e:
            raise ImportError(
                "transformers4rec non disponible. "
                "Installez avec: pip install transformers4rec==23.04.00"
            ) from e
    
    def _convert_to_t4rec_schema(self, schema: Dict[str, Any]) -> Any:
        """
        Convertit notre schéma vers un schéma T4Rec natif.
        
        Args:
            schema: Notre schéma interne
            
        Returns:
            Schéma T4Rec natif
        """
        try:
            import transformers4rec.torch as tr
            
            # Pour l'instant, créer un schéma simple
            # Dans une implémentation complète, il faudrait mapper
            # nos feature_specs vers le format T4Rec
            
            # Schéma basique pour commencer
            t4rec_schema = tr.Schema()
            
            return t4rec_schema
            
        except ImportError:
            # Fallback si T4Rec n'est pas disponible
            return schema
    
    def get_architecture_info(self) -> Dict[str, Any]:
        """
        Retourne les informations sur l'architecture.
        
        Returns:
            Dictionnaire avec les informations de l'architecture
        """
        return {
            'name': self.__class__.__name__.replace('ModelBuilder', '').lower(),
            'default_config': self.default_config,
            'required_params': self.required_params,
            'description': self.__doc__ or "Pas de description disponible"
        }
    
    def __repr__(self) -> str:
        """Représentation string du builder."""
        arch_name = self.__class__.__name__.replace('ModelBuilder', '')
        return f"{arch_name}Builder()"