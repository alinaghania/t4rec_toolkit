from typing import Dict, Type, List, Any, Optional
import logging

from ..core.exceptions import ConfigurationError
from .base_builder import BaseModelBuilder

logger = logging.getLogger(__name__)


class ModelRegistry:
    """
    Registry pour l'enregistrement et la création de modèles.
    
    Cette classe maintient un mapping des noms d'architectures
    vers leurs builders correspondants, permettant une création
    dynamique et extensible des modèles.
    """
    
    def __init__(self):
        """Initialise le registry vide."""
        self._builders: Dict[str, Type[BaseModelBuilder]] = {}
        self._metadata: Dict[str, Dict[str, Any]] = {}
    
    def register(self, 
                 name: str, 
                 builder_class: Type[BaseModelBuilder],
                 metadata: Optional[Dict[str, Any]] = None):
        """
        Enregistre une nouvelle architecture.
        
        Args:
            name: Nom de l'architecture (ex: 'xlnet', 'gpt2')
            builder_class: Classe du builder
            metadata: Métadonnées optionnelles sur l'architecture
            
        Raises:
            ValueError: Si le nom est déjà enregistré
        """
        if name in self._builders:
            logger.warning(f"Architecture '{name}' déjà enregistrée - écrasement")
        
        if not issubclass(builder_class, BaseModelBuilder):
            raise ValueError(
                f"Builder class doit hériter de BaseModelBuilder, reçu: {builder_class}"
            )
        
        self._builders[name] = builder_class
        self._metadata[name] = metadata or {}
        
        logger.info(f"Architecture '{name}' enregistrée avec succès")
    
    def unregister(self, name: str):
        """
        Désenregistre une architecture.
        
        Args:
            name: Nom de l'architecture à supprimer
        """
        if name not in self._builders:
            logger.warning(f"Architecture '{name}' non trouvée pour désenregistrement")
            return
        
        del self._builders[name]
        del self._metadata[name]
        logger.info(f"Architecture '{name}' désenregistrée")
    
    def create_model(self, 
                    architecture: str, 
                    schema: Dict[str, Any],
                    **config) -> Any:
        """
        Crée un modèle selon l'architecture spécifiée.
        
        Args:
            architecture: Nom de l'architecture
            schema: Schéma T4Rec des données
            **config: Configuration spécifique au modèle
            
        Returns:
            Modèle T4Rec configuré
            
        Raises:
            ConfigurationError: Si l'architecture n'est pas trouvée
        """
        if architecture not in self._builders:
            available = list(self._builders.keys())
            raise ConfigurationError(
                f"Architecture '{architecture}' non trouvée",
                config_key="architecture",
                valid_values=available
            )
        
        builder_class = self._builders[architecture]
        builder = builder_class()
        
        logger.info(f"Création du modèle {architecture} avec config: {config}")
        
        try:
            model = builder.build_model(schema, **config)
            logger.info(f"Modèle {architecture} créé avec succès")
            return model
        except Exception as e:
            logger.error(f"Erreur lors de la création du modèle {architecture}: {e}")
            raise ConfigurationError(
                f"Échec de création du modèle {architecture}: {str(e)}",
                config_key="architecture"
            )
    
    def get_available_models(self) -> List[str]:
        """
        Retourne la liste des architectures disponibles.
        
        Returns:
            Liste des noms d'architectures
        """
        return list(self._builders.keys())
    
    def get_model_metadata(self, architecture: str) -> Dict[str, Any]:
        """
        Retourne les métadonnées d'une architecture.
        
        Args:
            architecture: Nom de l'architecture
            
        Returns:
            Dictionnaire des métadonnées
        """
        return self._metadata.get(architecture, {})
    
    def get_builder_class(self, architecture: str) -> Type[BaseModelBuilder]:
        """
        Retourne la classe builder pour une architecture.
        
        Args:
            architecture: Nom de l'architecture
            
        Returns:
            Classe du builder
            
        Raises:
            ConfigurationError: Si l'architecture n'existe pas
        """
        if architecture not in self._builders:
            raise ConfigurationError(
                f"Architecture '{architecture}' non trouvée",
                config_key="architecture",
                valid_values=list(self._builders.keys())
            )
        
        return self._builders[architecture]
    
    def is_registered(self, architecture: str) -> bool:
        """
        Vérifie si une architecture est enregistrée.
        
        Args:
            architecture: Nom de l'architecture
            
        Returns:
            True si l'architecture est disponible
        """
        return architecture in self._builders
    
    def __len__(self) -> int:
        """Retourne le nombre d'architectures enregistrées."""
        return len(self._builders)
    
    def __contains__(self, architecture: str) -> bool:
        """Support pour l'opérateur 'in'."""
        return self.is_registered(architecture)
    
    def __repr__(self) -> str:
        """Représentation string du registry."""
        models = ", ".join(self._builders.keys())
        return f"ModelRegistry({len(self._builders)} models: {models})"


# Instance globale du registry
_global_registry = ModelRegistry()

def get_registry() -> ModelRegistry:
    """
    Retourne l'instance globale du registry.
    
    Returns:
        Registry global
    """
    return _global_registry

def register_model(name: str, 
                  builder_class: Type[BaseModelBuilder],
                  metadata: Optional[Dict[str, Any]] = None):
    """
    Fonction utilitaire pour enregistrer un modèle dans le registry global.
    
    Args:
        name: Nom de l'architecture
        builder_class: Classe du builder
        metadata: Métadonnées optionnelles
    """
    _global_registry.register(name, builder_class, metadata)

def create_model(architecture: str, 
                schema: Dict[str, Any],
                **config) -> Any:
    """
    Fonction utilitaire pour créer un modèle via le registry global.
    
    Args:
        architecture: Nom de l'architecture
        schema: Schéma T4Rec
        **config: Configuration du modèle
        
    Returns:
        Modèle T4Rec configuré
    """
    return _global_registry.create_model(architecture, schema, **config)

def get_available_models() -> List[str]:
    """
    Fonction utilitaire pour lister les modèles disponibles.
    
    Returns:
        Liste des architectures disponibles
    """
    return _global_registry.get_available_models()


class ModelRegistryDecorator:
    """
    Décorateur pour enregistrer automatiquement les builders.
    
    Usage:
        @model_registry.register_builder("bert")
        class BertModelBuilder(BaseModelBuilder):
            pass
    """
    
    def __init__(self, registry: ModelRegistry):
        """
        Initialise le décorateur.
        
        Args:
            registry: Instance du registry à utiliser
        """
        self.registry = registry
    
    def register_builder(self, 
                        name: str, 
                        metadata: Optional[Dict[str, Any]] = None):
        """
        Décorateur pour l'enregistrement automatique.
        
        Args:
            name: Nom de l'architecture
            metadata: Métadonnées optionnelles
            
        Returns:
            Décorateur de classe
        """
        def decorator(builder_class: Type[BaseModelBuilder]):
            self.registry.register(name, builder_class, metadata)
            return builder_class
        
        return decorator

# Instance du décorateur pour le registry global
model_registry = ModelRegistryDecorator(_global_registry)