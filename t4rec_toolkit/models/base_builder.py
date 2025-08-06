# models/base_builder.py
"""
Interface de base pour les builders de modèles.

Ce module définit l'interface commune que tous les builders
de modèles doivent implémenter pour assurer la cohérence
et la compatibilité avec le system de registry.

Version corrigée pour Transformers4Rec 23.04.00 avec résolution
du problème input_module = None.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Tuple
import logging

logger = logging.getLogger(__name__)


class BaseModelBuilder(ABC):
    """
    Classe de base abstraite pour tous les builders de modèles.

    Cette classe définit l'interface commune que tous les builders
    doivent implémenter pour créer des modèles T4Rec compatibles
    avec la version 23.04.00.
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
    def build_model(self, schema: Dict[str, Any], **config) -> Any:
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

        if "feature_specs" in schema:
            for spec in schema["feature_specs"]:
                feature_name = spec.get("name")
                vocab_size = spec.get("vocab_size")

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
        return schema.get("sequence_length", 20)

    def build_input_module(
        self,
        schema: Dict[str, Any],
        d_model: int,
        max_sequence_length: int,
        masking: str = "mlm",
    ) -> Any:
        """
        Construit le module d'entrée TabularSequenceFeatures.
        
        Version corrigée pour Transformers4Rec 23.04.00 avec
        meilleure gestion d'erreurs et approches alternatives.

        Args:
            schema: Schéma T4Rec
            d_model: Dimension du modèle
            max_sequence_length: Longueur maximale des séquences
            masking: Type de masking ('mlm', 'clm', 'plm')

        Returns:
            Module d'entrée T4Rec ou lève une exception
        """
        logger.info(f"Création du module d'entrée avec {len(schema.get('feature_specs', []))} features")
        
        try:
            import transformers4rec.torch as tr

            # Approche 1: Utiliser merlin.schema si disponible
            input_module = self._build_input_with_merlin_schema(
                schema, d_model, max_sequence_length, masking
            )
            
            if input_module is not None:
                logger.info("Module d'entrée créé avec succès via merlin.schema")
                return input_module

            # Approche 2: Utiliser une approche directe
            input_module = self._build_input_direct_approach(
                schema, d_model, max_sequence_length, masking
            )
            
            if input_module is not None:
                logger.info("Module d'entrée créé avec succès via approche directe")
                return input_module

            # Si toutes les approches échouent
            raise ValueError("Impossible de créer le module d'entrée avec toutes les approches")

        except ImportError as e:
            logger.error(f"transformers4rec non disponible: {e}")
            raise ImportError(
                "transformers4rec non disponible. "
                "Installez avec: pip install transformers4rec==23.04.00"
            ) from e

    def _build_input_with_merlin_schema(
        self,
        schema: Dict[str, Any],
        d_model: int,
        max_sequence_length: int,
        masking: str
    ) -> Any:
        """
        Crée le module d'entrée en utilisant merlin.schema.

        Args:
            schema: Schéma T4Rec
            d_model: Dimension du modèle
            max_sequence_length: Longueur de séquence
            masking: Type de masking

        Returns:
            Module d'entrée ou None si échec
        """
        try:
            import transformers4rec.torch as tr
            
            # Convertir le schéma vers merlin.schema
            merlin_schema = self._convert_to_merlin_schema(schema)
            
            if merlin_schema is None:
                logger.warning("Impossible de convertir vers merlin.schema")
                return None

            # Créer le module avec TabularSequenceFeatures.from_schema
            input_module = tr.TabularSequenceFeatures.from_schema(
                schema=merlin_schema,
                max_sequence_length=max_sequence_length,
                continuous_projection=d_model,
                aggregation="concat",
                masking=masking,
                automatic_build=True
            )

            # Vérifier que le module a été créé correctement
            if input_module is None:
                logger.warning("TabularSequenceFeatures.from_schema a retourné None")
                return None

            # Vérifier que le module a l'attribut masking
            if not hasattr(input_module, 'masking'):
                logger.warning("Le module créé n'a pas d'attribut masking")
                # Ajouter l'attribut masking manuellement si nécessaire
                input_module.masking = masking

            return input_module

        except Exception as e:
            logger.warning(f"Échec de création avec merlin.schema: {e}")
            return None

    def _build_input_direct_approach(
        self,
        schema: Dict[str, Any],
        d_model: int,
        max_sequence_length: int,
        masking: str
    ) -> Any:
        """
        Crée le module d'entrée avec une approche directe.

        Args:
            schema: Schéma T4Rec
            d_model: Dimension du modèle
            max_sequence_length: Longueur de séquence
            masking: Type de masking

        Returns:
            Module d'entrée ou None si échec
        """
        try:
            import transformers4rec.torch as tr

            # Créer manuellement les modules pour les features
            categorical_features = {}
            continuous_features = []

            for spec in schema.get("feature_specs", []):
                feature_name = spec["name"]
                
                if spec.get("is_continuous", False):
                    continuous_features.append(feature_name)
                else:
                    # Feature catégorielle
                    vocab_size = spec.get("vocab_size", 100)
                    categorical_features[feature_name] = vocab_size

            # Créer les modules appropriés
            modules = []

            # Module pour features catégorielles
            if categorical_features:
                from transformers4rec.torch.features.embedding import EmbeddingFeatures, FeatureConfig
                
                feature_configs = {}
                for name, vocab_size in categorical_features.items():
                    feature_configs[name] = FeatureConfig(
                        table_config=EmbeddingTableConfig(
                            vocabulary_size=vocab_size,
                            dim=d_model // 4  # Dimension réduite pour les embeddings
                        )
                    )
                
                categorical_module = EmbeddingFeatures(feature_configs)
                modules.append(categorical_module)

            # Module pour features continues
            if continuous_features:
                from transformers4rec.torch.features.continuous import ContinuousFeatures
                continuous_module = ContinuousFeatures(
                    features=continuous_features,
                    projection=d_model // 4
                )
                modules.append(continuous_module)

            # Combiner les modules si nécessaire
            if len(modules) == 1:
                input_module = modules[0]
            elif len(modules) > 1:
                # Utiliser SequentialBlock pour combiner les modules
                input_module = tr.SequentialBlock(*modules)
            else:
                logger.error("Aucun module créé")
                return None

            # Ajouter l'attribut masking
            input_module.masking = masking

            return input_module

        except Exception as e:
            logger.warning(f"Échec de l'approche directe: {e}")
            return None

    def _convert_to_merlin_schema(self, schema: Dict[str, Any]) -> Any:
        """
        Convertit notre schéma vers un schéma merlin.schema.Schema.

        Args:
            schema: Notre schéma interne

        Returns:
            merlin.schema.Schema ou None si échec
        """
        try:
            from merlin.schema import Schema, ColumnSchema, Tags
            
            # Créer les colonnes pour le schéma merlin
            column_schemas = []

            # Traiter chaque spécification de feature
            for feature_spec in schema.get("feature_specs", []):
                feature_name = feature_spec["name"]
                dtype = feature_spec.get("dtype", "int32")
                is_sequence = feature_spec.get("is_sequence", False)
                vocab_size = feature_spec.get("vocab_size", 100)
                is_continuous = feature_spec.get("is_continuous", False)

                # Créer les tags appropriés
                tags = set()
                
                if is_continuous:
                    tags.add(Tags.CONTINUOUS)
                    column_dtype = "float32" if dtype == "float32" else "float64"
                else:
                    tags.add(Tags.CATEGORICAL)
                    column_dtype = "int32" if dtype == "int32" else "int64"
                
                if is_sequence:
                    tags.add(Tags.LIST)
                    is_list = True
                else:
                    is_list = False

                # Ajouter des tags spéciaux si c'est un ID d'item
                if "item" in feature_name.lower() or "product" in feature_name.lower():
                    tags.add(Tags.ITEM)
                    tags.add(Tags.ID)

                # Créer la colonne avec les métadonnées appropriées
                properties = {}
                if not is_continuous:
                    properties["vocab_size"] = vocab_size

                column_schema = ColumnSchema(
                    name=feature_name,
                    tags=tags,
                    dtype=column_dtype,
                    is_list=is_list,
                    properties=properties
                )
                
                column_schemas.append(column_schema)

            # Créer le schéma merlin avec toutes les colonnes
            if column_schemas:
                merlin_schema = Schema(column_schemas)
                logger.info(f"Schéma merlin créé avec {len(column_schemas)} colonnes")
                return merlin_schema
            else:
                logger.warning("Aucune colonne trouvée pour créer le schéma merlin")
                return None

        except ImportError as e:
            logger.error(f"merlin.schema non disponible: {e}")
            return None
        except Exception as e:
            logger.error(f"Erreur lors de la conversion du schéma: {e}")
            return None

    def get_architecture_info(self) -> Dict[str, Any]:
        """
        Retourne les informations sur l'architecture.

        Returns:
            Dictionnaire avec les informations de l'architecture
        """
        return {
            "name": self.__class__.__name__.replace("ModelBuilder", "").lower(),
            "default_config": self.default_config,
            "required_params": self.required_params,
            "description": self.__doc__ or "Pas de description disponible",
        }

    def __repr__(self) -> str:
        """Représentation string du builder."""
        arch_name = self.__class__.__name__.replace("ModelBuilder", "")
        return f"{arch_name}Builder()"


# Classe utilitaire pour la configuration des embeddings
class EmbeddingTableConfig:
    """
    Configuration pour les tables d'embedding.
    """
    
    def __init__(self, vocabulary_size: int, dim: int):
        """
        Initialise la configuration.
        
        Args:
            vocabulary_size: Taille du vocabulaire
            dim: Dimension des embeddings
        """
        self.vocabulary_size = vocabulary_size
        self.dim = dim


def test_input_module_creation(schema: Dict[str, Any]) -> Dict[str, Any]:
    """
    Teste la création du module d'entrée avec diagnostic.

    Args:
        schema: Schéma à tester

    Returns:
        Rapport de diagnostic
    """
    logger.info("Test de création du module d'entrée")
    
    from .gpt2_builder import GPT2ModelBuilder
    
    builder = GPT2ModelBuilder()
    
    test_results = {
        "schema_valid": False,
        "merlin_schema_created": False,
        "input_module_created": False,
        "masking_attribute": False,
        "errors": []
    }
    
    try:
        # Valider le schéma
        if "feature_specs" in schema and len(schema["feature_specs"]) > 0:
            test_results["schema_valid"] = True
        
        # Tester la conversion merlin
        merlin_schema = builder._convert_to_merlin_schema(schema)
        if merlin_schema is not None:
            test_results["merlin_schema_created"] = True
        
        # Tester la création du module
        input_module = builder.build_input_module(schema, 192, 15, "clm")
        if input_module is not None:
            test_results["input_module_created"] = True
            
            # Vérifier l'attribut masking
            if hasattr(input_module, 'masking'):
                test_results["masking_attribute"] = True
        
    except Exception as e:
        test_results["errors"].append(str(e))
        logger.error(f"Erreur lors du test: {e}")
    
    return test_results
    
    return diagnostics
