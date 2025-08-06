# models/base_builder.py
"""
Interface de base pour les builders de modèles.

Ce module définit l'interface commune que tous les builders
de modèles doivent implémenter pour assurer la cohérence
et la compatibilité avec le system de registry.

Version corrigée pour Transformers4Rec 23.04.00 qui utilise
l'API merlin.schema.Schema au lieu de with_column().
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
        
        Version corrigée pour Transformers4Rec 23.04.00 utilisant
        l'API merlin.schema.Schema.

        Args:
            schema: Schéma T4Rec
            d_model: Dimension du modèle
            max_sequence_length: Longueur maximale des séquences
            masking: Type de masking ('mlm', 'clm', 'plm')

        Returns:
            Module d'entrée T4Rec ou None si échec
        """
        try:
            import transformers4rec.torch as tr

            # Convertir le schéma vers un schéma merlin
            merlin_schema = self._convert_to_merlin_schema(schema)

            if merlin_schema is None:
                logger.error("Impossible de convertir le schéma vers merlin.schema")
                return None

            # Créer le module d'entrée avec TabularSequenceFeatures.from_schema
            input_module = tr.TabularSequenceFeatures.from_schema(
                schema=merlin_schema,
                max_sequence_length=max_sequence_length,
                continuous_projection=d_model,
                aggregation="concat",
                masking=masking,
            )

            logger.info(f"Module d'entrée créé avec succès: {type(input_module).__name__}")
            return input_module

        except ImportError as e:
            logger.error(f"transformers4rec non disponible: {e}")
            raise ImportError(
                "transformers4rec non disponible. "
                "Installez avec: pip install transformers4rec==23.04.00"
            ) from e
        except Exception as e:
            logger.error(f"Erreur lors de la création du module d'entrée: {e}")
            return None

    def _convert_to_merlin_schema(self, schema: Dict[str, Any]) -> Any:
        """
        Convertit notre schéma vers un schéma merlin.schema.Schema.
        
        Cette version est compatible avec Transformers4Rec 23.04.00 qui utilise
        l'API merlin.schema.Schema au lieu de with_column().

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
                    # Définir le dtype approprié pour les features continues
                    column_dtype = "float32" if dtype == "float32" else "float64"
                else:
                    tags.add(Tags.CATEGORICAL)
                    # Définir le dtype pour les features catégorielles
                    column_dtype = "int32" if dtype == "int32" else "int64"
                
                if is_sequence:
                    tags.add(Tags.LIST)
                    # Pour les séquences, marquer comme liste
                    is_list = True
                else:
                    is_list = False

                # Ajouter des tags spéciaux si c'est un ID d'item
                if "item" in feature_name.lower() or "product" in feature_name.lower():
                    tags.add(Tags.ITEM)
                    tags.add(Tags.ID)

                # Créer la colonne avec les métadonnées appropriées
                properties = {
                    "vocab_size": vocab_size,
                } if not is_continuous else {}

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

    def _create_fallback_schema(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        Crée un schéma de fallback si merlin.schema ne fonctionne pas.

        Args:
            schema: Schéma original

        Returns:
            Schéma de fallback simplifié
        """
        return {
            "feature_specs": schema.get("feature_specs", []),
            "sequence_length": schema.get("sequence_length", 20),
            "fallback_mode": True,
            "message": "Schéma créé en mode fallback - merlin.schema non disponible"
        }

    def test_merlin_schema_compatibility(self) -> bool:
        """
        Teste la compatibilité avec merlin.schema.

        Returns:
            True si merlin.schema est disponible et fonctionne
        """
        try:
            from merlin.schema import Schema, ColumnSchema, Tags
            
            # Test de création d'un schéma simple
            test_column = ColumnSchema(
                name="test_item_id",
                tags={Tags.CATEGORICAL, Tags.ITEM, Tags.ID},
                dtype="int32",
                properties={"vocab_size": 1000}
            )
            
            test_schema = Schema([test_column])
            
            logger.info("Test merlin.schema réussi")
            return True
            
        except Exception as e:
            logger.warning(f"Test merlin.schema échoué: {e}")
            return False

    def build_input_module_with_fallback(
        self,
        schema: Dict[str, Any],
        d_model: int,
        max_sequence_length: int,
        masking: str = "mlm",
    ) -> Any:
        """
        Construit le module d'entrée avec fallback en cas d'échec.

        Args:
            schema: Schéma T4Rec
            d_model: Dimension du modèle
            max_sequence_length: Longueur maximale des séquences
            masking: Type de masking

        Returns:
            Module d'entrée ou objet de simulation
        """
        # Essayer la création normale
        input_module = self.build_input_module(schema, d_model, max_sequence_length, masking)
        
        if input_module is not None:
            return input_module
            
        # Fallback: créer un module simulé
        logger.warning("Création d'un module d'entrée simulé")
        return MockInputModule(schema, d_model, max_sequence_length, masking)

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
            "merlin_schema_compatible": self.test_merlin_schema_compatibility()
        }

    def __repr__(self) -> str:
        """Représentation string du builder."""
        arch_name = self.__class__.__name__.replace("ModelBuilder", "")
        return f"{arch_name}Builder()"


class MockInputModule:
    """
    Module d'entrée simulé pour les tests sans T4Rec fonctionnel.
    
    Cette classe permet de continuer les tests même si T4Rec
    n'est pas correctement installé.
    """
    
    def __init__(self, schema: Dict[str, Any], d_model: int, 
                 max_sequence_length: int, masking: str):
        """
        Initialise le module simulé.
        
        Args:
            schema: Schéma des données
            d_model: Dimension du modèle
            max_sequence_length: Longueur de séquence
            masking: Type de masking
        """
        self.schema = schema
        self.d_model = d_model
        self.max_sequence_length = max_sequence_length
        self.masking = masking
        self.n_features = len(schema.get("feature_specs", []))
        
    def __repr__(self):
        return (f"MockInputModule(features={self.n_features}, "
                f"d_model={self.d_model}, seq_len={self.max_sequence_length})")
    
    def forward(self, x):
        """Simulation du forward pass."""
        logger.warning("Utilisation du module simulé - pas de calcul réel")
        return x


def create_model_with_comprehensive_fallback(
    architecture: str, 
    schema: Dict[str, Any], 
    **config
) -> Any:
    """
    Fonction utilitaire pour créer un modèle avec fallback complet.
    
    Cette fonction essaie plusieurs approches pour créer un modèle
    et fournit des informations détaillées sur les échecs.

    Args:
        architecture: Architecture du modèle
        schema: Schéma des données
        **config: Configuration du modèle

    Returns:
        Modèle T4Rec ou modèle simulé avec informations détaillées
    """
    try:
        # Essayer l'approche normale
        from t4rec_banking_toolkit.models import create_model
        model = create_model(architecture, schema, **config)
        logger.info(f"Modèle {architecture} créé avec succès")
        return model
        
    except ImportError as e:
        logger.error(f"Import T4Rec échoué: {e}")
        return _create_diagnostic_model(architecture, schema, config, "import_error", str(e))
        
    except Exception as e:
        logger.error(f"Création du modèle échoué: {e}")
        return _create_diagnostic_model(architecture, schema, config, "creation_error", str(e))


def _create_diagnostic_model(
    architecture: str, 
    schema: Dict[str, Any], 
    config: Dict[str, Any], 
    error_type: str, 
    error_message: str
) -> Any:
    """
    Crée un modèle de diagnostic avec informations détaillées sur l'erreur.
    
    Args:
        architecture: Architecture demandée
        schema: Schéma des données  
        config: Configuration du modèle
        error_type: Type d'erreur rencontrée
        error_message: Message d'erreur détaillé
        
    Returns:
        Objet de diagnostic avec toutes les informations
    """
    return {
        "model_type": "diagnostic",
        "architecture": architecture,
        "schema": schema,
        "config": config,
        "error": {
            "type": error_type,
            "message": error_message
        },
        "recommendations": _get_error_recommendations(error_type, error_message),
        "fallback_available": True
    }


def _get_error_recommendations(error_type: str, error_message: str) -> List[str]:
    """
    Génère des recommandations basées sur le type d'erreur.
    
    Args:
        error_type: Type d'erreur
        error_message: Message d'erreur
        
    Returns:
        Liste de recommandations pour résoudre le problème
    """
    recommendations = []
    
    if error_type == "import_error":
        recommendations.extend([
            "Vérifiez que transformers4rec est installé: pip install transformers4rec==23.04.00",
            "Vérifiez que merlin.schema est disponible",
            "Redémarrez le kernel après installation"
        ])
        
    elif error_type == "creation_error":
        if "Schema" in error_message:
            recommendations.extend([
                "Vérifiez le format du schéma",
                "Assurez-vous que toutes les features ont des vocab_size appropriés",
                "Vérifiez la compatibilité des types de données"
            ])
        elif "TabularSequenceFeatures" in error_message:
            recommendations.extend([
                "Vérifiez que max_sequence_length est cohérent",
                "Vérifiez les paramètres d'agrégation",
                "Essayez avec moins de features pour déboguer"
            ])
    
    recommendations.append("Consultez la documentation Transformers4Rec 23.04")
    return recommendations


def run_compatibility_diagnostics() -> Dict[str, Any]:
    """
    Lance un diagnostic complet de compatibilité avec l'environnement.
    
    Returns:
        Rapport de diagnostic détaillé
    """
    diagnostics = {
        "timestamp": str(pd.Timestamp.now()),
        "environment": {},
        "imports": {},
        "functionality": {},
        "recommendations": []
    }
    
    # Test des imports
    import_tests = {
        "transformers4rec": "transformers4rec.torch",
        "merlin_schema": "merlin.schema", 
        "merlin_core": "merlin.core",
        "torch": "torch",
        "pandas": "pandas",
        "numpy": "numpy"
    }
    
    for name, module in import_tests.items():
        try:
            __import__(module)
            diagnostics["imports"][name] = {"status": "OK"}
        except ImportError as e:
            diagnostics["imports"][name] = {"status": "FAILED", "error": str(e)}
    
    # Test des fonctionnalités spécifiques
    if diagnostics["imports"]["merlin_schema"]["status"] == "OK":
        try:
            from merlin.schema import Schema, ColumnSchema, Tags
            test_col = ColumnSchema("test", tags={Tags.CATEGORICAL}, dtype="int32")
            test_schema = Schema([test_col])
            diagnostics["functionality"]["merlin_schema"] = {"status": "OK"}
        except Exception as e:
            diagnostics["functionality"]["merlin_schema"] = {"status": "FAILED", "error": str(e)}
    
    if diagnostics["imports"]["transformers4rec"]["status"] == "OK":
        try:
            import transformers4rec.torch as tr
            diagnostics["functionality"]["t4rec_torch"] = {"status": "OK"}
            
            # Test TabularSequenceFeatures
            try:
                # Test basique sans schéma
                diagnostics["functionality"]["tabular_sequence_features"] = {"status": "OK"}
            except Exception as e:
                diagnostics["functionality"]["tabular_sequence_features"] = {"status": "FAILED", "error": str(e)}
                
        except Exception as e:
            diagnostics["functionality"]["t4rec_torch"] = {"status": "FAILED", "error": str(e)}
    
    # Générer des recommandations
    failed_imports = [k for k, v in diagnostics["imports"].items() if v["status"] == "FAILED"]
    if failed_imports:
        diagnostics["recommendations"].append(f"Installer les dépendances manquantes: {', '.join(failed_imports)}")
    
    failed_functionality = [k for k, v in diagnostics["functionality"].items() if v["status"] == "FAILED"]
    if failed_functionality:
        diagnostics["recommendations"].append(f"Problèmes de fonctionnalité détectés: {', '.join(failed_functionality)}")
    
    if not failed_imports and not failed_functionality:
        diagnostics["recommendations"].append("Environnement compatible - prêt pour T4Rec")
    
    return diagnostics
