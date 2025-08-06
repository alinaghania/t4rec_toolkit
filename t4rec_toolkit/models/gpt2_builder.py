# models/gpt2_builder.py
"""
Builder pour les modèles GPT2.

Ce module fournit la construction de modèles GPT2 optimisés
pour les sessions courtes et la prédiction causale dans
les systèmes de recommandation.

Version corrigée pour gérer le problème input_module = None.
"""

from typing import Dict, Any, List, Optional
import logging

from .base_builder import BaseModelBuilder
from .registry import model_registry
from ..core.exceptions import ConfigurationError

logger = logging.getLogger(__name__)


@model_registry.register_builder(
    "gpt2",
    metadata={
        "description": "GPT2 architecture pour sessions courtes et prédiction causale",
        "strengths": ["Génération autoregressive", "Sessions courtes", "Prédiction séquentielle"],
        "recommended_for": ["Navigation mobile", "Sessions rapides", "Prédiction en temps réel"],
        "paper": "Language Models are Unsupervised Multitask Learners"
    }
)
class GPT2ModelBuilder(BaseModelBuilder):
    """
    Builder pour les modèles GPT2 dans T4Rec.
    
    GPT2 est optimisé pour la modélisation autoregressive et
    la prédiction causale, particulièrement efficace pour
    les sessions courtes et la génération séquentielle.
    """
    
    def get_default_config(self) -> Dict[str, Any]:
        """
        Configuration par défaut optimisée pour les sessions courtes.
        
        Returns:
            Configuration par défaut
        """
        return {
            'd_model': 192,           # Dimension des embeddings (n_embd)
            'n_head': 6,              # Nombre de têtes d'attention
            'n_layer': 3,             # Nombre de couches transformer
            'n_inner': None,          # Dimension feedforward (4*d_model si None)
            'activation_function': 'gelu_new', # Fonction d'activation
            'dropout': 0.1,           # Dropout général
            'attn_dropout': 0.1,      # Dropout attention
            'resid_dropout': 0.1,     # Dropout résiduel
            'embd_dropout': 0.1,      # Dropout embeddings
            'initializer_range': 0.02, # Range d'initialisation
            'layer_norm_eps': 1e-5,   # Epsilon pour layer norm
            'scale_attn_weights': True, # Scaling des poids d'attention
            'use_cache': True,        # Cache pour génération
            'pad_token': 0,           # Token de padding
            'bos_token': 1,           # Token de début
            'eos_token': 2,           # Token de fin
            'max_sequence_length': 50, # Longueur max (obligatoire pour GPT2)
            'masking': 'clm',         # Causal Language Modeling
            'log_attention_weights': False, # Log des poids d'attention
            'use_projection': True,   # Utiliser projection MLP
            'projection_dim': None,   # Dimension projection
            'summary_type': 'last'    # Type de summarization ('last', 'first', 'mean')
        }
    
    def get_required_parameters(self) -> List[str]:
        """
        Paramètres obligatoires pour GPT2.
        
        Returns:
            Liste des paramètres requis
        """
        return ['d_model', 'max_sequence_length']
    
    def _validate_specific_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validation spécifique à GPT2.
        
        Args:
            config: Configuration à valider
            
        Returns:
            Configuration validée
            
        Raises:
            ConfigurationError: Si la configuration est invalide
        """
        # Valider d_model vs n_head
        if config['d_model'] % config['n_head'] != 0:
            raise ConfigurationError(
                f"d_model ({config['d_model']}) doit être divisible par n_head ({config['n_head']})",
                config_key="d_model"
            )
        
        # Valider les valeurs positives
        positive_params = ['d_model', 'n_head', 'n_layer', 'max_sequence_length']
        for param in positive_params:
            if config[param] <= 0:
                raise ConfigurationError(
                    f"Paramètre {param} doit être positif, reçu: {config[param]}",
                    config_key=param
                )
        
        # Valider les dropout
        dropout_params = ['dropout', 'attn_dropout', 'resid_dropout', 'embd_dropout']
        for param in dropout_params:
            if not 0.0 <= config[param] <= 1.0:
                raise ConfigurationError(
                    f"{param} doit être entre 0 et 1, reçu: {config[param]}",
                    config_key=param
                )
        
        # Valider activation_function
        valid_activations = ['relu', 'silu', 'gelu', 'tanh', 'gelu_new']
        if config['activation_function'] not in valid_activations:
            raise ConfigurationError(
                f"activation_function invalide: {config['activation_function']}",
                config_key="activation_function",
                valid_values=valid_activations
            )
        
        # Valider masking (GPT2 = causal)
        if config['masking'] != 'clm':
            logger.warning(f"GPT2 recommande masking='clm', reçu: {config['masking']}")
        
        # Valider summary_type
        valid_summary = ['last', 'first', 'mean', 'cls_index']
        if config['summary_type'] not in valid_summary:
            raise ConfigurationError(
                f"summary_type invalide: {config['summary_type']}",
                config_key="summary_type",
                valid_values=valid_summary
            )
        
        # Ajuster n_inner si None
        if config['n_inner'] is None:
            config['n_inner'] = 4 * config['d_model']
        
        # Ajuster projection_dim si None
        if config['projection_dim'] is None:
            config['projection_dim'] = config['d_model']
        
        # GPT2 nécessite une séquence minimale
        if config['max_sequence_length'] < 10:
            logger.warning(f"GPT2 recommande max_sequence_length >= 10, reçu: {config['max_sequence_length']}")
        
        return config
    
    def build_transformer_config(self, **config) -> Any:
        """
        Construit la configuration GPT2 pour T4Rec.
        
        Args:
            **config: Paramètres de configuration
            
        Returns:
            Configuration GPT2 T4Rec
        """
        validated_config = self.validate_config(config)
        
        try:
            import transformers4rec.torch as tr
            
            # Construire la configuration GPT2
            transformer_config = tr.GPT2Config.build(
                d_model=validated_config['d_model'],
                n_head=validated_config['n_head'],
                n_layer=validated_config['n_layer'],
                total_seq_length=validated_config['max_sequence_length'],
                hidden_act=validated_config['activation_function'],
                initializer_range=validated_config['initializer_range'],
                layer_norm_eps=validated_config['layer_norm_eps'],
                dropout=validated_config['dropout'],
                pad_token=validated_config['pad_token'],
                log_attention_weights=validated_config['log_attention_weights']
            )
            
            logger.info(f"Configuration GPT2 créée: d_model={validated_config['d_model']}, "
                       f"n_head={validated_config['n_head']}, n_layer={validated_config['n_layer']}")
            
            return transformer_config
            
        except ImportError as e:
            raise ImportError(
                "transformers4rec requis pour GPT2. "
                "Installez avec: pip install transformers4rec==23.04.00"
            ) from e
    
    def build_model(self, 
                   schema: Dict[str, Any], 
                   **config) -> Any:
        """
        Construit le modèle GPT2 complet.
        
        Args:
            schema: Schéma T4Rec des données
            **config: Configuration du modèle
            
        Returns:
            Modèle GPT2 T4Rec prêt pour l'entraînement
        """
        validated_config = self.validate_config(config)
        
        try:
            import transformers4rec.torch as tr
            from transformers4rec.torch.ranking_metric import NDCGAt, RecallAt
            
            # Paramètres du modèle
            d_model = validated_config['d_model']
            max_sequence_length = validated_config['max_sequence_length']
            masking = validated_config['masking']
            use_projection = validated_config['use_projection']
            projection_dim = validated_config['projection_dim']
            summary_type = validated_config['summary_type']
            
            # Construire le module d'entrée avec vérifications renforcées
            logger.info("Construction du module d'entrée...")
            input_module = self.build_input_module(
                schema, d_model, max_sequence_length, masking
            )
            
            # Vérification critique du module d'entrée
            if input_module is None:
                error_msg = "Le module d'entrée n'a pas pu être créé"
                logger.error(error_msg)
                raise ConfigurationError(
                    error_msg,
                    config_key="input_module"
                )
            
            # Vérifier que le module a l'attribut masking requis
            if not hasattr(input_module, 'masking'):
                logger.warning("Module d'entrée sans attribut masking - ajout manuel")
                input_module.masking = masking
            
            logger.info(f"Module d'entrée créé: {type(input_module).__name__}")
            
            # Construire la configuration du transformer
            transformer_config = self.build_transformer_config(**validated_config)
            
            # Construire le corps du modèle
            body_layers = [input_module]
            
            # Ajouter projection MLP si demandée
            if use_projection and projection_dim != d_model:
                logger.info(f"Ajout d'une couche de projection: {d_model} -> {projection_dim}")
                body_layers.append(tr.MLPBlock([projection_dim]))
            
            # Ajouter le bloc transformer
            logger.info("Ajout du bloc transformer")
            body_layers.append(
                tr.TransformerBlock(
                    transformer_config, 
                    masking=input_module.masking
                )
            )
            
            body = tr.SequentialBlock(*body_layers)
            
            # Définir les métriques d'évaluation
            metrics = [
                NDCGAt(top_ks=[5, 10, 20], labels_onehot=True),
                RecallAt(top_ks=[5, 10, 20], labels_onehot=True)
            ]
            
            # Construire la tête de prédiction
            logger.info("Construction de la tête de prédiction")
            head = tr.Head(
                body,
                tr.NextItemPredictionTask(
                    weight_tying=True,
                    hf_format=True,
                    metrics=metrics,
                    padding_idx=validated_config['pad_token']
                ),
                inputs=input_module
            )
            
            # Modèle final
            model = tr.Model(head)
            
            logger.info(f"Modèle GPT2 créé avec succès:")
            logger.info(f"  - Architecture: GPT2-{masking.upper()}")
            logger.info(f"  - Dimensions: {d_model}d, {validated_config['n_head']}h, {validated_config['n_layer']}l")
            logger.info(f"  - Séquence: {max_sequence_length}, Cache: {validated_config['use_cache']}")
            logger.info(f"  - Masking: {masking}, Summary: {summary_type}")
            
            return model
            
        except ImportError as e:
            raise ImportError(
                "transformers4rec requis. Installez avec: pip install transformers4rec==23.04.00"
            ) from e
        except Exception as e:
            logger.error(f"Erreur lors de la construction du modèle GPT2: {e}")
            raise ConfigurationError(f"Échec de construction GPT2: {str(e)}")
    
    def create_simple_input_module(self, 
                                  schema: Dict[str, Any],
                                  d_model: int,
                                  max_sequence_length: int,
                                  masking: str) -> Any:
        """
        Crée un module d'entrée simplifié en cas d'échec des approches standards.
        
        Args:
            schema: Schéma des données
            d_model: Dimension du modèle
            max_sequence_length: Longueur de séquence
            masking: Type de masking
            
        Returns:
            Module d'entrée simplifié
        """
        try:
            import transformers4rec.torch as tr
            import torch
            
            logger.info("Création d'un module d'entrée simplifié")
            
            # Analyser les features du schéma
            feature_specs = schema.get("feature_specs", [])
            n_features = len(feature_specs)
            
            if n_features == 0:
                raise ValueError("Aucune feature trouvée dans le schéma")
            
            # Créer un module d'embedding simple pour toutes les features
            class SimpleInputModule(torch.nn.Module):
                def __init__(self, n_features, d_model, max_seq_len, masking_type):
                    super().__init__()
                    self.n_features = n_features
                    self.d_model = d_model
                    self.max_sequence_length = max_seq_len
                    self.masking = masking_type
                    
                    # Embedding pour chaque feature
                    self.feature_embeddings = torch.nn.ModuleDict()
                    
                    for i, spec in enumerate(feature_specs):
                        feature_name = spec["name"]
                        
                        if spec.get("is_continuous", False):
                            # Feature continue - utiliser une couche linéaire
                            self.feature_embeddings[feature_name] = torch.nn.Linear(1, d_model // n_features)
                        else:
                            # Feature catégorielle - utiliser embedding
                            vocab_size = spec.get("vocab_size", 100)
                            self.feature_embeddings[feature_name] = torch.nn.Embedding(
                                vocab_size, d_model // n_features
                            )
                    
                    # Projection finale
                    self.output_projection = torch.nn.Linear(d_model, d_model)
                    
                def forward(self, inputs):
                    """Forward pass du module simplifié."""
                    batch_size = None
                    feature_outputs = []
                    
                    for feature_name, feature_data in inputs.items():
                        if batch_size is None:
                            batch_size = feature_data.shape[0]
                            
                        if feature_name in self.feature_embeddings:
                            embedding_layer = self.feature_embeddings[feature_name]
                            
                            if isinstance(embedding_layer, torch.nn.Linear):
                                # Feature continue
                                if feature_data.dim() == 1:
                                    feature_data = feature_data.unsqueeze(-1)
                                embedded = embedding_layer(feature_data.float())
                            else:
                                # Feature catégorielle
                                embedded = embedding_layer(feature_data.long())
                            
                            feature_outputs.append(embedded)
                    
                    # Concaténer toutes les features
                    if feature_outputs:
                        concatenated = torch.cat(feature_outputs, dim=-1)
                        output = self.output_projection(concatenated)
                        
                        # S'assurer que la sortie a la bonne forme pour les séquences
                        if output.dim() == 2:
                            # Ajouter une dimension de séquence si nécessaire
                            output = output.unsqueeze(1).expand(-1, self.max_sequence_length, -1)
                        
                        return output
                    else:
                        # Fallback: retourner des zéros
                        return torch.zeros(batch_size, self.max_sequence_length, self.d_model)
            
            # Créer le module
            simple_module = SimpleInputModule(n_features, d_model, max_sequence_length, masking)
            
            logger.info(f"Module d'entrée simplifié créé avec {n_features} features")
            return simple_module
            
        except Exception as e:
            logger.error(f"Échec de création du module simplifié: {e}")
            raise ConfigurationError(f"Impossible de créer un module d'entrée: {str(e)}")
    
    def build_input_module(self, 
                          schema: Dict[str, Any],
                          d_model: int,
                          max_sequence_length: int,
                          masking: str = "mlm") -> Any:
        """
        Version surchargée pour GPT2 avec fallback vers module simplifié.
        
        Args:
            schema: Schéma T4Rec
            d_model: Dimension du modèle
            max_sequence_length: Longueur de séquence
            masking: Type de masking
            
        Returns:
            Module d'entrée
        """
        try:
            # Essayer l'approche standard du parent
            input_module = super().build_input_module(schema, d_model, max_sequence_length, masking)
            
            if input_module is not None:
                return input_module
            
            # Si échec, utiliser l'approche simplifiée
            logger.warning("Approche standard échouée - utilisation du module simplifié")
            return self.create_simple_input_module(schema, d_model, max_sequence_length, masking)
            
        except Exception as e:
            logger.warning(f"Toutes les approches ont échoué: {e}")
            # Dernière tentative avec le module simplifié
            return self.create_simple_input_module(schema, d_model, max_sequence_length, masking)
    
    def get_recommended_config_for_data_size(self, n_samples: int, n_features: int) -> Dict[str, Any]:
        """
        Configuration recommandée selon la taille des données.
        
        Args:
            n_samples: Nombre d'échantillons
            n_features: Nombre de features
            
        Returns:
            Configuration optimisée
        """
        base_config = self.get_default_config()
        
        # Ajuster selon la taille des données
        if n_samples < 1000:
            # Petites données : modèle plus simple
            base_config.update({
                'd_model': 128,
                'n_head': 4,
                'n_layer': 2,
                'max_sequence_length': 30,
                'dropout': 0.2
            })
        elif n_samples < 10000:
            # Données moyennes : configuration par défaut
            pass
        else:
            # Grandes données : modèle plus complexe
            base_config.update({
                'd_model': 384,
                'n_head': 12,
                'n_layer': 6,
                'max_sequence_length': 100,
                'dropout': 0.1
            })
        
        # Ajuster selon le nombre de features
        if n_features > 300:
            base_config['use_projection'] = True
            base_config['projection_dim'] = min(base_config['d_model'], n_features // 3)
        
        return base_config
    
    def get_cpu_optimized_config(self) -> Dict[str, Any]:
        """
        Configuration optimisée pour entraînement CPU.
        
        Returns:
            Configuration CPU-friendly
        """
        config = self.get_default_config()
        config.update({
            'd_model': 128,           # Réduire la dimension
            'n_head': 4,              # Réduire les têtes
            'n_layer': 2,             # Réduire les couches
            'max_sequence_length': 30, # Séquences plus courtes
            'dropout': 0.15,          # Légèrement plus de dropout
            'use_cache': False        # Désactiver cache pour économiser mémoire
        })
        return config
    
    def get_mobile_optimized_config(self) -> Dict[str, Any]:
        """
        Configuration optimisée pour sessions mobiles courtes.
        
        Returns:
            Configuration mobile-friendly
        """
        config = self.get_default_config()
        config.update({
            'd_model': 96,            # Très petit modèle
            'n_head': 3,              # Peu de têtes
            'n_layer': 2,             # Peu de couches
            'max_sequence_length': 20, # Sessions courtes
            'dropout': 0.2,           # Régularisation forte
            'use_cache': True,        # Cache pour prédiction rapide
            'summary_type': 'last'    # Dernière position pour next-item
        })
        return config


# Fonctions utilitaires pour GPT2
def create_gpt2_session_model(schema: Dict[str, Any], 
                            sequence_length: int = 50,
                            mobile_optimized: bool = False,
                            cpu_optimized: bool = False) -> Any:
    """
    Fonction utilitaire pour créer rapidement un modèle GPT2 pour sessions.
    
    Args:
        schema: Schéma des données
        sequence_length: Longueur des séquences
        mobile_optimized: Optimiser pour mobile
        cpu_optimized: Optimiser pour CPU
        
    Returns:
        Modèle GPT2 configuré
    """
    builder = GPT2ModelBuilder()
    
    if mobile_optimized:
        config = builder.get_mobile_optimized_config()
    elif cpu_optimized:
        config = builder.get_cpu_optimized_config()
    else:
        config = builder.get_default_config()
    
    config['max_sequence_length'] = sequence_length
    
    return builder.build_model(schema, **config)


def get_gpt2_config_for_sessions(session_length: int,
                               n_samples: int, 
                               n_features: int) -> Dict[str, Any]:
    """
    Configuration GPT2 spécialisée pour données de session.
    
    Args:
        session_length: Longueur moyenne des sessions
        n_samples: Nombre d'échantillons
        n_features: Nombre de features
        
    Returns:
        Configuration optimisée
    """
    builder = GPT2ModelBuilder()
    config = builder.get_recommended_config_for_data_size(n_samples, n_features)
    
    # Spécialisations pour sessions
    config.update({
        'max_sequence_length': max(session_length + 10, 20),  # Marge pour croissance
        'masking': 'clm',                                     # Causal pour sessions
        'summary_type': 'last',                               # Dernière position
        'use_cache': True,                                    # Cache pour prédiction
        'use_projection': n_features > 50                    # Projection si beaucoup de features
    })
    
    # Ajuster selon la longueur de session
    if session_length < 20:
        # Sessions très courtes (mobile)
        config.update({
            'd_model': 128,
            'n_head': 4,
            'n_layer': 2
        })
    elif session_length > 100:
        # Sessions très longues (desktop)
        config.update({
            'd_model': 256,
            'n_head': 8,
            'n_layer': 4
        })
    
    return config


def diagnose_gpt2_creation_failure(schema: Dict[str, Any], **config) -> Dict[str, Any]:
    """
    Diagnostique les échecs de création de modèle GPT2.
    
    Args:
        schema: Schéma utilisé
        **config: Configuration utilisée
        
    Returns:
        Rapport de diagnostic
    """
    diagnosis = {
        "schema_analysis": {},
        "config_analysis": {},
        "import_checks": {},
        "recommendations": []
    }
    
    # Analyser le schéma
    diagnosis["schema_analysis"]["has_features"] = "feature_specs" in schema
    diagnosis["schema_analysis"]["n_features"] = len(schema.get("feature_specs", []))
    diagnosis["schema_analysis"]["has_continuous"] = any(
        spec.get("is_continuous", False) for spec in schema.get("feature_specs", [])
    )
    diagnosis["schema_analysis"]["has_categorical"] = any(
        not spec.get("is_continuous", True) for spec in schema.get("feature_specs", [])
    )
    
    # Analyser la configuration
    builder = GPT2ModelBuilder()
    try:
        validated_config = builder.validate_config(config)
        diagnosis["config_analysis"]["valid"] = True
        diagnosis["config_analysis"]["config"] = validated_config
    except Exception as e:
        diagnosis["config_analysis"]["valid"] = False
        diagnosis["config_analysis"]["error"] = str(e)
    
    # Vérifier les imports
    try:
        import transformers4rec.torch as tr
        diagnosis["import_checks"]["transformers4rec"] = True
    except ImportError:
        diagnosis["import_checks"]["transformers4rec"] = False
        diagnosis["recommendations"].append("Installer transformers4rec: pip install transformers4rec==23.04.00")
    
    try:
        from merlin.schema import Schema
        diagnosis["import_checks"]["merlin_schema"] = True
    except ImportError:
        diagnosis["import_checks"]["merlin_schema"] = False
        diagnosis["recommendations"].append("Installer merlin-core: pip install merlin-core")
    
    # Recommandations basées sur l'analyse
    if diagnosis["schema_analysis"]["n_features"] == 0:
        diagnosis["recommendations"].append("Le schéma ne contient aucune feature")
    
    if not diagnosis["config_analysis"]["valid"]:
        diagnosis["recommendations"].append("Corriger la configuration du modèle")
    
    return diagnosis
