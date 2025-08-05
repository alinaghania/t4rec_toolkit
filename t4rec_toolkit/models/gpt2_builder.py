# models/gpt2_builder.py
"""
Builder pour les modèles GPT2.

Ce module fournit la construction de modèles GPT2 optimisés
pour les sessions courtes et la prédiction causale dans
les systèmes de recommandation.
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
            
            # Construire le module d'entrée
            input_module = self.build_input_module(
                schema, d_model, max_sequence_length, masking
            )
            
            # Construire la configuration du transformer
            transformer_config = self.build_transformer_config(**validated_config)
            
            # Construire le corps du modèle
            body_layers = [input_module]
            
            # Ajouter projection MLP si demandée
            if use_projection and projection_dim != d_model:
                body_layers.append(tr.MLPBlock([projection_dim]))
            
            # Ajouter le bloc transformer
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


    # Utilitaires pour GPT2
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