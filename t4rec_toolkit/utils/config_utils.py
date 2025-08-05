# utils/config_utils.py
"""
Utilitaires pour la gestion des configurations.

Ce module fournit les fonctions pour valider, fusionner
et gérer les configurations du toolkit.
"""

import os
from typing import Dict, Any, List, Optional, Union
import logging

logger = logging.getLogger(__name__)


def merge_configs(*configs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Fusionne plusieurs configurations en donnant priorité aux dernières.
    
    Args:
        *configs: Configurations à fusionner
        
    Returns:
        Configuration fusionnée
    """
    if not configs:
        return {}
    
    merged = {}
    
    for config in configs:
        if config:
            merged.update(config)
    
    return merged


def validate_config_schema(config: Dict[str, Any], 
                          required_keys: List[str],
                          optional_keys: Optional[List[str]] = None) -> bool:
    """
    Valide qu'une configuration respecte un schéma.
    
    Args:
        config: Configuration à valider
        required_keys: Clés obligatoires
        optional_keys: Clés optionnelles autorisées
        
    Returns:
        True si la configuration est valide
        
    Raises:
        ValueError: Si la configuration est invalide
    """
    # Vérifier les clés obligatoires
    missing_keys = set(required_keys) - set(config.keys())
    if missing_keys:
        raise ValueError(f"Clés obligatoires manquantes: {missing_keys}")
    
    # Vérifier les clés non autorisées
    if optional_keys is not None:
        allowed_keys = set(required_keys) + set(optional_keys)
        extra_keys = set(config.keys()) - allowed_keys
        if extra_keys:
            logger.warning(f"Clés non reconnues dans la configuration: {extra_keys}")
    
    return True


def get_default_training_args(cpu_mode: bool = False,
                             batch_size: Optional[int] = None,
                             learning_rate: Optional[float] = None,
                             num_epochs: int = 10) -> Dict[str, Any]:
    """
    Génère des arguments d'entraînement par défaut pour T4Rec.
    
    Args:
        cpu_mode: Mode CPU uniquement
        batch_size: Taille de batch (auto si None)
        learning_rate: Taux d'apprentissage (auto si None)
        num_epochs: Nombre d'époques
        
    Returns:
        Arguments d'entraînement T4Rec
    """
    # Configuration de base
    training_args = {
        'output_dir': './results',
        'overwrite_output_dir': True,
        'num_train_epochs': num_epochs,
        'per_device_eval_batch_size': 32,
        'logging_steps': 100,
        'eval_steps': 500,
        'save_steps': 1000,
        'evaluation_strategy': 'steps',
        'save_strategy': 'steps',
        'metric_for_best_model': 'eval_loss',
        'load_best_model_at_end': True,
        'dataloader_drop_last': True,
        'remove_unused_columns': False,
        'label_names': ['labels']
    }
    
    # Ajustements pour CPU
    if cpu_mode:
        training_args.update({
            'no_cuda': True,
            'dataloader_num_workers': min(4, os.cpu_count() or 1),
            'per_device_train_batch_size': batch_size or 16,
            'learning_rate': learning_rate or 1e-4,
            'warmup_steps': 100,
            'lr_scheduler_type': 'linear',
            'fp16': False  # FP16 peut ne pas marcher sur CPU
        })
    else:
        # Configuration GPU
        training_args.update({
            'per_device_train_batch_size': batch_size or 64,
            'learning_rate': learning_rate or 5e-4,
            'warmup_steps': 200,
            'lr_scheduler_type': 'cosine',
            'fp16': True
        })
    
    return training_args


def get_environment_config() -> Dict[str, Any]:
    """
    Détecte automatiquement la configuration de l'environnement.
    
    Returns:
        Configuration de l'environnement
    """
    config = {
        'cpu_count': os.cpu_count(),
        'cuda_available': False,
        'dataiku_env': False,
        'memory_gb': None
    }
    
    # Vérifier CUDA
    try:
        import torch
        config['cuda_available'] = torch.cuda.is_available()
        if config['cuda_available']:
            config['gpu_count'] = torch.cuda.device_count()
            config['gpu_names'] = [torch.cuda.get_device_name(i) 
                                 for i in range(torch.cuda.device_count())]
    except ImportError:
        pass
    
    # Vérifier Dataiku
    try:
        import dataiku
        config['dataiku_env'] = True
        config['dataiku_version'] = dataiku.__version__
    except ImportError:
        pass
    
    # Estimer la mémoire disponible
    try:
        import psutil
        memory_info = psutil.virtual_memory()
        config['memory_gb'] = round(memory_info.total / (1024**3), 1)
        config['memory_available_gb'] = round(memory_info.available / (1024**3), 1)
    except ImportError:
        pass
    
    return config


def adapt_config_to_environment(base_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Adapte une configuration à l'environnement détecté.
    
    Args:
        base_config: Configuration de base
        
    Returns:
        Configuration adaptée
    """
    env_config = get_environment_config()
    adapted_config = base_config.copy()
    
    # Adaptations selon la mémoire disponible
    if env_config.get('memory_gb'):
        memory_gb = env_config['memory_gb']
        
        if memory_gb < 8:
            # Mémoire limitée
            adapted_config.update({
                'd_model': min(adapted_config.get('d_model', 256), 128),
                'batch_size': min(adapted_config.get('batch_size', 32), 16),
                'max_sequence_length': min(adapted_config.get('max_sequence_length', 50), 30)
            })
            logger.info(f"Configuration adaptée pour mémoire limitée ({memory_gb}GB)")
        
        elif memory_gb > 64:
            # Beaucoup de mémoire
            adapted_config.update({
                'batch_size': max(adapted_config.get('batch_size', 32), 128)
            })
            logger.info(f"Configuration adaptée pour haute mémoire ({memory_gb}GB)")
    
    # Adaptations pour Dataiku
    if env_config.get('dataiku_env'):
        adapted_config.update({
            'dataloader_engine': 'pyarrow',  # Pas nvtabular par défaut
            'save_steps': 500,  # Sauvegardes plus fréquentes
            'logging_steps': 50  # Logs plus fréquents
        })
        logger.info("Configuration adaptée pour environnement Dataiku")
    
    # Adaptations CPU/GPU
    if not env_config.get('cuda_available'):
        adapted_config.update({
            'no_cuda': True,
            'fp16': False,
            'dataloader_num_workers': min(4, env_config.get('cpu_count', 1))
        })
        logger.info("Configuration adaptée pour CPU uniquement")
    
    return adapted_config


def create_config_template(architecture: str = 'xlnet') -> Dict[str, Any]:
    """
    Crée un template de configuration pour une architecture.
    
    Args:
        architecture: Architecture du modèle
        
    Returns:
        Template de configuration
    """
    base_template = {
        'model': {
            'architecture': architecture,
            'max_sequence_length': 20,
        },
        'training': get_default_training_args(),
        'data': {
            'target_column': 'target',
            'validation_split': 0.2,
            'test_split': 0.1
        },
        'preprocessing': {
            'sequence_transformer': {
                'vocab_size': 10000,
                'encoding_strategy': 'quantile'
            },
            'categorical_transformer': {
                'max_categories': 1000,
                'handle_unknown': 'encode'
            }
        }
    }
    
    # Ajustements spécifiques par architecture
    if architecture == 'xlnet':
        base_template['model'].update({
            'd_model': 256,
            'n_head': 8,
            'n_layer': 4,
            'mem_len': 50,
            'masking': 'mlm'
        })
    elif architecture == 'gpt2':
        base_template['model'].update({
            'd_model': 192,
            'n_head': 6,
            'n_layer': 3,
            'max_sequence_length': 50,
            'masking': 'clm'
        })
    
    return base_template


def validate_training_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Valide et nettoie une configuration d'entraînement.
    
    Args:
        config: Configuration à valider
        
    Returns:
        Configuration validée
        
    Raises:
        ValueError: Si la configuration est invalide
    """
    required_sections = ['model', 'training']
    
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Section requise manquante: {section}")
    
    # Valider la section model
    model_config = config['model']
    required_model_keys = ['architecture', 'max_sequence_length']
    validate_config_schema(model_config, required_model_keys)
    
    # Valider la section training
    training_config = config['training']
    required_training_keys = ['num_train_epochs', 'per_device_train_batch_size']
    validate_config_schema(training_config, required_training_keys)
    
    # Nettoyer les valeurs
    validated_config = config.copy()
    
    # S'assurer que les valeurs numériques sont positives
    numeric_checks = [
        ('model.max_sequence_length', int),
        ('training.num_train_epochs', int),
        ('training.per_device_train_batch_size', int)
    ]
    
    for key_path, expected_type in numeric_checks:
        keys = key_path.split('.')
        value = validated_config
        for key in keys:
            value = value[key]
        
        if not isinstance(value, expected_type) or value <= 0:
            raise ValueError(f"Valeur invalide pour {key_path}: {value}")
    
    return validated_config