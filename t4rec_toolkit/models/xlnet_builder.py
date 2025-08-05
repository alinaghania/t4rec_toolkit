from typing import Dict, Any, List, Optional
import logging

from .base_builder import BaseModelBuilder
from .registry import model_registry
from ..core.exceptions import ConfigurationError

logger = logging.getLogger(__name__)


@model_registry.register_builder(
   "xlnet",
   metadata={
       "description": "XLNet architecture optimisée pour les séquences longues",
       "strengths": ["Attention bidirectionnelle", "Cache mémoire", "Séquences longues"],
       "recommended_for": ["Historiques bancaires", "Sessions longues", "Données temporelles"],
       "paper": "XLNet: Generalized Autoregressive Pretraining for Language Understanding"
   }
)
class XLNetModelBuilder(BaseModelBuilder):
   """
   Builder pour les modèles XLNet dans T4Rec.
   
   XLNet est particulièrement adapté pour les données séquentielles
   avec de longs historiques, grâce à son mécanisme de cache mémoire
   et son attention bidirectionnelle sans masquage artificiel.
   """
   
   def get_default_config(self) -> Dict[str, Any]:
       """
       Configuration par défaut optimisée pour les données bancaires.
       
       Returns:
           Configuration par défaut
       """
       return {
           'd_model': 256,           # Dimension des embeddings
           'n_head': 8,              # Nombre de têtes d'attention
           'n_layer': 4,             # Nombre de couches transformer
           'hidden_act': 'gelu',     # Fonction d'activation
           'dropout': 0.1,           # Taux de dropout
           'initializer_range': 0.02, # Range d'initialisation
           'layer_norm_eps': 1e-12,  # Epsilon pour layer norm
           'mem_len': 50,            # Longueur de cache mémoire
           'attn_type': 'bi',        # Type d'attention bidirectionnelle
           'pad_token': 0,           # Token de padding
           'max_sequence_length': 20, # Longueur max des séquences
           'masking': 'mlm',         # Type de masking
           'log_attention_weights': False, # Log des poids d'attention
           'use_projection': True,   # Utiliser projection MLP
           'projection_dim': None    # Dimension projection (None = d_model)
       }
   
   def get_required_parameters(self) -> List[str]:
       """
       Paramètres obligatoires pour XLNet.
       
       Returns:
           Liste des paramètres requis
       """
       return ['d_model', 'max_sequence_length']
   
   def _validate_specific_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
       """
       Validation spécifique à XLNet.
       
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
       
       # Valider dropout
       if not 0.0 <= config['dropout'] <= 1.0:
           raise ConfigurationError(
               f"Dropout doit être entre 0 et 1, reçu: {config['dropout']}",
               config_key="dropout"
           )
       
       # Valider mem_len
       if config['mem_len'] < 0:
           raise ConfigurationError(
               f"mem_len doit être >= 0, reçu: {config['mem_len']}",
               config_key="mem_len"
           )
       
       # Valider attn_type
       valid_attn_types = ['bi', 'uni']
       if config['attn_type'] not in valid_attn_types:
           raise ConfigurationError(
               f"attn_type invalide: {config['attn_type']}",
               config_key="attn_type",
               valid_values=valid_attn_types
           )
       
       # Valider masking
       valid_masking = ['mlm', 'plm', 'clm']
       if config['masking'] not in valid_masking:
           raise ConfigurationError(
               f"masking invalide: {config['masking']}",
               config_key="masking",
               valid_values=valid_masking
           )
       
       # Ajuster projection_dim si None
       if config['projection_dim'] is None:
           config['projection_dim'] = config['d_model']
       
       return config
   
   def build_transformer_config(self, **config) -> Any:
       """
       Construit la configuration XLNet pour T4Rec.
       
       Args:
           **config: Paramètres de configuration
           
       Returns:
           Configuration XLNet T4Rec
       """
       validated_config = self.validate_config(config)
       
       try:
           import transformers4rec.torch as tr
           
           # Construire la configuration XLNet
           transformer_config = tr.XLNetConfig.build(
               d_model=validated_config['d_model'],
               n_head=validated_config['n_head'],
               n_layer=validated_config['n_layer'],
               total_seq_length=validated_config['max_sequence_length'],
               attn_type=validated_config['attn_type'],
               hidden_act=validated_config['hidden_act'],
               initializer_range=validated_config['initializer_range'],
               layer_norm_eps=validated_config['layer_norm_eps'],
               dropout=validated_config['dropout'],
               pad_token=validated_config['pad_token'],
               log_attention_weights=validated_config['log_attention_weights'],
               mem_len=validated_config['mem_len']
           )
           
           logger.info(f"Configuration XLNet créée: d_model={validated_config['d_model']}, "
                      f"n_head={validated_config['n_head']}, n_layer={validated_config['n_layer']}, "
                      f"mem_len={validated_config['mem_len']}")
           
           return transformer_config
           
       except ImportError as e:
           raise ImportError(
               "transformers4rec requis pour XLNet. "
               "Installez avec: pip install transformers4rec==23.04.00"
           ) from e
   
   def build_model(self, 
                  schema: Dict[str, Any], 
                  **config) -> Any:
       """
       Construit le modèle XLNet complet.
       
       Args:
           schema: Schéma T4Rec des données
           **config: Configuration du modèle
           
       Returns:
           Modèle XLNet T4Rec prêt pour l'entraînement
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
               NDCGAt(top_ks=[10, 20], labels_onehot=True),
               RecallAt(top_ks=[10, 20], labels_onehot=True)
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
           
           logger.info(f"Modèle XLNet créé avec succès:")
           logger.info(f"  - Architecture: XLNet-{masking.upper()}")
           logger.info(f"  - Dimensions: {d_model}d, {validated_config['n_head']}h, {validated_config['n_layer']}l")
           logger.info(f"  - Séquence: {max_sequence_length}, Mémoire: {validated_config['mem_len']}")
           logger.info(f"  - Masking: {masking}, Attention: {validated_config['attn_type']}")
           
           return model
           
       except ImportError as e:
           raise ImportError(
               "transformers4rec requis. Installez avec: pip install transformers4rec==23.04.00"
           ) from e
       except Exception as e:
           logger.error(f"Erreur lors de la construction du modèle XLNet: {e}")
           raise ConfigurationError(f"Échec de construction XLNet: {str(e)}")
   
   def get_recommended_config_for_data_size(self, n_samples: int, n_features: int) -> Dict[str, Any]:
       """
       Retourne une configuration recommandée selon la taille des données.
       
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
               'mem_len': 20,
               'dropout': 0.2
           })
       elif n_samples < 10000:
           # Données moyennes : configuration par défaut
           pass
       else:
           # Grandes données : modèle plus complexe
           base_config.update({
               'd_model': 512,
               'n_head': 16,
               'n_layer': 6,
               'mem_len': 100,
               'dropout': 0.1
           })
       
       # Ajuster selon le nombre de features
       if n_features > 500:
           base_config['use_projection'] = True
           base_config['projection_dim'] = min(base_config['d_model'], n_features // 2)
       
       return base_config
   
   def get_cpu_optimized_config(self) -> Dict[str, Any]:
       """
       Configuration optimisée pour entraînement CPU.
       
       Returns:
           Configuration CPU-friendly
       """
       config = self.get_default_config()
       config.update({
           'd_model': 192,           # Réduire la dimension
           'n_head': 6,              # Réduire les têtes
           'n_layer': 3,             # Réduire les couches
           'mem_len': 30,            # Réduire la mémoire
           'dropout': 0.15,          # Légèrement plus de dropout
           'max_sequence_length': 15 # Séquences plus courtes
       })
       return config


# Utilitaires pour XLNet
def create_xlnet_banking_model(schema: Dict[str, Any], 
                             sequence_length: int = 20,
                             cpu_optimized: bool = False) -> Any:
   """
   Fonction utilitaire pour créer rapidement un modèle XLNet bancaire.
   
   Args:
       schema: Schéma des données
       sequence_length: Longueur des séquences
       cpu_optimized: Optimiser pour CPU
       
   Returns:
       Modèle XLNet configuré
   """
   builder = XLNetModelBuilder()
   
   if cpu_optimized:
       config = builder.get_cpu_optimized_config()
   else:
       config = builder.get_default_config()
   
   config['max_sequence_length'] = sequence_length
   
   return builder.build_model(schema, **config)

def get_xlnet_config_for_banking_data(n_samples: int, 
                                    n_features: int,
                                    sequence_length: int = 20) -> Dict[str, Any]:
   """
   Configuration XLNet spécialisée pour données bancaires.
   
   Args:
       n_samples: Nombre d'échantillons
       n_features: Nombre de features
       sequence_length: Longueur des séquences
       
   Returns:
       Configuration optimisée
   """
   builder = XLNetModelBuilder()
   config = builder.get_recommended_config_for_data_size(n_samples, n_features)
   
   # Spécialisations bancaires
   config.update({
       'max_sequence_length': sequence_length,
       'mem_len': min(sequence_length * 2, 50),  # Mémoire proportionnelle
       'masking': 'mlm',                         # MLM pour données bancaires
       'attn_type': 'bi',                        # Attention bidirectionnelle
       'use_projection': n_features > 100       # Projection si beaucoup de features
   })
   
   return config