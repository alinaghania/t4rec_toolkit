# transformers/categorical_transformer.py
"""
Transformer pour les données catégorielles.

Ce module gère la transformation des features catégorielles,
incluant l'encodage des variables dummy, la gestion des catégories
rares et la création de vocabulaires pour T4Rec.
"""

from typing import Dict, List, Any, Optional, Union, Tuple, Set
import pandas as pd
import numpy as np
from collections import Counter

from ..core.base_transformer import BaseTransformer, TransformationResult
from ..core.exceptions import TransformationError


class CategoricalTransformer(BaseTransformer):
   """
   Transformer pour les données catégorielles.
   
   Cette classe transforme les features catégorielles en encodages
   numériques appropriés pour les modèles de recommandation,
   avec gestion des catégories rares et inconnues.
   """
   
   def __init__(self,
                max_categories: int = 1000,
                min_frequency: int = 1,
                handle_unknown: str = 'encode',
                unknown_value: int = 1,
                rare_category_threshold: float = 0.01,
                drop_first: bool = False,
                name: Optional[str] = None):
       """
       Initialise le transformer catégoriel.
       
       Args:
           max_categories: Nombre maximum de catégories par feature
           min_frequency: Fréquence minimale pour garder une catégorie
           handle_unknown: Stratégie pour les valeurs inconnues ('encode', 'ignore', 'error')
           unknown_value: Valeur pour les catégories inconnues
           rare_category_threshold: Seuil pour considérer une catégorie comme rare
           drop_first: Supprimer la première catégorie (pour éviter la multicolinéarité)
           name: Nom du transformer
       """
       super().__init__(name)
       self.max_categories = max_categories
       self.min_frequency = min_frequency
       self.handle_unknown = handle_unknown
       self.unknown_value = unknown_value
       self.rare_category_threshold = rare_category_threshold
       self.drop_first = drop_first
       
       # Paramètres ajustés pendant le fit
       self.category_mappings = {}
       self.vocabulary_sizes = {}
       self.category_frequencies = {}
       self.rare_categories = {}
       
       # Configuration
       self.config.update({
           'max_categories': max_categories,
           'min_frequency': min_frequency,
           'handle_unknown': handle_unknown,
           'unknown_value': unknown_value,
           'rare_category_threshold': rare_category_threshold,
           'drop_first': drop_first
       })
   
   def fit(self, 
           data: pd.DataFrame, 
           feature_columns: Optional[List[str]] = None,
           **kwargs) -> 'CategoricalTransformer':
       """
       Ajuste le transformer sur les données catégorielles.
       
       Args:
           data: DataFrame contenant les features catégorielles
           feature_columns: Colonnes catégorielles à traiter
           **kwargs: Paramètres supplémentaires
           
       Returns:
           Self pour le chaînage
       """
       categorical_columns = self.get_feature_columns(data, feature_columns)
       
       if not categorical_columns:
           raise TransformationError(
               "Aucune colonne catégorielle détectée",
               transformer_name=self.name,
               step="fit"
           )
       
       # Traiter chaque colonne catégorielle
       for col in categorical_columns:
           self._fit_column(data[col], col)
           self.feature_mappings[col] = f"{col}_encoded"
       
       self.is_fitted = True
       return self
   
   def transform(self, data: pd.DataFrame) -> TransformationResult:
       """
       Transforme les features catégorielles en encodages numériques.
       
       Args:
           data: DataFrame contenant les features à transformer
           
       Returns:
           Résultat de la transformation
       """
       self._check_fitted()
       
       transformed_data = {}
       feature_info = {}
       original_columns = []
       transformation_steps = []
       
       for col in self.category_mappings.keys():
           if col not in data.columns:
               raise TransformationError(
                   f"Colonne manquante pour la transformation: {col}",
                   transformer_name=self.name,
                   step="transform"
               )
           
           # Transformer la colonne
           encoded_values = self._transform_column(data[col], col)
           
           # Stocker les résultats
           feature_name = self.feature_mappings[col]
           transformed_data[feature_name] = encoded_values
           
           feature_info[feature_name] = {
               'original_column': col,
               'dtype': 'int32',
               'shape': encoded_values.shape,
               'vocab_size': self.vocabulary_sizes[col],
               'is_sequence': False,
               'is_categorical': True,
               'n_categories': len(self.category_mappings[col]),
               'handle_unknown': self.handle_unknown,
               'unknown_value': self.unknown_value
           }
           
           original_columns.append(col)
           transformation_steps.append(f"encode_categorical_{col}")
       
       return TransformationResult(
           data=transformed_data,
           feature_info=feature_info,
           config=self.get_config(),
           original_columns=original_columns,
           transformation_steps=transformation_steps
       )
   
   def _auto_detect_features(self, data: pd.DataFrame) -> List[str]:
       """
       Détecte automatiquement les colonnes catégorielles.
       
       Args:
           data: DataFrame à analyser
           
       Returns:
           Liste des colonnes catégorielles détectées
       """
       categorical_columns = []
       
       for col in data.columns:
           if self._is_categorical_column(data[col], col):
               categorical_columns.append(col)
       
       return categorical_columns
   
   def _is_categorical_column(self, series: pd.Series, col_name: str) -> bool:
       """
       Détermine si une colonne est catégorielle.
       
       Args:
           series: Série à analyser
           col_name: Nom de la colonne
           
       Returns:
           True si la colonne est catégorielle
       """
       # Vérifier les patterns de noms typiques
       if col_name.startswith('dummy') or '_cat_' in col_name or col_name.endswith('_id'):
           return True
       
       # Vérifier le type de données
       if series.dtype == 'object' or series.dtype.name == 'category':
           return True
       
       # Vérifier la cardinalité pour les colonnes numériques
       if pd.api.types.is_numeric_dtype(series):
           n_unique = series.nunique()
           n_total = len(series.dropna())
           
           if n_total == 0:
               return False
               
           cardinality_ratio = n_unique / n_total
           
           # Si peu de valeurs uniques par rapport au total, probablement catégoriel
           if cardinality_ratio < 0.1 and n_unique < 100:
               return True
           
           # Vérifier si les valeurs sont des entiers dans une plage limitée
           if series.dtype in ['int32', 'int64']:
               unique_values = series.dropna().unique()
               if len(unique_values) <= 50 and all(isinstance(x, (int, np.integer)) for x in unique_values):
                   return True
       
       return False
   
   def _fit_column(self, series: pd.Series, col_name: str):
       """
       Ajuste le transformer sur une colonne catégorielle.
       
       Args:
           series: Série à traiter
           col_name: Nom de la colonne
       """
       # Compter les fréquences des catégories
       value_counts = series.value_counts(dropna=False)
       total_count = len(series)
       
       # Identifier les catégories rares
       rare_threshold = max(self.min_frequency, int(total_count * self.rare_category_threshold))
       rare_categories = set(value_counts[value_counts < rare_threshold].index)
       
       # Créer le mapping des catégories
       category_mapping = {}
       current_id = 0
       
       # Réserver l'ID 0 pour les valeurs inconnues/padding si nécessaire
       if self.handle_unknown == 'encode':
           current_id = self.unknown_value
       
       # Trier les catégories par fréquence (plus fréquentes en premier)
       sorted_categories = value_counts.index.tolist()
       
       # Filtrer les catégories rares si nécessaire
       kept_categories = []
       for cat in sorted_categories:
           if cat not in rare_categories:
               kept_categories.append(cat)
           if len(kept_categories) >= self.max_categories:
               break
       
       # Créer le mapping
       for i, category in enumerate(kept_categories):
           if self.drop_first and i == 0:
               continue  # Skip la première catégorie
           
           category_mapping[category] = current_id
           current_id += 1
       
       # Gérer les catégories rares en les mappant vers une valeur spéciale
       if rare_categories:
           rare_id = current_id
           for rare_cat in rare_categories:
               category_mapping[rare_cat] = rare_id
           current_id += 1
       
       # Stocker les résultats
       self.category_mappings[col_name] = category_mapping
       self.vocabulary_sizes[col_name] = current_id
       self.category_frequencies[col_name] = value_counts.to_dict()
       self.rare_categories[col_name] = rare_categories
   
   def _transform_column(self, series: pd.Series, col_name: str) -> np.ndarray:
       """
       Transforme une colonne catégorielle.
       
       Args:
           series: Série à transformer
           col_name: Nom de la colonne
           
       Returns:
           Array numpy des valeurs encodées
       """
       mapping = self.category_mappings[col_name]
       encoded_values = np.zeros(len(series), dtype=np.int32)
       
       for i, value in enumerate(series):
           if pd.isna(value):
               # Gérer les valeurs manquantes
               if self.handle_unknown == 'encode':
                   encoded_values[i] = self.unknown_value
               else:
                   encoded_values[i] = 0  # Valeur par défaut
           elif value in mapping:
               encoded_values[i] = mapping[value]
           else:
               # Gérer les catégories inconnues
               if self.handle_unknown == 'encode':
                   encoded_values[i] = self.unknown_value
               elif self.handle_unknown == 'ignore':
                   encoded_values[i] = 0
               elif self.handle_unknown == 'error':
                   raise TransformationError(
                       f"Catégorie inconnue '{value}' dans la colonne '{col_name}'",
                       transformer_name=self.name,
                       step="transform"
                   )
       
       return encoded_values
   
   def get_category_mapping(self, column_name: str) -> Dict[Any, int]:
       """
       Retourne le mapping des catégories pour une colonne.
       
       Args:
           column_name: Nom de la colonne
           
       Returns:
           Dictionnaire du mapping catégorie -> ID
       """
       if column_name not in self.category_mappings:
           raise ValueError(f"Pas de mapping pour la colonne: {column_name}")
       
       return self.category_mappings[column_name].copy()
   
   def get_vocab_size(self, column_name: str) -> int:
       """
       Retourne la taille du vocabulaire pour une colonne.
       
       Args:
           column_name: Nom de la colonne
           
       Returns:
           Taille du vocabulaire
       """
       return self.vocabulary_sizes.get(column_name, 0)
   
   def get_category_frequencies(self, column_name: str) -> Dict[Any, int]:
       """
       Retourne les fréquences des catégories pour une colonne.
       
       Args:
           column_name: Nom de la colonne
           
       Returns:
           Dictionnaire des fréquences
       """
       return self.category_frequencies.get(column_name, {})
   
   def decode_categories(self, 
                        encoded_values: np.ndarray, 
                        column_name: str) -> List[Any]:
       """
       Décode les valeurs encodées vers les catégories originales.
       
       Args:
           encoded_values: Valeurs encodées
           column_name: Nom de la colonne originale
           
       Returns:
           Liste des catégories décodées
       """
       if column_name not in self.category_mappings:
           raise ValueError(f"Pas de mapping pour la colonne: {column_name}")
       
       # Inverser le mapping
       inverse_mapping = {v: k for k, v in self.category_mappings[column_name].items()}
       
       decoded_categories = []
       for encoded_val in encoded_values:
           if encoded_val in inverse_mapping:
               decoded_categories.append(inverse_mapping[encoded_val])
           else:
               decoded_categories.append(f"<unknown_{encoded_val}>")
       
       return decoded_categories
   
   def get_feature_importance_by_frequency(self, column_name: str) -> Dict[Any, float]:
       """
       Calcule l'importance des catégories basée sur leur fréquence.
       
       Args:
           column_name: Nom de la colonne
           
       Returns:
           Dictionnaire des importances normalisées
       """
       if column_name not in self.category_frequencies:
           return {}
       
       frequencies = self.category_frequencies[column_name]
       total_freq = sum(frequencies.values())
       
       # Normaliser les fréquences
       importance = {cat: freq / total_freq for cat, freq in frequencies.items()}
       
       return importance


class DummyVariableTransformer(CategoricalTransformer):
   """
   Transformer spécialisé pour les variables dummy existantes.
   
   Cette classe gère spécifiquement les colonnes qui sont déjà
   au format dummy (comme 'dummy:category:value') et les traite
   de manière optimisée.
   """
   
   def __init__(self, 
                dummy_prefix: str = 'dummy',
                separator: str = ':',
                aggregate_strategy: str = 'keep_all',
                name: Optional[str] = None):
       """
       Initialise le transformer pour variables dummy.
       
       Args:
           dummy_prefix: Préfixe des colonnes dummy
           separator: Séparateur dans les noms de colonnes dummy
           aggregate_strategy: Stratégie d'agrégation ('keep_all', 'most_frequent', 'sum')
           name: Nom du transformer
       """
       super().__init__(name=name)
       self.dummy_prefix = dummy_prefix
       self.separator = separator
       self.aggregate_strategy = aggregate_strategy
       
       # Groupes de variables dummy détectés
       self.dummy_groups = {}
       
       self.config.update({
           'dummy_prefix': dummy_prefix,
           'separator': separator,
           'aggregate_strategy': aggregate_strategy
       })
   
   def _auto_detect_features(self, data: pd.DataFrame) -> List[str]:
       """
       Détecte les colonnes dummy variables.
       
       Args:
           data: DataFrame à analyser
           
       Returns:
           Liste des colonnes dummy détectées
       """
       dummy_columns = []
       
       for col in data.columns:
           if str(col).startswith(self.dummy_prefix):
               dummy_columns.append(col)
       
       return dummy_columns
   
   def fit(self, 
           data: pd.DataFrame, 
           feature_columns: Optional[List[str]] = None,
           **kwargs) -> 'DummyVariableTransformer':
       """
       Ajuste le transformer sur les variables dummy.
       
       Args:
           data: DataFrame contenant les variables dummy
           feature_columns: Colonnes dummy à traiter
           **kwargs: Paramètres supplémentaires
           
       Returns:
           Self pour le chaînage
       """
       dummy_columns = self.get_feature_columns(data, feature_columns)
       
       if not dummy_columns:
           raise TransformationError(
               "Aucune variable dummy détectée",
               transformer_name=self.name,
               step="fit"
           )
       
       # Grouper les variables dummy par catégorie
       self._group_dummy_variables(dummy_columns)
       
       # Traiter chaque groupe
       for group_name, group_columns in self.dummy_groups.items():
           self._fit_dummy_group(data[group_columns], group_name, group_columns)
       
       self.is_fitted = True
       return self
   
   def _group_dummy_variables(self, dummy_columns: List[str]):
       """
       Groupe les variables dummy par catégorie.
       
       Args:
           dummy_columns: Liste des colonnes dummy
       """
       groups = {}
       
       for col in dummy_columns:
           # Parser le nom de la colonne dummy
           parts = str(col).split(self.separator)
           if len(parts) >= 2:
               # Le groupe est généralement la partie avant la dernière
               group_key = self.separator.join(parts[:-1])
               if group_key not in groups:
                   groups[group_key] = []
               groups[group_key].append(col)
           else:
               # Colonne dummy simple
               groups[col] = [col]
       
       self.dummy_groups = groups
   
   def _fit_dummy_group(self, 
                       group_data: pd.DataFrame, 
                       group_name: str, 
                       group_columns: List[str]):
       """
       Ajuste le transformer sur un groupe de variables dummy.
       
       Args:
           group_data: Données du groupe
           group_name: Nom du groupe
           group_columns: Colonnes du groupe
       """
       if self.aggregate_strategy == 'keep_all':
           # Garder toutes les colonnes dummy comme features séparées
           for col in group_columns:
               # Les dummy variables sont déjà encodées (0/1)
               self.category_mappings[col] = {0: 0, 1: 1}
               self.vocabulary_sizes[col] = 2
               self.feature_mappings[col] = f"{col}_encoded"
               
       elif self.aggregate_strategy == 'most_frequent':
           # Créer une feature unique basée sur la catégorie la plus fréquente
           # Trouver quelle colonne est active pour chaque ligne
           active_categories = []
           for _, row in group_data.iterrows():
               active_cols = [col for col in group_columns if row[col] == 1]
               if active_cols:
                   active_categories.append(active_cols[0])  # Prendre la première si plusieurs
               else:
                   active_categories.append(None)  # Aucune catégorie active
           
           # Créer le mapping pour cette feature agrégée
           unique_categories = list(set(cat for cat in active_categories if cat is not None))
           category_mapping = {cat: i + 1 for i, cat in enumerate(unique_categories)}
           category_mapping[None] = 0  # Pour les cas où aucune catégorie n'est active
           
           aggregated_name = f"{group_name}_aggregated"
           self.category_mappings[aggregated_name] = category_mapping
           self.vocabulary_sizes[aggregated_name] = len(category_mapping)
           self.feature_mappings[aggregated_name] = f"{aggregated_name}_encoded"
           
           # Stocker les données nécessaires pour la transformation
           self.dummy_groups[aggregated_name] = group_columns