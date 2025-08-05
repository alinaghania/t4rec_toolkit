# transformers/numerical_transformer.py
"""
Transformer pour les données numériques.

Ce module gère la transformation des features numériques simples,
incluant la normalisation, la discrétisation et la gestion
des valeurs manquantes.
"""

from typing import Dict, List, Any, Optional, Union, Tuple
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, KBinsDiscretizer

from ..core.base_transformer import BaseTransformer, TransformationResult
from ..core.exceptions import TransformationError


class NumericalTransformer(BaseTransformer):
    """
    Transformer pour les données numériques.
    
    Cette classe transforme les features numériques simples
    (non-séquentielles) avec différentes stratégies de
    normalisation et discrétisation.
    """
    
    def __init__(self,
                 strategy: str = 'normalize',
                 scaler_type: str = 'standard',
                 n_bins: int = 10,
                 bin_strategy: str = 'quantile',
                 handle_missing: str = 'mean',
                 clip_outliers: bool = False,
                 outlier_method: str = 'iqr',
                 outlier_factor: float = 1.5,
                 name: Optional[str] = None):
        """
        Initialise le transformer numérique.
        
        Args:
            strategy: Stratégie de transformation ('normalize', 'discretize', 'both')
            scaler_type: Type de normalisation ('standard', 'minmax', 'robust')
            n_bins: Nombre de bins pour la discrétisation
            bin_strategy: Stratégie de binning ('uniform', 'quantile', 'kmeans')
            handle_missing: Stratégie pour les valeurs manquantes ('mean', 'median', 'zero', 'drop')
            clip_outliers: Écrêter les outliers
            outlier_method: Méthode de détection des outliers ('iqr', 'zscore')
            outlier_factor: Facteur pour la détection des outliers
            name: Nom du transformer
        """
        super().__init__(name)
        self.strategy = strategy
        self.scaler_type = scaler_type
        self.n_bins = n_bins
        self.bin_strategy = bin_strategy
        self.handle_missing = handle_missing
        self.clip_outliers = clip_outliers
        self.outlier_method = outlier_method
        self.outlier_factor = outlier_factor
        
        # Scalers et discretizers ajustés
        self.scalers = {}
        self.discretizers = {}
        self.missing_values = {}
        self.outlier_bounds = {}
        
        # Validation des paramètres
        self._validate_parameters()
        
        # Configuration
        self.config.update({
            'strategy': strategy,
            'scaler_type': scaler_type,
            'n_bins': n_bins,
            'bin_strategy': bin_strategy,
            'handle_missing': handle_missing,
            'clip_outliers': clip_outliers,
            'outlier_method': outlier_method,
            'outlier_factor': outlier_factor
        })
    
    def _validate_parameters(self):
        """Valide les paramètres du transformer."""
        valid_strategies = ['normalize', 'discretize', 'both']
        if self.strategy not in valid_strategies:
            raise TransformationError(
                f"Stratégie invalide: {self.strategy}. Valides: {valid_strategies}",
                transformer_name=self.name,
                step="init"
            )
        
        valid_scalers = ['standard', 'minmax', 'robust']
        if self.scaler_type not in valid_scalers:
            raise TransformationError(
                f"Type de scaler invalide: {self.scaler_type}. Valides: {valid_scalers}",
                transformer_name=self.name,
                step="init"
            )
        
        valid_bin_strategies = ['uniform', 'quantile', 'kmeans']
        if self.bin_strategy not in valid_bin_strategies:
            raise TransformationError(
                f"Stratégie de binning invalide: {self.bin_strategy}. Valides: {valid_bin_strategies}",
                transformer_name=self.name,
                step="init"
            )
        
        valid_missing_strategies = ['mean', 'median', 'zero', 'drop']
        if self.handle_missing not in valid_missing_strategies:
            raise TransformationError(
                f"Stratégie de gestion des manquants invalide: {self.handle_missing}. Valides: {valid_missing_strategies}",
                transformer_name=self.name,
                step="init"
            )
    
    def fit(self, 
            data: pd.DataFrame, 
            feature_columns: Optional[List[str]] = None,
            **kwargs) -> 'NumericalTransformer':
        """
        Ajuste le transformer sur les données numériques.
        
        Args:
            data: DataFrame contenant les features numériques
            feature_columns: Colonnes numériques à traiter
            **kwargs: Paramètres supplémentaires
            
        Returns:
            Self pour le chaînage
        """
        numerical_columns = self.get_feature_columns(data, feature_columns)
        
        if not numerical_columns:
            raise TransformationError(
                "Aucune colonne numérique détectée",
                transformer_name=self.name,
                step="fit"
            )
        
        # Traiter chaque colonne numérique
        for col in numerical_columns:
            self._fit_column(data[col], col)
            
            # Définir le nom de la feature transformée
            if self.strategy == 'normalize':
                self.feature_mappings[col] = f"{col}_normalized"
            elif self.strategy == 'discretize':
                self.feature_mappings[col] = f"{col}_discretized"
            elif self.strategy == 'both':
                self.feature_mappings[f"{col}_norm"] = f"{col}_normalized"
                self.feature_mappings[f"{col}_disc"] = f"{col}_discretized"
        
        self.is_fitted = True
        return self
    
    def transform(self, data: pd.DataFrame) -> TransformationResult:
        """
        Transforme les features numériques.
        
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
        
        for col in self.scalers.keys():
            if col not in data.columns:
                raise TransformationError(
                    f"Colonne manquante pour la transformation: {col}",
                    transformer_name=self.name,
                    step="transform"
                )
            
            # Transformer la colonne
            results = self._transform_column(data[col], col)
            
            # Stocker les résultats
            for result_name, result_data in results.items():
                transformed_data[result_name] = result_data
                
                # Créer les métadonnées
                is_discretized = 'discretized' in result_name
                feature_info[result_name] = {
                    'original_column': col,
                    'dtype': 'int32' if is_discretized else 'float32',
                    'shape': result_data.shape,
                    'is_sequence': False,
                    'is_numerical': True,
                    'is_discretized': is_discretized,
                    'vocab_size': self.n_bins + 1 if is_discretized else None,
                    'scaler_type': self.scaler_type if not is_discretized else None,
                    'n_bins': self.n_bins if is_discretized else None
                }
            
            original_columns.append(col)
            if self.strategy == 'normalize':
                transformation_steps.append(f"normalize_{col}")
            elif self.strategy == 'discretize':
                transformation_steps.append(f"discretize_{col}")
            elif self.strategy == 'both':
                transformation_steps.extend([f"normalize_{col}", f"discretize_{col}"])
        
        return TransformationResult(
            data=transformed_data,
            feature_info=feature_info,
            config=self.get_config(),
            original_columns=original_columns,
            transformation_steps=transformation_steps
        )
    
    def _auto_detect_features(self, data: pd.DataFrame) -> List[str]:
        """
        Détecte automatiquement les colonnes numériques.
        
        Args:
            data: DataFrame à analyser
            
        Returns:
            Liste des colonnes numériques détectées
        """
        numerical_columns = []
        
        for col in data.columns:
            if self._is_numerical_column(data[col], col):
                numerical_columns.append(col)
        
        return numerical_columns
    
    def _is_numerical_column(self, series: pd.Series, col_name: str) -> bool:
        """
        Détermine si une colonne est numérique (et non catégorielle).
        
        Args:
            series: Série à analyser
            col_name: Nom de la colonne
            
        Returns:
            True si la colonne est numérique
        """
        # Exclure les colonnes qui ressemblent à des identifiants
        if col_name.endswith('_id') or col_name.startswith('dummy'):
            return False
        
        # Vérifier le type de données
        if not pd.api.types.is_numeric_dtype(series):
            return False
        
        # Vérifier la cardinalité pour éviter les features catégorielles
        n_unique = series.nunique()
        n_total = len(series.dropna())
        
        if n_total == 0:
            return False
        
        # Si trop peu de valeurs uniques, probablement catégoriel
        if n_unique < 10 and n_unique / n_total < 0.1:
            return False
        
        # Vérifier si ce sont des entiers séquentiels (IDs)
        if series.dtype in ['int32', 'int64']:
            unique_vals = sorted(series.dropna().unique())
            if len(unique_vals) > 1:
                # Vérifier si c'est une séquence d'entiers
                diffs = np.diff(unique_vals)
                if np.all(diffs == 1) and len(unique_vals) > n_total * 0.8:
                    return False  # Probablement des IDs séquentiels
        
        return True
    
    def _fit_column(self, series: pd.Series, col_name: str):
       """
       Ajuste les transformers sur une colonne numérique.
       
       Args:
           series: Série à traiter
           col_name: Nom de la colonne
       """
       # Calculer la valeur de remplacement pour les manquants
       if self.handle_missing == 'mean':
           fill_value = series.mean()
       elif self.handle_missing == 'median':
           fill_value = series.median()
       elif self.handle_missing == 'zero':
           fill_value = 0.0
       else:  # 'drop' sera géré lors de la transformation
           fill_value = 0.0
       
       self.missing_values[col_name] = fill_value
       
       # Préparer les données pour l'ajustement (sans les valeurs manquantes)
       clean_data = series.dropna()
       if len(clean_data) == 0:
           raise TransformationError(
               f"Aucune valeur non-manquante dans la colonne {col_name}",
               transformer_name=self.name,
               step="fit"
           )
       
       # Calculer les bornes pour les outliers si nécessaire
       if self.clip_outliers:
           self.outlier_bounds[col_name] = self._calculate_outlier_bounds(clean_data)
       
       # Appliquer le clipping des outliers pour l'ajustement
       if self.clip_outliers and col_name in self.outlier_bounds:
           lower_bound, upper_bound = self.outlier_bounds[col_name]
           clean_data = np.clip(clean_data, lower_bound, upper_bound)
       
       # Ajuster le scaler si nécessaire
       if self.strategy in ['normalize', 'both']:
           scaler = self._create_scaler()
           scaler.fit(clean_data.values.reshape(-1, 1))
           self.scalers[col_name] = scaler
       
       # Ajuster le discretizer si nécessaire
       if self.strategy in ['discretize', 'both']:
           discretizer = KBinsDiscretizer(
               n_bins=self.n_bins,
               encode='ordinal',
               strategy=self.bin_strategy
           )
           discretizer.fit(clean_data.values.reshape(-1, 1))
           self.discretizers[col_name] = discretizer
   
    def _create_scaler(self):
        """
        Crée un scaler selon le type configuré.
        
        Returns:
            Instance du scaler
        """
        if self.scaler_type == 'standard':
            return StandardScaler()
        elif self.scaler_type == 'minmax':
            return MinMaxScaler()
        elif self.scaler_type == 'robust':
            return RobustScaler()
        else:
            raise TransformationError(
                f"Type de scaler inconnu: {self.scaler_type}",
                transformer_name=self.name,
                step="create_scaler"
            )
   
    def _calculate_outlier_bounds(self, data: pd.Series) -> Tuple[float, float]:
        """
        Calcule les bornes pour l'écrêtage des outliers.
        
        Args:
            data: Données pour calculer les bornes
            
        Returns:
            Tuple (borne_inférieure, borne_supérieure)
        """
        if self.outlier_method == 'iqr':
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - self.outlier_factor * IQR
            upper_bound = Q3 + self.outlier_factor * IQR
            
        elif self.outlier_method == 'zscore':
            mean = data.mean()
            std = data.std()
            lower_bound = mean - self.outlier_factor * std
            upper_bound = mean + self.outlier_factor * std
            
        else:
            # Fallback: utiliser les percentiles
            lower_bound = data.quantile(0.01)
            upper_bound = data.quantile(0.99)
        
        return float(lower_bound), float(upper_bound)
   
    def _transform_column(self, series: pd.Series, col_name: str) -> Dict[str, np.ndarray]:
        """
        Transforme une colonne numérique.
        
        Args:
            series: Série à transformer
            col_name: Nom de la colonne
            
        Returns:
            Dictionnaire des résultats transformés
        """
        results = {}
        
        # Gérer les valeurs manquantes
        if self.handle_missing == 'drop':
            # Pour 'drop', on remplace par la valeur moyenne temporairement
            # Dans un vrai pipeline, il faudrait supprimer les lignes
            filled_series = series.fillna(self.missing_values[col_name])
        else:
            filled_series = series.fillna(self.missing_values[col_name])
        
        # Appliquer le clipping des outliers
        if self.clip_outliers and col_name in self.outlier_bounds:
            lower_bound, upper_bound = self.outlier_bounds[col_name]
            filled_series = np.clip(filled_series, lower_bound, upper_bound)
        
        # Normalisation
        if self.strategy in ['normalize', 'both']:
            scaler = self.scalers[col_name]
            normalized = scaler.transform(filled_series.values.reshape(-1, 1)).flatten()
            
            if self.strategy == 'normalize':
                results[self.feature_mappings[col_name]] = normalized.astype(np.float32)
            else:  # 'both'
                results[self.feature_mappings[f"{col_name}_norm"]] = normalized.astype(np.float32)
        
        # Discrétisation
        if self.strategy in ['discretize', 'both']:
            discretizer = self.discretizers[col_name]
            discretized = discretizer.transform(filled_series.values.reshape(-1, 1)).flatten()
            # Ajouter 1 pour éviter l'index 0 (réservé pour padding/unknown)
            discretized = discretized + 1
            
            if self.strategy == 'discretize':
                results[self.feature_mappings[col_name]] = discretized.astype(np.int32)
            else:  # 'both'
                results[self.feature_mappings[f"{col_name}_disc"]] = discretized.astype(np.int32)
        
        return results
   
    def get_scaler(self, column_name: str):
        """
        Retourne le scaler ajusté pour une colonne.
        
        Args:
            column_name: Nom de la colonne
            
        Returns:
            Scaler ajusté
        """
        return self.scalers.get(column_name)
   
    def get_discretizer(self, column_name: str):
        """
        Retourne le discretizer ajusté pour une colonne.
        
        Args:
            column_name: Nom de la colonne
            
        Returns:
            Discretizer ajusté
        """
        return self.discretizers.get(column_name)
   
    def get_feature_bounds(self, column_name: str) -> Optional[Tuple[float, float]]:
        """
        Retourne les bornes calculées pour une colonne.
        
        Args:
            column_name: Nom de la colonne
            
        Returns:
            Tuple des bornes ou None
        """
        return self.outlier_bounds.get(column_name)
    
    def inverse_transform_normalized(self, 
                                    normalized_values: np.ndarray, 
                                    column_name: str) -> np.ndarray:
        """
        Inverse la transformation de normalisation.
        
        Args:
            normalized_values: Valeurs normalisées
            column_name: Nom de la colonne originale
            
        Returns:
            Valeurs dans l'échelle originale
        """
        if column_name not in self.scalers:
            raise ValueError(f"Pas de scaler pour la colonne: {column_name}")
        
        scaler = self.scalers[column_name]
        return scaler.inverse_transform(normalized_values.reshape(-1, 1)).flatten()
    
    def get_bin_edges(self, column_name: str) -> Optional[np.ndarray]:
        """
        Retourne les bords des bins pour une colonne discrétisée.
        
        Args:
            column_name: Nom de la colonne
            
        Returns:
            Array des bords des bins ou None
        """
        if column_name not in self.discretizers:
            return None
        
        discretizer = self.discretizers[column_name]
        return discretizer.bin_edges_[0]  # Premier (et seul) feature
    
    def get_statistics(self, column_name: str) -> Dict[str, Any]:
        """
        Retourne les statistiques calculées pour une colonne.
        
        Args:
            column_name: Nom de la colonne
            
        Returns:
            Dictionnaire des statistiques
        """
        stats = {
            'missing_fill_value': self.missing_values.get(column_name),
            'has_scaler': column_name in self.scalers,
            'has_discretizer': column_name in self.discretizers,
            'outlier_bounds': self.outlier_bounds.get(column_name)
        }
        
        # Ajouter les statistiques du scaler
        if column_name in self.scalers:
            scaler = self.scalers[column_name]
            if hasattr(scaler, 'mean_'):
                stats['scaler_mean'] = float(scaler.mean_[0])
            if hasattr(scaler, 'scale_'):
                stats['scaler_scale'] = float(scaler.scale_[0])
            if hasattr(scaler, 'center_'):
                stats['scaler_center'] = float(scaler.center_[0])
        
        # Ajouter les informations du discretizer
        if column_name in self.discretizers:
            discretizer = self.discretizers[column_name]
            stats['n_bins'] = self.n_bins
            stats['bin_strategy'] = self.bin_strategy
            if hasattr(discretizer, 'bin_edges_'):
                stats['bin_edges'] = discretizer.bin_edges_[0].tolist()
        
        return stats


class CombinedNumericalTransformer(BaseTransformer):
   """
   Transformer qui combine plusieurs stratégies numériques.
   
   Cette classe permet d'appliquer différentes transformations
   à différentes colonnes selon leurs caractéristiques.
   """
   
   def __init__(self, 
                column_strategies: Optional[Dict[str, Dict[str, Any]]] = None,
                default_strategy: Dict[str, Any] = None,
                name: Optional[str] = None):
       """
       Initialise le transformer combiné.
       
       Args:
           column_strategies: Stratégies spécifiques par colonne
           default_strategy: Stratégie par défaut
           name: Nom du transformer
       """
       super().__init__(name)
       self.column_strategies = column_strategies or {}
       self.default_strategy = default_strategy or {'strategy': 'normalize'}
       
       # Transformers individuels
       self.transformers = {}
       
       self.config.update({
           'column_strategies': self.column_strategies,
           'default_strategy': self.default_strategy
       })
   
   def fit(self, 
           data: pd.DataFrame, 
           feature_columns: Optional[List[str]] = None,
           **kwargs) -> 'CombinedNumericalTransformer':
       """
       Ajuste les transformers selon les stratégies définies.
       
       Args:
           data: DataFrame contenant les features numériques
           feature_columns: Colonnes numériques à traiter
           **kwargs: Paramètres supplémentaires
           
       Returns:
           Self pour le chaînage
       """
       numerical_columns = self.get_feature_columns(data, feature_columns)
       
       for col in numerical_columns:
           # Déterminer la stratégie pour cette colonne
           if col in self.column_strategies:
               strategy_config = self.column_strategies[col]
           else:
               strategy_config = self.default_strategy
           
           # Créer et ajuster le transformer pour cette colonne
           transformer = NumericalTransformer(**strategy_config, name=f"{self.name}_{col}")
           transformer.fit(data, feature_columns=[col])
           
           self.transformers[col] = transformer
           
           # Copier les mappings
           for orig_col, mapped_col in transformer.feature_mappings.items():
               self.feature_mappings[orig_col] = mapped_col
       
       self.is_fitted = True
       return self
   
   def transform(self, data: pd.DataFrame) -> TransformationResult:
       """
       Applique toutes les transformations configurées.
       
       Args:
           data: DataFrame à transformer
           
       Returns:
           Résultat de la transformation combinée
       """
       self._check_fitted()
       
       combined_data = {}
       combined_feature_info = {}
       combined_columns = []
       combined_steps = []
       
       for col, transformer in self.transformers.items():
           result = transformer.transform(data)
           
           # Combiner les résultats
           combined_data.update(result.data)
           combined_feature_info.update(result.feature_info)
           combined_columns.extend(result.original_columns)
           combined_steps.extend(result.transformation_steps)
       
       return TransformationResult(
           data=combined_data,
           feature_info=combined_feature_info,
           config=self.get_config(),
           original_columns=list(set(combined_columns)),
           transformation_steps=combined_steps
       )
   
   def _auto_detect_features(self, data: pd.DataFrame) -> List[str]:
       """Utilise la détection du transformer numérique standard."""
       temp_transformer = NumericalTransformer()
       return temp_transformer._auto_detect_features(data)
   
   def get_transformer(self, column_name: str) -> Optional[NumericalTransformer]:
       """
       Retourne le transformer pour une colonne spécifique.
       
       Args:
           column_name: Nom de la colonne
           
       Returns:
           Transformer pour cette colonne ou None
       """
       return self.transformers.get(column_name)