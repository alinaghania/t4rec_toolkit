# core/base_transformer.py
"""
Interface de base pour tous les transformers.

Ce module définit l'interface commune que tous les transformers
doivent implémenter, ainsi que les structures de données
pour les résultats de transformation.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
import pandas as pd
import numpy as np

from .exceptions import TransformationError


@dataclass
class TransformationResult:
    """
    Résultat d'une transformation de données.
    
    Cette classe encapsule tous les éléments produits par
    une transformation : données transformées, métadonnées,
    et informations de configuration.
    """
    
    data: Dict[str, np.ndarray]
    """Données transformées, organisées par nom de feature"""
    
    feature_info: Dict[str, Dict[str, Any]]
    """Métadonnées sur chaque feature transformée"""
    
    config: Dict[str, Any]
    """Configuration utilisée pour la transformation"""
    
    original_columns: List[str]
    """Liste des colonnes originales utilisées"""
    
    transformation_steps: List[str]
    """Liste des étapes de transformation appliquées"""
    
    def get_feature_names(self) -> List[str]:
        """
        Retourne la liste des noms de features transformées.
        
        Returns:
            Liste des noms de features
        """
        return list(self.data.keys())
    
    def get_feature_info(self, feature_name: str) -> Dict[str, Any]:
        """
        Retourne les métadonnées d'une feature spécifique.
        
        Args:
            feature_name: Nom de la feature
            
        Returns:
            Dictionnaire des métadonnées
            
        Raises:
            KeyError: Si la feature n'existe pas
        """
        if feature_name not in self.feature_info:
            raise KeyError(f"Feature '{feature_name}' non trouvée")
        return self.feature_info[feature_name]
    
    def get_shape_info(self) -> Dict[str, tuple]:
        """
        Retourne les dimensions de chaque feature.
        
        Returns:
            Dictionnaire avec les shapes de chaque feature
        """
        return {name: data.shape for name, data in self.data.items()}


class BaseTransformer(ABC):
    """
    Classe de base abstraite pour tous les transformers.
    
    Cette classe définit l'interface commune que tous les
    transformers doivent implémenter pour assurer la
    cohérence et la composabilité.
    """
    
    def __init__(self, name: Optional[str] = None):
        """
        Initialise le transformer.
        
        Args:
            name: Nom du transformer (optionnel)
        """
        self.name = name or self.__class__.__name__
        self.is_fitted = False
        self.config = {}
        self.feature_mappings = {}
        
    @abstractmethod
    def fit(self, 
            data: pd.DataFrame, 
            feature_columns: Optional[List[str]] = None,
            **kwargs) -> 'BaseTransformer':
        """
        Ajuste le transformer sur les données d'entraînement.
        
        Args:
            data: DataFrame d'entraînement
            feature_columns: Colonnes à utiliser (toutes si None)
            **kwargs: Paramètres supplémentaires
            
        Returns:
            Self pour le chaînage
        """
        pass
    
    @abstractmethod
    def transform(self, data: pd.DataFrame) -> TransformationResult:
        """
        Transforme les données selon les paramètres ajustés.
        
        Args:
            data: DataFrame à transformer
            
        Returns:
            Résultat de la transformation
            
        Raises:
            TransformationError: Si le transformer n'est pas ajusté
        """
        pass
    
    def fit_transform(self, 
                     data: pd.DataFrame, 
                     feature_columns: Optional[List[str]] = None,
                     **kwargs) -> TransformationResult:
        """
        Ajuste puis transforme les données en une seule étape.
        
        Args:
            data: DataFrame à traiter
            feature_columns: Colonnes à utiliser (toutes si None)
            **kwargs: Paramètres supplémentaires
            
        Returns:
            Résultat de la transformation
        """
        return self.fit(data, feature_columns, **kwargs).transform(data)
    
    def get_feature_columns(self, 
                           data: pd.DataFrame, 
                           feature_columns: Optional[List[str]] = None) -> List[str]:
        """
        Détermine les colonnes de features à utiliser.
        
        Args:
            data: DataFrame source
            feature_columns: Colonnes spécifiées (optionnel)
            
        Returns:
            Liste des colonnes à traiter
        """
        if feature_columns is None:
            return self._auto_detect_features(data)
        
        # Valider que les colonnes existent
        missing_cols = set(feature_columns) - set(data.columns)
        if missing_cols:
            raise TransformationError(
                f"Colonnes manquantes: {missing_cols}",
                transformer_name=self.name,
                step="feature_selection"
            )
        
        return feature_columns
    
    @abstractmethod
    def _auto_detect_features(self, data: pd.DataFrame) -> List[str]:
        """
        Détecte automatiquement les colonnes de features appropriées.
        
        Args:
            data: DataFrame à analyser
            
        Returns:
            Liste des colonnes détectées
        """
        pass
    
    def _check_fitted(self):
        """
        Vérifie que le transformer a été ajusté.
        
        Raises:
            TransformationError: Si le transformer n'est pas ajusté
        """
        if not self.is_fitted:
            raise TransformationError(
                f"Transformer '{self.name}' doit être ajusté avant transformation",
                transformer_name=self.name,
                step="check_fitted"
            )
    
    def get_config(self) -> Dict[str, Any]:
        """
        Retourne la configuration du transformer.
        
        Returns:
            Dictionnaire de configuration
        """
        return self.config.copy()
    
    def set_config(self, **config):
        """
        Met à jour la configuration du transformer.
        
        Args:
            **config: Nouveaux paramètres de configuration
        """
        self.config.update(config)
    
    def get_feature_mapping(self, original_column: str) -> Optional[str]:
        """
        Retourne le nom de la feature transformée pour une colonne originale.
        
        Args:
            original_column: Nom de la colonne originale
            
        Returns:
            Nom de la feature transformée ou None
        """
        return self.feature_mappings.get(original_column)
    
    def __repr__(self) -> str:
        """Représentation string du transformer."""
        status = "fitted" if self.is_fitted else "not fitted"
        return f"{self.name}({status})"


class CompositeTransformer(BaseTransformer):
    """
    Transformer composite qui combine plusieurs transformers.
    
    Cette classe permet de composer plusieurs transformers
    pour créer des pipelines de transformation complexes.
    """
    
    def __init__(self, 
                 transformers: List[BaseTransformer], 
                 name: Optional[str] = None):
        """
        Initialise le transformer composite.
        
        Args:
            transformers: Liste des transformers à composer
            name: Nom du transformer composite
        """
        super().__init__(name)
        self.transformers = transformers
        
    def fit(self, 
            data: pd.DataFrame, 
            feature_columns: Optional[List[str]] = None,
            **kwargs) -> 'CompositeTransformer':
        """
        Ajuste tous les transformers en séquence.
        
        Args:
            data: DataFrame d'entraînement
            feature_columns: Colonnes à utiliser
            **kwargs: Paramètres supplémentaires
            
        Returns:
            Self pour le chaînage
        """
        current_data = data
        
        for transformer in self.transformers:
            transformer.fit(current_data, feature_columns, **kwargs)
            # Pour le prochain transformer, utiliser toutes les colonnes
            # car les colonnes ont pu être transformées
            feature_columns = None
            
        self.is_fitted = True
        return self
    
    def transform(self, data: pd.DataFrame) -> TransformationResult:
        """
        Applique tous les transformers en séquence.
        
        Args:
            data: DataFrame à transformer
            
        Returns:
            Résultat de la transformation composite
        """
        self._check_fitted()
        
        # Combiner tous les résultats
        combined_data = {}
        combined_feature_info = {}
        combined_steps = []
        combined_columns = []
        
        current_data = data
        
        for transformer in self.transformers:
            result = transformer.transform(current_data)
            
            # Merger les résultats
            combined_data.update(result.data)
            combined_feature_info.update(result.feature_info)
            combined_steps.extend(result.transformation_steps)
            combined_columns.extend(result.original_columns)
        
        return TransformationResult(
            data=combined_data,
            feature_info=combined_feature_info,
            config=self.get_config(),
            original_columns=list(set(combined_columns)),
            transformation_steps=combined_steps
        )
    
    def _auto_detect_features(self, data: pd.DataFrame) -> List[str]:
        """
        Utilise la détection du premier transformer.
        
        Args:
            data: DataFrame à analyser
            
        Returns:
            Liste des colonnes détectées
        """
        if self.transformers:
            return self.transformers[0]._auto_detect_features(data)
        return list(data.columns)
    