# adapters/t4rec_adapter.py
"""
Adaptateur pour l'intégration avec T4Rec.

Ce module fournit les fonctionnalités nécessaires pour convertir
les données transformées au format requis par T4Rec et générer
les schémas appropriés pour les modèles de recommandation.
"""

from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod


class T4RecAdapter:
    """
    Adaptateur pour convertir les données vers le format T4Rec.
    
    Cette classe gère la conversion des données transformées vers
    les structures de données et schémas requis par T4Rec pour
    l'entraînement des modèles de recommandation.
    """
    
    def __init__(self, max_sequence_length: int = 20):
        """
        Initialise l'adaptateur T4Rec.
        
        Args:
            max_sequence_length: Longueur maximale des séquences pour T4Rec
        """
        self.max_sequence_length = max_sequence_length
        self.feature_specs = []
        self.vocab_sizes = {}
        
    def create_schema(self, 
                     feature_info: Dict[str, Any], 
                     target_column: str) -> Dict[str, Any]:
        """
        Crée un schéma T4Rec à partir des informations de features.
        
        Args:
            feature_info: Dictionnaire contenant les métadonnées des features
            target_column: Nom de la colonne cible
            
        Returns:
            Dictionnaire représentant le schéma T4Rec
        """
        schema = {
            "feature_specs": [],
            "target_column": target_column,
            "sequence_length": self.max_sequence_length
        }
        
        # Traiter les features séquentielles
        if "sequence_features" in feature_info:
            for feature_name, feature_data in feature_info["sequence_features"].items():
                spec = {
                    "name": feature_name,
                    "dtype": "int32",
                    "is_sequence": True,
                    "vocab_size": feature_data.get("vocab_size", 1000),
                    "sequence_length": self.max_sequence_length
                }
                schema["feature_specs"].append(spec)
                
        # Traiter les features catégorielles
        if "categorical_features" in feature_info:
            for feature_name, feature_data in feature_info["categorical_features"].items():
                spec = {
                    "name": feature_name,
                    "dtype": "int32", 
                    "is_sequence": False,
                    "vocab_size": feature_data.get("vocab_size", 100)
                }
                schema["feature_specs"].append(spec)
                
        # Traiter les features numériques
        if "numerical_features" in feature_info:
            for feature_name, feature_data in feature_info["numerical_features"].items():
                spec = {
                    "name": feature_name,
                    "dtype": "float32",
                    "is_sequence": False,
                    "is_continuous": True
                }
                schema["feature_specs"].append(spec)
                
        # Ajouter la spécification de la cible
        if target_column in feature_info.get("target_info", {}):
            target_spec = {
                "name": target_column,
                "dtype": "int32",
                "is_target": True,
                "vocab_size": feature_info["target_info"][target_column].get("vocab_size", 1000)
            }
            schema["feature_specs"].append(target_spec)
            
        return schema
    
    def prepare_tabular_features(self, 
                               transformed_data: Dict[str, np.ndarray],
                               schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prépare les données pour TabularSequenceFeatures de T4Rec.
        
        Args:
            transformed_data: Données transformées par les transformers
            schema: Schéma T4Rec généré
            
        Returns:
            Dictionnaire formaté pour TabularSequenceFeatures
        """
        tabular_data = {}
        
        for spec in schema["feature_specs"]:
            feature_name = spec["name"]
            
            if feature_name in transformed_data:
                data = transformed_data[feature_name]
                
                # Convertir selon le type de feature
                if spec.get("is_sequence", False):
                    # Assurer que les séquences ont la bonne longueur
                    if data.ndim == 2:
                        # Tronquer ou padder si nécessaire
                        if data.shape[1] > self.max_sequence_length:
                            data = data[:, :self.max_sequence_length]
                        elif data.shape[1] < self.max_sequence_length:
                            padding = np.zeros((data.shape[0], 
                                              self.max_sequence_length - data.shape[1]),
                                             dtype=data.dtype)
                            data = np.concatenate([data, padding], axis=1)
                            
                # Convertir au bon type de données
                if spec["dtype"] == "int32":
                    data = data.astype(np.int32)
                elif spec["dtype"] == "float32":
                    data = data.astype(np.float32)
                    
                tabular_data[feature_name] = data
                
        return tabular_data
    
    def create_t4rec_dataset(self,
                           tabular_data: Dict[str, np.ndarray],
                           target_column: str) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        """
        Crée un dataset au format T4Rec.
        
        Args:
            tabular_data: Données tabulaires préparées
            target_column: Nom de la colonne cible
            
        Returns:
            Tuple contenant les features et les targets séparés
        """
        features = {}
        targets = None
        
        for feature_name, data in tabular_data.items():
            if feature_name == target_column:
                targets = data
            else:
                features[feature_name] = data
                
        if targets is None:
            raise ValueError(f"Colonne cible '{target_column}' non trouvée dans les données")
            
        return features, targets
    
    def validate_schema_compatibility(self, schema: Dict[str, Any]) -> bool:
        """
        Valide la compatibilité du schéma avec T4Rec.
        
        Args:
            schema: Schéma à valider
            
        Returns:
            True si le schéma est compatible, False sinon
        """
        required_fields = ["feature_specs", "target_column"]
        
        # Vérifier les champs requis
        for field in required_fields:
            if field not in schema:
                return False
                
        # Vérifier que chaque spec a les champs nécessaires
        for spec in schema["feature_specs"]:
            if "name" not in spec or "dtype" not in spec:
                return False
                
        return True
    
    def get_feature_cardinalities(self, transformed_data: Dict[str, np.ndarray]) -> Dict[str, int]:
        """
        Calcule les cardinalités des features catégorielles.
        
        Args:
            transformed_data: Données transformées
            
        Returns:
            Dictionnaire des cardinalités par feature
        """
        cardinalities = {}
        
        for feature_name, data in transformed_data.items():
            if data.dtype in [np.int32, np.int64]:
                # Pour les features catégorielles, calculer le nombre de valeurs uniques
                unique_values = np.unique(data)
                cardinalities[feature_name] = len(unique_values)
                
        return cardinalities