# transformers/sequence_transformer.py
"""
Transformer pour les données séquentielles.

Ce module gère la transformation des colonnes contenant des séquences
au format string vers des arrays numériques encodés, avec gestion
du padding, troncature et vocabulaire.
"""

from typing import Dict, List, Any, Optional, Union, Tuple
import pandas as pd
import numpy as np
import json
import ast
import re
from collections import Counter

from ..core.base_transformer import BaseTransformer, TransformationResult
from ..core.exceptions import TransformationError
from ..core.validator import SequenceValidator


class SequenceTransformer(BaseTransformer):
    """
    Transformer pour les données séquentielles.
    
    Cette classe transforme les colonnes contenant des séquences
    (au format string comme "[0.1, 0.2, 0.3]") en arrays numériques
    encodés prêts pour les modèles de recommandation.
    """
    
    def __init__(self, 
                 max_sequence_length: int = 20,
                 vocab_size: int = 10000,
                 padding_value: int = 0,
                 truncation_strategy: str = 'right',
                 encoding_strategy: str = 'quantile',
                 name: Optional[str] = None):
        """
        Initialise le transformer de séquences.
        
        Args:
            max_sequence_length: Longueur maximale des séquences
            vocab_size: Taille du vocabulaire pour l'encodage
            padding_value: Valeur utilisée pour le padding
            truncation_strategy: Stratégie de troncature ('left', 'right')
            encoding_strategy: Stratégie d'encodage ('quantile', 'uniform', 'minmax')
            name: Nom du transformer
        """
        super().__init__(name)
        self.max_sequence_length = max_sequence_length
        self.vocab_size = vocab_size
        self.padding_value = padding_value
        self.truncation_strategy = truncation_strategy
        self.encoding_strategy = encoding_strategy
        
        # Paramètres ajustés pendant le fit
        self.encoders = {}
        self.sequence_stats = {}
        self.validator = SequenceValidator(expected_length=max_sequence_length)
        
        # Configuration
        self.config.update({
            'max_sequence_length': max_sequence_length,
            'vocab_size': vocab_size,
            'padding_value': padding_value,
            'truncation_strategy': truncation_strategy,
            'encoding_strategy': encoding_strategy
        })
    
    def fit(self, 
            data: pd.DataFrame, 
            feature_columns: Optional[List[str]] = None,
            **kwargs) -> 'SequenceTransformer':
        """
        Ajuste le transformer sur les données de séquences.
        
        Args:
            data: DataFrame contenant les séquences
            feature_columns: Colonnes de séquences à traiter
            **kwargs: Paramètres supplémentaires
            
        Returns:
            Self pour le chaînage
        """
        sequence_columns = self.get_feature_columns(data, feature_columns)
        
        if not sequence_columns:
            raise TransformationError(
                "Aucune colonne de séquence détectée",
                transformer_name=self.name,
                step="fit"
            )
        
        # Collecter toutes les valeurs pour calculer les statistiques globales
        all_values = []
        column_stats = {}
        
        for col in sequence_columns:
            col_values = self._extract_all_values_from_column(data[col])
            all_values.extend(col_values)
            
            column_stats[col] = {
                'n_sequences': data[col].notna().sum(),
                'n_values': len(col_values),
                'value_range': (min(col_values), max(col_values)) if col_values else (0, 0)
            }
        
        # Calculer les statistiques globales
        if all_values:
            self.sequence_stats = {
                'global_min': min(all_values),
                'global_max': max(all_values),
                'global_mean': np.mean(all_values),
                'global_std': np.std(all_values),
                'n_total_values': len(all_values)
            }
        else:
            raise TransformationError(
                "Aucune valeur numérique extraite des séquences",
                transformer_name=self.name,
                step="fit"
            )
        
        # Créer les encoders pour chaque colonne
        for col in sequence_columns:
            col_values = self._extract_all_values_from_column(data[col])
            encoder = self._create_encoder(col_values, col)
            self.encoders[col] = encoder
            self.feature_mappings[col] = f"{col}_encoded"
        
        self.sequence_stats['column_stats'] = column_stats
        self.is_fitted = True
        
        return self
    
    def transform(self, data: pd.DataFrame) -> TransformationResult:
        """
        Transforme les séquences en arrays encodés.
        
        Args:
            data: DataFrame contenant les séquences à transformer
            
        Returns:
            Résultat de la transformation
        """
        self._check_fitted()
        
        transformed_data = {}
        feature_info = {}
        original_columns = []
        transformation_steps = []
        
        for col, encoder in self.encoders.items():
            if col not in data.columns:
                raise TransformationError(
                    f"Colonne manquante pour la transformation: {col}",
                    transformer_name=self.name,
                    step="transform"
                )
            
            # Transformer la colonne
            encoded_sequences = self._transform_column(data[col], encoder, col)
            
            # Stocker les résultats
            feature_name = self.feature_mappings[col]
            transformed_data[feature_name] = encoded_sequences
            
            feature_info[feature_name] = {
                'original_column': col,
                'dtype': 'int32',
                'shape': encoded_sequences.shape,
                'vocab_size': self.vocab_size,
                'is_sequence': True,
                'sequence_length': self.max_sequence_length,
                'encoding_strategy': self.encoding_strategy,
                'padding_value': self.padding_value
            }
            
            original_columns.append(col)
            transformation_steps.append(f"encode_sequence_{col}")
        
        return TransformationResult(
            data=transformed_data,
            feature_info=feature_info,
            config=self.get_config(),
            original_columns=original_columns,
            transformation_steps=transformation_steps
        )
    
    def _auto_detect_features(self, data: pd.DataFrame) -> List[str]:
        """
        Détecte automatiquement les colonnes contenant des séquences.
        
        Args:
            data: DataFrame à analyser
            
        Returns:
            Liste des colonnes de séquences détectées
        """
        sequence_columns = []
        
        for col in data.columns:
            if self._is_sequence_column(data[col]):
                sequence_columns.append(col)
        
        return sequence_columns
    
    def _is_sequence_column(self, series: pd.Series) -> bool:
        """
        Détermine si une colonne contient des séquences.
        
        Args:
            series: Série à analyser
            
        Returns:
            True si la colonne contient des séquences
        """
        # Échantillon pour l'analyse
        sample = series.dropna().head(20)
        if len(sample) == 0:
            return False
        
        sequence_count = 0
        for value in sample:
            if self._looks_like_sequence(str(value)):
                sequence_count += 1
        
        # Si plus de 80% des valeurs ressemblent à des séquences
        return sequence_count / len(sample) > 0.8
    
    def _looks_like_sequence(self, value_str: str) -> bool:
        """
        Vérifie si une string ressemble à une séquence.
        
        Args:
            value_str: String à vérifier
            
        Returns:
            True si ça ressemble à une séquence
        """
        value_str = value_str.strip()
        
        # Vérifier le format de base [...]
        if not (value_str.startswith('[') and value_str.endswith(']')):
            return False
        
        # Vérifier qu'il y a du contenu numérique
        content = value_str[1:-1].strip()
        if not content:
            return False
        
        # Vérifier la présence de nombres et virgules
        return bool(re.search(r'[\d\.\-\+e]', content))
    
    def _parse_sequence_string(self, seq_str: str) -> Optional[List[float]]:
        """
        Parse une string de séquence en liste de nombres.
        
        Args:
            seq_str: String à parser
            
        Returns:
            Liste de nombres ou None si échec
        """
        if pd.isna(seq_str):
            return None
        
        seq_str = str(seq_str).strip()
        
        try:
            # Méthode 1: JSON
            return json.loads(seq_str)
        except:
            try:
                # Méthode 2: ast.literal_eval
                return ast.literal_eval(seq_str)
            except:
                try:
                    # Méthode 3: parsing manuel
                    content = seq_str.strip('[]').strip()
                    if not content:
                        return []
                    
                    values = []
                    for item in content.split(','):
                        item = item.strip()
                        if item:
                            values.append(float(item))
                    return values
                except:
                    return None
    
    def _extract_all_values_from_column(self, series: pd.Series) -> List[float]:
        """
        Extrait toutes les valeurs numériques d'une colonne de séquences.
        
        Args:
            series: Série contenant les séquences
            
        Returns:
            Liste de toutes les valeurs numériques
        """
        all_values = []
        
        for value in series.dropna():
            parsed_seq = self._parse_sequence_string(value)
            if parsed_seq:
                all_values.extend(parsed_seq)
        
        return all_values
    
    def _create_encoder(self, values: List[float], column_name: str) -> Dict[str, Any]:
        """
        Crée un encodeur pour mapper les valeurs continues vers des entiers.
        
        Args:
            values: Liste des valeurs à encoder
            column_name: Nom de la colonne
            
        Returns:
            Dictionnaire contenant l'encodeur
        """
        if not values:
            return {
                'type': 'empty',
                'mapping': {},
                'default_value': self.padding_value
            }
        
        values_array = np.array(values)
        
        if self.encoding_strategy == 'quantile':
            # Encodage basé sur les quantiles
            quantiles = np.linspace(0, 1, self.vocab_size - 1)  # -1 pour le padding
            bins = np.quantile(values_array, quantiles)
            bins = np.unique(bins)  # Enlever les doublons
            
            encoder = {
                'type': 'quantile',
                'bins': bins,
                'n_bins': len(bins),
                'default_value': self.padding_value
            }
            
        elif self.encoding_strategy == 'uniform':
            # Encodage uniforme
            min_val, max_val = values_array.min(), values_array.max()
            if min_val == max_val:
                # Valeur constante
                encoder = {
                    'type': 'constant',
                    'value': min_val,
                    'encoded_value': 1,  # Éviter le padding_value
                    'default_value': self.padding_value
                }
            else:
                bins = np.linspace(min_val, max_val, self.vocab_size - 1)
                encoder = {
                    'type': 'uniform',
                    'bins': bins,
                    'min_val': min_val,
                    'max_val': max_val,
                    'default_value': self.padding_value
                }
                
        elif self.encoding_strategy == 'minmax':
            # Encodage min-max normalisé puis discrétisé
            min_val, max_val = values_array.min(), values_array.max()
            if min_val == max_val:
                encoder = {
                    'type': 'constant',
                    'value': min_val,
                    'encoded_value': 1,
                    'default_value': self.padding_value
                }
            else:
                encoder = {
                    'type': 'minmax',
                    'min_val': min_val,
                    'max_val': max_val,
                    'vocab_size': self.vocab_size,
                    'default_value': self.padding_value
                }
        
        else:
            raise TransformationError(
                f"Stratégie d'encodage inconnue: {self.encoding_strategy}",
                transformer_name=self.name,
                step="create_encoder"
            )
        
        return encoder
    
    def _encode_value(self, value: float, encoder: Dict[str, Any]) -> int:
        """
        Encode une valeur selon l'encodeur fourni.
        
        Args:
            value: Valeur à encoder
            encoder: Encodeur à utiliser
            
        Returns:
            Valeur encodée
        """
        if pd.isna(value):
            return encoder['default_value']
        
        encoder_type = encoder['type']
        
        if encoder_type == 'empty':
            return encoder['default_value']
        
        elif encoder_type == 'constant':
            return encoder['encoded_value']
        
        elif encoder_type == 'quantile':
            bins = encoder['bins']
            # np.digitize retourne des indices 1-based, on ajoute 1 pour éviter le padding
            encoded = np.digitize(value, bins) + 1
            return min(encoded, self.vocab_size - 1)
        
        elif encoder_type == 'uniform':
            bins = encoder['bins']
            encoded = np.digitize(value, bins) + 1
            return min(encoded, self.vocab_size - 1)
        
        elif encoder_type == 'minmax':
            min_val = encoder['min_val']
            max_val = encoder['max_val']
            # Normaliser puis discrétiser
            normalized = (value - min_val) / (max_val - min_val)
            normalized = np.clip(normalized, 0, 1)
            encoded = int(normalized * (self.vocab_size - 2)) + 1  # +1 pour éviter padding
            return min(encoded, self.vocab_size - 1)
        
        else:
            return encoder['default_value']
    
    def _transform_column(self, 
                         series: pd.Series, 
                         encoder: Dict[str, Any], 
                         column_name: str) -> np.ndarray:
        """
        Transforme une colonne de séquences.
        
        Args:
            series: Série à transformer
            encoder: Encodeur à utiliser
            column_name: Nom de la colonne
            
        Returns:
            Array numpy des séquences encodées
        """
        n_samples = len(series)
        encoded_sequences = np.full(
            (n_samples, self.max_sequence_length), 
            self.padding_value, 
            dtype=np.int32
        )
        
        for i, value in enumerate(series):
            parsed_seq = self._parse_sequence_string(value)
            
            if parsed_seq:
                # Encoder chaque valeur de la séquence
                encoded_seq = [self._encode_value(v, encoder) for v in parsed_seq]
                
                # Appliquer la troncature si nécessaire
                if len(encoded_seq) > self.max_sequence_length:
                    if self.truncation_strategy == 'right':
                        encoded_seq = encoded_seq[:self.max_sequence_length]
                    elif self.truncation_strategy == 'left':
                        encoded_seq = encoded_seq[-self.max_sequence_length:]
                
                # Placer dans le array avec padding
                seq_len = min(len(encoded_seq), self.max_sequence_length)
                encoded_sequences[i, :seq_len] = encoded_seq[:seq_len]
        
        return encoded_sequences
    
    def get_vocab_size(self, column_name: str) -> int:
        """
        Retourne la taille du vocabulaire pour une colonne.
        
        Args:
            column_name: Nom de la colonne
            
        Returns:
            Taille du vocabulaire
        """
        if column_name in self.encoders:
            return self.vocab_size
        return 0
    
    def decode_sequence(self, 
                       encoded_sequence: np.ndarray, 
                       column_name: str) -> List[float]:
        """
        Décode une séquence encodée (pour debugging/inspection).
        
        Args:
            encoded_sequence: Séquence encodée
            column_name: Nom de la colonne originale
            
        Returns:
            Liste des valeurs décodées approximatives
        """
        if column_name not in self.encoders:
            raise ValueError(f"Pas d'encodeur pour la colonne: {column_name}")
        
        encoder = self.encoders[column_name]
        decoded_values = []
        
        for encoded_val in encoded_sequence:
            if encoded_val == self.padding_value:
                continue
                
            # Décodage approximatif (perte d'information)
            if encoder['type'] == 'quantile':
                bins = encoder['bins']
                if encoded_val - 1 < len(bins):
                    decoded_values.append(float(bins[encoded_val - 1]))
                    
            elif encoder['type'] == 'uniform':
                bins = encoder['bins']
                if encoded_val - 1 < len(bins):
                    decoded_values.append(float(bins[encoded_val - 1]))
                    
            elif encoder['type'] == 'minmax':
                min_val = encoder['min_val']
                max_val = encoder['max_val']
                normalized = (encoded_val - 1) / (self.vocab_size - 2)
                decoded_val = min_val + normalized * (max_val - min_val)
                decoded_values.append(float(decoded_val))
                
            elif encoder['type'] == 'constant':
                decoded_values.append(encoder['value'])
        
        return decoded_values