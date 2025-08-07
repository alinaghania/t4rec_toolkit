# transformers/sequence_transformer.py
"""
Transformer pour les données séquentielles.

Ce module gère la transformation des colonnes contenant des séquences
au format string vers des arrays numériques encodés, avec gestion
du padding, troncature et vocabulaire.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import pandas as pd
from scipy import stats
import logging
from ..core.base_transformer import BaseTransformer, TransformationResult
from ..core.exceptions import TransformationError


class DataQualityLevel(Enum):
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"


@dataclass
class ColumnAnalysis:
    """Analyse détaillée d'une colonne de données."""

    name: str
    dtype: str
    missing_ratio: float
    unique_ratio: float
    numeric_ratio: float
    quality_level: DataQualityLevel
    recommendations: List[str]
    warnings: List[str]
    stats: Dict[str, float]


class SequenceTransformer(BaseTransformer):
    """
    Transformer intelligent pour les données séquentielles.

    Cette classe analyse les données en profondeur et fournit des
    recommandations adaptées au contexte des données.
    """

    def __init__(
        self,
        max_sequence_length: int = 20,
        vocab_size: int = 10000,
        auto_adjust: bool = True,
        quality_threshold: float = 0.8,
        name: Optional[str] = None,
    ):
        """
        Args:
            max_sequence_length: Longueur maximale des séquences
            vocab_size: Taille du vocabulaire pour l'encodage
            auto_adjust: Ajuster automatiquement les paramètres selon les données
            quality_threshold: Seuil de qualité minimum acceptable
            name: Nom du transformer
        """
        super().__init__(name)
        self.max_sequence_length = max_sequence_length
        self.vocab_size = vocab_size
        self.auto_adjust = auto_adjust
        self.quality_threshold = quality_threshold
        self.logger = logging.getLogger(self.__class__.__name__)
        self.column_analyses = {}

    def _analyze_column(self, series: pd.Series, name: str) -> ColumnAnalysis:
        """Analyse approfondie d'une colonne avec recommandations."""
        # Statistiques de base
        missing_ratio = series.isna().mean()
        unique_ratio = series.nunique() / len(series)

        # Tentative de conversion numérique
        numeric_series = pd.to_numeric(series, errors="coerce")
        numeric_ratio = (~numeric_series.isna()).mean()

        # Statistiques si numérique
        column_stats = {}
        if numeric_ratio > 0:
            valid_values = numeric_series.dropna()
            if len(valid_values) > 0:
                column_stats.update(
                    {
                        "mean": valid_values.mean(),
                        "std": valid_values.std(),
                        "min": valid_values.min(),
                        "max": valid_values.max(),
                        "skew": stats.skew(valid_values)
                        if len(valid_values) > 2
                        else 0,
                    }
                )

        # Évaluation de la qualité
        recommendations = []
        warnings = []

        if missing_ratio > 0.3:
            warnings.append(f"Taux élevé de valeurs manquantes: {missing_ratio:.1%}")
            recommendations.append(
                "Considérer un prétraitement des valeurs manquantes ou "
                "exclure cette colonne si non critique"
            )

        if numeric_ratio < 0.9:
            warnings.append(
                f"Données non-numériques détectées: {1 - numeric_ratio:.1%}"
            )
            recommendations.append(
                "Vérifier le format des données. Exemples non-numériques: "
                f"{series[~numeric_series.notna()].head().tolist()}"
            )

        if unique_ratio < 0.01:
            warnings.append("Très faible variabilité des données")
            recommendations.append(
                "Cette colonne pourrait être mieux traitée comme catégorielle"
            )

        # Recommandations sur la configuration
        if stats and "std" in stats:
            if stats["std"] == 0:
                warnings.append("Colonne constante")
                recommendations.append("Considérer l'exclusion de cette colonne")
            elif stats["std"] < 0.1:
                recommendations.append(
                    "Considérer une normalisation ou standardisation "
                    "due à la faible variance"
                )

        # Déterminer le niveau de qualité
        if numeric_ratio > 0.95 and missing_ratio < 0.1:
            quality = DataQualityLevel.EXCELLENT
        elif numeric_ratio > 0.8 and missing_ratio < 0.2:
            quality = DataQualityLevel.GOOD
        elif numeric_ratio > 0.6 and missing_ratio < 0.3:
            quality = DataQualityLevel.FAIR
        else:
            quality = DataQualityLevel.POOR

        return ColumnAnalysis(
            name=name,
            dtype=str(series.dtype),
            missing_ratio=missing_ratio,
            unique_ratio=unique_ratio,
            numeric_ratio=numeric_ratio,
            quality_level=quality,
            recommendations=recommendations,
            warnings=warnings,
            stats=column_stats,
        )

    def _suggest_parameters(
        self, analyses: Dict[str, ColumnAnalysis]
    ) -> Dict[str, Any]:
        """Suggère des paramètres optimaux basés sur l'analyse des données."""
        suggestions = {}

        # Analyser les longueurs de séquence
        sequence_lengths = [
            len(str(val).split(","))
            for analysis in analyses.values()
            for val in analysis.stats.values()
            if isinstance(val, (list, str))
        ]

        if sequence_lengths:
            p95_length = np.percentile(sequence_lengths, 95)
            suggestions["max_sequence_length"] = min(
                max(int(p95_length * 1.2), 5),  # Au moins 5
                100,  # Maximum raisonnable
            )

        # Analyser la cardinalité des valeurs
        unique_values = sum(
            analysis.stats.get("unique_count", 0) for analysis in analyses.values()
        )

        if unique_values > 0:
            suggestions["vocab_size"] = min(
                max(int(unique_values * 1.5), 100),  # Au moins 100
                50000,  # Maximum raisonnable
            )

        return suggestions

    def fit(
        self, data: pd.DataFrame, feature_columns: Optional[List[str]] = None, **kwargs
    ) -> "SequenceTransformer":
        """
        Ajuste le transformer avec analyse et recommandations.

        Args:
            data: DataFrame d'entraînement
            feature_columns: Colonnes à utiliser
            **kwargs: Paramètres supplémentaires
        """
        self.logger.info("Début de l'analyse des données...")

        # Obtenir les colonnes
        columns = self.get_feature_columns(data, feature_columns)

        # Analyser chaque colonne
        analyses = {}
        for col in columns:
            analysis = self._analyze_column(data[col], col)
            analyses[col] = analysis

            # Logger les résultats
            self.logger.info(f"\nAnalyse de {col}:")
            self.logger.info(f"Qualité: {analysis.quality_level.value}")

            if analysis.warnings:
                for warning in analysis.warnings:
                    self.logger.warning(f"⚠️ {warning}")

            if analysis.recommendations:
                self.logger.info("Recommandations:")
                for i, rec in enumerate(analysis.recommendations, 1):
                    self.logger.info(f"  {i}. {rec}")

        # Suggérer des paramètres optimaux
        if self.auto_adjust:
            suggestions = self._suggest_parameters(analyses)
            if suggestions:
                self.logger.info("\nParamètres suggérés:")
                for param, value in suggestions.items():
                    current = getattr(self, param)
                    if current != value:
                        self.logger.info(
                            f"  {param}: {current} → {value} (modification suggérée)"
                        )
                        if self.auto_adjust:
                            setattr(self, param, value)
                            self.logger.info("  → Paramètre ajusté automatiquement")

        # Stocker les analyses
        self.column_analyses = analyses

        # Vérifier la qualité globale
        poor_quality_cols = [
            col
            for col, analysis in analyses.items()
            if analysis.quality_level in (DataQualityLevel.POOR, DataQualityLevel.FAIR)
        ]

        if poor_quality_cols:
            msg = (
                f"Colonnes de qualité insuffisante détectées: {poor_quality_cols}\n"
                "Considérer un prétraitement ou l'exclusion de ces colonnes."
            )
            if len(poor_quality_cols) / len(columns) > 0.5:
                raise TransformationError(
                    msg, transformer_name=self.name, step="quality_validation"
                )
            else:
                self.logger.warning(msg)

        # Continuer avec le fit standard
        return super().fit(data, feature_columns, **kwargs)

    def transform(self, data: pd.DataFrame) -> TransformationResult:
        """
        Transforme les données avec validation de qualité.

        Args:
            data: DataFrame à transformer
        """
        result = super().transform(data)

        # Ajouter les métriques de qualité
        quality_metrics = {}
        for col in self.feature_columns:
            if col in self.column_analyses:
                analysis = self.column_analyses[col]
                quality_metrics[col] = {
                    "quality_level": analysis.quality_level.value,
                    "numeric_ratio": analysis.numeric_ratio,
                    "missing_ratio": analysis.missing_ratio,
                    "recommendations": analysis.recommendations,
                }

        result.statistics["quality_metrics"] = quality_metrics

        return result

    def get_transformation_summary(self) -> Dict[str, Any]:
        """
        Retourne un résumé détaillé des transformations et analyses.

        Returns:
            Dictionnaire avec le résumé des transformations
        """
        return {
            "parameters": {
                "max_sequence_length": self.max_sequence_length,
                "vocab_size": self.vocab_size,
                "auto_adjust": self.auto_adjust,
            },
            "column_analyses": {
                name: {
                    "quality_level": analysis.quality_level.value,
                    "warnings": analysis.warnings,
                    "recommendations": analysis.recommendations,
                    "stats": analysis.stats,
                }
                for name, analysis in self.column_analyses.items()
            },
        }

