# core/validator.py
"""
Validateur de données pour le preprocessing.

Ce module fournit les fonctionnalités de validation des données
d'entrée et de sortie pour s'assurer de la qualité et de la
conformité des transformations.
"""

from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass
import pandas as pd
import numpy as np
import re

from .exceptions import DataValidationError


@dataclass
class ValidationResult:
    """
    Résultat d'une validation de données.

    Cette classe encapsule les résultats d'une validation,
    incluant les erreurs, warnings et statistiques.
    """

    is_valid: bool
    """Indique si les données sont valides"""

    errors: List[Dict[str, Any]]
    """Liste des erreurs de validation"""

    warnings: List[Dict[str, Any]]
    """Liste des warnings de validation"""

    statistics: Dict[str, Any]
    """Statistiques sur les données validées"""

    def add_error(
        self,
        error_type: str,
        message: str,
        column: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        """
        Ajoute une erreur au résultat.

        Args:
            error_type: Type d'erreur
            message: Message d'erreur
            column: Colonne concernée (optionnel)
            details: Détails supplémentaires (optionnel)
        """
        error = {
            "type": error_type,
            "message": message,
            "column": column,
            "details": details or {},
        }
        self.errors.append(error)
        self.is_valid = False

    def add_warning(
        self,
        warning_type: str,
        message: str,
        column: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        """
        Ajoute un warning au résultat.

        Args:
            warning_type: Type de warning
            message: Message du warning
            column: Colonne concernée (optionnel)
            details: Détails supplémentaires (optionnel)
        """
        warning = {
            "type": warning_type,
            "message": message,
            "column": column,
            "details": details or {},
        }
        self.warnings.append(warning)

    def get_summary(self) -> str:
        """
        Retourne un résumé de la validation.

        Returns:
            String résumant les résultats
        """
        summary = f"Validation: {'PASSED' if self.is_valid else 'FAILED'}\n"
        summary += f"Erreurs: {len(self.errors)}, Warnings: {len(self.warnings)}"
        return summary


class DataValidator:
    """
    Validateur de données pour les transformations.

    Cette classe fournit les méthodes de validation pour
    s'assurer que les données respectent les contraintes
    attendues avant et après transformation.
    """

    def __init__(self):
        """Initialise le validateur."""
        self.sequence_patterns = {
            "list_format": r"^\[.*\]$",
            "numeric_sequence": r"^\[[\d\.\,\s\-\+e]+\]$",
        }

    def validate_input_data(
        self,
        data: pd.DataFrame,
        required_columns: Optional[List[str]] = None,
        target_column: Optional[str] = None,
    ) -> ValidationResult:
        """
        Valide les données d'entrée.

        Args:
            data: DataFrame à valider
            required_columns: Colonnes requises (optionnel)
            target_column: Colonne cible (optionnel)

        Returns:
            Résultat de la validation
        """
        result = ValidationResult(is_valid=True, errors=[], warnings=[], statistics={})

        # Validation de base
        self._validate_basic_structure(data, result)

        # Validation des colonnes requises
        if required_columns:
            self._validate_required_columns(data, required_columns, result)

        # Validation de la colonne cible
        if target_column:
            self._validate_target_column(data, target_column, result)

        # Statistiques générales
        result.statistics.update(
            {
                "n_rows": len(data),
                "n_columns": len(data.columns),
                "memory_usage_mb": data.memory_usage(deep=True).sum() / 1024**2,
                "missing_values_total": data.isnull().sum().sum(),
            }
        )

        return result

    def validate_sequence_data(
        self,
        data: pd.DataFrame,
        sequence_columns: List[str],
        expected_length: Optional[int] = None,
    ) -> ValidationResult:
        """
        Valide spécifiquement les données de séquences.

        Args:
            data: DataFrame contenant les séquences
            sequence_columns: Colonnes contenant les séquences
            expected_length: Longueur attendue des séquences (optionnel)

        Returns:
            Résultat de la validation des séquences
        """
        result = ValidationResult(is_valid=True, errors=[], warnings=[], statistics={})

        sequence_stats = {}

        for col in sequence_columns:
            if col not in data.columns:
                result.add_error(
                    "missing_column",
                    f"Colonne de séquence manquante: {col}",
                    column=col,
                )
                continue

            col_stats = self._validate_sequence_column(
                data[col], col, result, expected_length
            )
            sequence_stats[col] = col_stats

        result.statistics["sequence_stats"] = sequence_stats
        return result

    def validate_categorical_data(
        self,
        data: pd.DataFrame,
        categorical_columns: List[str],
        max_categories: Optional[int] = None,
    ) -> ValidationResult:
        """
        Valide les données catégorielles.

        Args:
            data: DataFrame contenant les features catégorielles
            categorical_columns: Colonnes catégorielles
            max_categories: Nombre maximum de catégories autorisé (optionnel)

        Returns:
            Résultat de la validation des features catégorielles
        """
        result = ValidationResult(is_valid=True, errors=[], warnings=[], statistics={})

        categorical_stats = {}

        for col in categorical_columns:
            if col not in data.columns:
                result.add_error(
                    "missing_column",
                    f"Colonne catégorielle manquante: {col}",
                    column=col,
                )
                continue

            col_stats = self._validate_categorical_column(
                data[col], col, result, max_categories
            )
            categorical_stats[col] = col_stats

        result.statistics["categorical_stats"] = categorical_stats
        return result

    def _validate_basic_structure(self, data: pd.DataFrame, result: ValidationResult):
        """Valide la structure de base du DataFrame."""
        if data.empty:
            result.add_error("empty_dataframe", "DataFrame vide")
            return

        if len(data.columns) == 0:
            result.add_error("no_columns", "Aucune colonne dans le DataFrame")
            return

        # Vérifier les colonnes dupliquées
        duplicate_cols = data.columns[data.columns.duplicated()].tolist()
        if duplicate_cols:
            result.add_error(
                "duplicate_columns", f"Colonnes dupliquées: {duplicate_cols}"
            )

    def _validate_required_columns(
        self, data: pd.DataFrame, required_columns: List[str], result: ValidationResult
    ):
        """Valide la présence des colonnes requises."""
        missing_cols = set(required_columns) - set(data.columns)
        if missing_cols:
            result.add_error(
                "missing_required_columns",
                f"Colonnes requises manquantes: {list(missing_cols)}",
            )

    def _validate_target_column(
        self, data: pd.DataFrame, target_column: str, result: ValidationResult
    ):
        """Valide la colonne cible."""
        if target_column not in data.columns:
            result.add_error(
                "missing_target",
                f"Colonne cible manquante: {target_column}",
                column=target_column,
            )
            return

        target_data = data[target_column]

        # Vérifier les valeurs manquantes
        missing_pct = target_data.isnull().mean() * 100
        if missing_pct > 50:
            result.add_error(
                "high_missing_target",
                f"Trop de valeurs manquantes dans la cible: {missing_pct:.1f}%",
                column=target_column,
            )
        elif missing_pct > 10:
            result.add_warning(
                "some_missing_target",
                f"Valeurs manquantes dans la cible: {missing_pct:.1f}%",
                column=target_column,
            )

        # Vérifier la variabilité
        n_unique = target_data.nunique()
        if n_unique == 1:
            result.add_error(
                "constant_target",
                "La colonne cible n'a qu'une seule valeur unique",
                column=target_column,
            )
        elif n_unique == len(target_data):
            result.add_warning(
                "unique_target_values",
                "Chaque observation a une valeur cible unique",
                column=target_column,
            )

    def _validate_sequence_column(
        self,
        series: pd.Series,
        col_name: str,
        result: ValidationResult,
        expected_length: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Valide une colonne de séquences."""
        stats = {
            "valid_sequences": 0,
            "invalid_sequences": 0,
            "lengths": [],
            "parse_errors": 0,
        }

        for idx, value in series.items():
            if pd.isna(value):
                continue

            # Tenter de parser la séquence
            try:
                parsed_seq = self._parse_sequence_string(str(value))
                if parsed_seq is not None:
                    stats["valid_sequences"] += 1
                    stats["lengths"].append(len(parsed_seq))
                else:
                    stats["invalid_sequences"] += 1
                    if stats["invalid_sequences"] <= 5:  # Limiter les exemples
                        result.add_warning(
                            "invalid_sequence_format",
                            f"Format de séquence invalide à l'index {idx}: {value[:50]}...",
                            column=col_name,
                        )
            except Exception:
                stats["parse_errors"] += 1

        # Analyser les longueurs si on a des séquences valides
        if stats["lengths"]:
            lengths_array = np.array(stats["lengths"])
            stats.update(
                {
                    "mean_length": float(np.mean(lengths_array)),
                    "std_length": float(np.std(lengths_array)),
                    "min_length": int(np.min(lengths_array)),
                    "max_length": int(np.max(lengths_array)),
                }
            )

            # Vérifier la longueur attendue
            if expected_length and stats["mean_length"] != expected_length:
                result.add_warning(
                    "unexpected_sequence_length",
                    f"Longueur moyenne {stats['mean_length']:.1f} != longueur attendue {expected_length}",
                    column=col_name,
                )

        return stats

    def _validate_categorical_column(
        self,
        series: pd.Series,
        col_name: str,
        result: ValidationResult,
        max_categories: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Valide une colonne catégorielle."""
        n_unique = series.nunique()
        n_total = len(series)
        missing_pct = series.isnull().mean() * 100

        stats = {
            "n_categories": n_unique,
            "n_total": n_total,
            "missing_percentage": missing_pct,
            "cardinality_ratio": n_unique / n_total if n_total > 0 else 0,
        }

        # Vérifier le nombre de catégories
        if max_categories and n_unique > max_categories:
            result.add_warning(
                "high_cardinality",
                f"Trop de catégories ({n_unique}) > limite ({max_categories})",
                column=col_name,
            )

        # Vérifier la cardinalité
        if stats["cardinality_ratio"] > 0.95:
            result.add_warning(
                "very_high_cardinality",
                f"Cardinalité très élevée: {stats['cardinality_ratio']:.2f}",
                column=col_name,
            )

        return stats

    def _parse_sequence_string(self, seq_str: str) -> Optional[List[float]]:
        """
        Parse une chaîne de séquence en liste de nombres.

        Args:
            seq_str: Chaîne à parser

        Returns:
            Liste de nombres ou None si le parsing échoue
        """
        try:
            # Nettoyer la chaîne
            seq_str = seq_str.strip()

            # Vérifier le format de base
            if not re.match(self.sequence_patterns["list_format"], seq_str):
                return None

            # Extraire le contenu entre crochets
            content = seq_str[1:-1].strip()
            if not content:
                return []

            # Séparer par les virgules
            values = []
            for item in content.split(","):
                item = item.strip()
                if item:
                    values.append(float(item))

            return values

        except (ValueError, AttributeError):
            return None

    def get_data_profile(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Génère un profil complet des données.

        Args:
            data: DataFrame à profiler

        Returns:
            Dictionnaire avec le profil des données
        """
        profile = {
            "shape": data.shape,
            "dtypes": data.dtypes.value_counts().to_dict(),
            "missing_values": data.isnull().sum().to_dict(),
            "memory_usage_mb": data.memory_usage(deep=True).sum() / 1024**2,
        }

        # Statistiques numériques
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            profile["numeric_stats"] = data[numeric_cols].describe().to_dict()

        # Détection automatique des types de colonnes
        profile["detected_types"] = self._detect_column_types(data)

        return profile

    def _detect_column_types(self, data: pd.DataFrame) -> Dict[str, List[str]]:
        """
        Détecte automatiquement les types de colonnes.

        Args:
            data: DataFrame à analyser

        Returns:
            Dictionnaire classifiant les colonnes par type
        """
        types = {
            "sequence": [],
            "categorical": [],
            "numerical": [],
            "datetime": [],
            "text": [],
            "other": [],
        }

        for col in data.columns:
            col_type = self._classify_column(data[col])
            types[col_type].append(col)

        return types

    def _classify_column(self, series: pd.Series) -> str:
        """
        Classifie le type d'une colonne basé sur son contenu.

        Args:
            series: Série à classifier

        Returns:
            Type de colonne détecté
        """
        # Échantillon pour l'analyse (max 1000 valeurs non-nulles)
        sample = series.dropna().head(1000)
        if len(sample) == 0:
            return "other"

        # Vérifier si c'est une séquence
        if self._is_sequence_column(sample):
            return "sequence"

        # Vérifier si c'est numérique
        if pd.api.types.is_numeric_dtype(series):
            return "numerical"

        # Vérifier si c'est datetime
        if pd.api.types.is_datetime64_any_dtype(series):
            return "datetime"

        # Vérifier si c'est catégoriel
        if self._is_categorical_column(sample):
            return "categorical"

        # Vérifier si c'est du texte
        if self._is_text_column(sample):
            return "text"

        return "other"

    def _is_sequence_column(self, sample: pd.Series) -> bool:
        """Détermine si une colonne contient des séquences."""
        # Vérifier un échantillon de valeurs
        test_values = sample.head(10).astype(str)
        sequence_count = 0

        for value in test_values:
            if re.match(self.sequence_patterns["list_format"], value.strip()):
                sequence_count += 1

        # Si plus de 50% des valeurs ressemblent à des séquences
        return sequence_count / len(test_values) > 0.5

    def _is_categorical_column(self, sample: pd.Series) -> bool:
        """Détermine si une colonne est catégorielle."""
        n_unique = sample.nunique()
        n_total = len(sample)

        # Critères pour être catégoriel:
        # - Peu de valeurs uniques par rapport au total
        # - Ou présence de patterns typiques (dummy variables, etc.)
        cardinality_ratio = n_unique / n_total if n_total > 0 else 0

        # Vérifier les patterns de noms de colonnes
        col_name = sample.name if hasattr(sample, "name") else ""
        is_dummy = str(col_name).startswith("dummy") or "_" in str(col_name)

        return cardinality_ratio < 0.1 or (cardinality_ratio < 0.5 and is_dummy)

    def _is_text_column(self, sample: pd.Series) -> bool:
        """Détermine si une colonne contient du texte."""
        # Vérifier la longueur moyenne des chaînes
        str_sample = sample.astype(str)
        avg_length = str_sample.str.len().mean()

        # Si la longueur moyenne > 20 caractères, probablement du texte
        return avg_length > 20


class SequenceValidator:
    """
    Validateur spécialisé pour les données de séquences.

    Cette classe fournit des validations avancées spécifiquement
    conçues pour les données séquentielles utilisées dans les
    modèles de recommandation.
    """

    def __init__(self, expected_length: Optional[int] = None):
        """
        Initialise le validateur de séquences.

        Args:
            expected_length: Longueur attendue des séquences
        """
        self.expected_length = expected_length

    def validate_sequence_consistency(
        self, sequences: Dict[str, np.ndarray]
    ) -> ValidationResult:
        """
        Valide la cohérence entre plusieurs séquences.

        Args:
            sequences: Dictionnaire de séquences transformées

        Returns:
            Résultat de la validation de cohérence
        """
        result = ValidationResult(is_valid=True, errors=[], warnings=[], statistics={})

        if not sequences:
            result.add_error("no_sequences", "Aucune séquence fournie")
            return result

        # Vérifier que toutes les séquences ont la même longueur de batch
        batch_sizes = {name: seq.shape[0] for name, seq in sequences.items()}
        unique_batch_sizes = set(batch_sizes.values())

        if len(unique_batch_sizes) > 1:
            result.add_error(
                "inconsistent_batch_sizes",
                f"Tailles de batch incohérentes: {batch_sizes}",
            )

        # Vérifier les longueurs de séquences
        sequence_lengths = {}
        for name, seq in sequences.items():
            if seq.ndim >= 2:
                sequence_lengths[name] = seq.shape[1]

        unique_seq_lengths = set(sequence_lengths.values())
        if len(unique_seq_lengths) > 1:
            result.add_warning(
                "different_sequence_lengths",
                f"Longueurs de séquences différentes: {sequence_lengths}",
            )

        # Statistiques
        result.statistics.update(
            {
                "n_sequences": len(sequences),
                "batch_sizes": batch_sizes,
                "sequence_lengths": sequence_lengths,
                "total_elements": sum(seq.size for seq in sequences.values()),
            }
        )

        return result

    def validate_sequence_values(
        self,
        sequence: np.ndarray,
        sequence_name: str,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
    ) -> ValidationResult:
        """
        Valide les valeurs dans une séquence.

        Args:
            sequence: Array de la séquence
            sequence_name: Nom de la séquence
            min_value: Valeur minimum autorisée (optionnel)
            max_value: Valeur maximum autorisée (optionnel)

        Returns:
            Résultat de la validation des valeurs
        """
        result = ValidationResult(is_valid=True, errors=[], warnings=[], statistics={})

        # Vérifications de base
        if sequence.size == 0:
            result.add_error("empty_sequence", f"Séquence vide: {sequence_name}")
            return result

        # Vérifier les valeurs infinies ou NaN
        n_inf = np.isinf(sequence).sum()
        n_nan = np.isnan(sequence).sum()

        if n_inf > 0:
            result.add_error(
                "infinite_values",
                f"Valeurs infinies détectées: {n_inf}",
                column=sequence_name,
            )

        if n_nan > 0:
            result.add_warning(
                "nan_values", f"Valeurs NaN détectées: {n_nan}", column=sequence_name
            )

        # Vérifier les plages de valeurs
        if min_value is not None:
            n_below_min = (sequence < min_value).sum()
            if n_below_min > 0:
                result.add_error(
                    "values_below_minimum",
                    f"Valeurs < {min_value}: {n_below_min}",
                    column=sequence_name,
                )

        if max_value is not None:
            n_above_max = (sequence > max_value).sum()
            if n_above_max > 0:
                result.add_error(
                    "values_above_maximum",
                    f"Valeurs > {max_value}: {n_above_max}",
                    column=sequence_name,
                )

        # Statistiques
        valid_values = sequence[~(np.isnan(sequence) | np.isinf(sequence))]
        if len(valid_values) > 0:
            result.statistics.update(
                {
                    "mean": float(np.mean(valid_values)),
                    "std": float(np.std(valid_values)),
                    "min": float(np.min(valid_values)),
                    "max": float(np.max(valid_values)),
                    "n_unique": len(np.unique(valid_values)),
                    "n_valid": len(valid_values),
                    "n_total": sequence.size,
                }
            )

        return result


class SchemaValidator:
    """
    Validateur pour les schémas T4Rec.

    Cette classe valide que les schémas générés sont conformes
    aux attentes de T4Rec et contiennent toutes les informations
    nécessaires.
    """

    def __init__(self):
        """Initialise le validateur de schémas."""
        self.required_schema_fields = ["feature_specs"]
        self.required_spec_fields = ["name", "dtype"]

    def validate_schema(self, schema: Dict[str, Any]) -> ValidationResult:
        """
        Valide un schéma T4Rec.

        Args:
            schema: Schéma à valider

        Returns:
            Résultat de la validation du schéma
        """
        result = ValidationResult(is_valid=True, errors=[], warnings=[], statistics={})

        # Vérifier les champs requis au niveau du schéma
        for field in self.required_schema_fields:
            if field not in schema:
                result.add_error(
                    "missing_schema_field",
                    f"Champ requis manquant dans le schéma: {field}",
                )

        # Valider les spécifications de features
        if "feature_specs" in schema:
            self._validate_feature_specs(schema["feature_specs"], result)

        # Statistiques du schéma
        if "feature_specs" in schema:
            specs = schema["feature_specs"]
            result.statistics.update(
                {
                    "n_features": len(specs),
                    "dtypes": self._count_dtypes(specs),
                    "sequence_features": sum(
                        1 for spec in specs if spec.get("is_sequence", False)
                    ),
                    "categorical_features": sum(
                        1
                        for spec in specs
                        if not spec.get("is_sequence", False)
                        and spec.get("dtype") == "int32"
                    ),
                    "numerical_features": sum(
                        1 for spec in specs if spec.get("dtype") == "float32"
                    ),
                }
            )

        return result

    def _validate_feature_specs(
        self, feature_specs: List[Dict[str, Any]], result: ValidationResult
    ):
        """Valide les spécifications des features."""
        if not isinstance(feature_specs, list):
            result.add_error(
                "invalid_feature_specs_type", "feature_specs doit être une liste"
            )
            return

        if len(feature_specs) == 0:
            result.add_error(
                "empty_feature_specs", "Aucune spécification de feature fournie"
            )
            return

        feature_names = []

        for i, spec in enumerate(feature_specs):
            if not isinstance(spec, dict):
                result.add_error(
                    "invalid_spec_type", f"Spécification {i} doit être un dictionnaire"
                )
                continue

            # Vérifier les champs requis
            for field in self.required_spec_fields:
                if field not in spec:
                    result.add_error(
                        "missing_spec_field",
                        f"Champ requis manquant dans la spec {i}: {field}",
                    )

            # Vérifier le nom unique
            if "name" in spec:
                name = spec["name"]
                if name in feature_names:
                    result.add_error(
                        "duplicate_feature_name", f"Nom de feature dupliqué: {name}"
                    )
                feature_names.append(name)

            # Valider le dtype
            if "dtype" in spec:
                self._validate_dtype(spec["dtype"], f"spec_{i}", result)

            # Valider vocab_size pour les features catégorielles
            if spec.get("dtype") == "int32" and "vocab_size" not in spec:
                result.add_warning(
                    "missing_vocab_size",
                    f"vocab_size manquant pour la feature catégorielle: {spec.get('name', f'spec_{i}')}",
                )

    def _validate_dtype(self, dtype: str, spec_name: str, result: ValidationResult):
        """Valide un type de données."""
        valid_dtypes = ["int32", "int64", "float32", "float64", "string"]

        if dtype not in valid_dtypes:
            result.add_error(
                "invalid_dtype",
                f"dtype invalide dans {spec_name}: {dtype}. Valides: {valid_dtypes}",
            )

    def _count_dtypes(self, feature_specs: List[Dict[str, Any]]) -> Dict[str, int]:
        """Compte les types de données dans les spécifications."""
        dtype_counts = {}
        for spec in feature_specs:
            dtype = spec.get("dtype", "unknown")
            dtype_counts[dtype] = dtype_counts.get(dtype, 0) + 1
        return dtype_counts
