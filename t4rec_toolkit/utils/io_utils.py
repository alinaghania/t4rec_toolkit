"""
Utilitaires pour les opérations d'entrée/sortie.

Ce module fournit les fonctions pour sauvegarder et charger
les modèles, configurations et autres artefacts du toolkit.
"""

import os
import json
import pickle
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
from datetime import datetime

logger = logging.getLogger(__name__)


def ensure_directory(path: Union[str, Path]) -> Path:
    """
    Assure qu'un répertoire existe.

    Args:
        path: Chemin vers le répertoire

    Returns:
        Path object vers le répertoire
    """
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def save_model(
    model: Any,
    save_path: Union[str, Path],
    model_config: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Sauvegarde un modèle T4Rec avec ses métadonnées.

    Args:
        model: Modèle T4Rec à sauvegarder
        save_path: Chemin de sauvegarde
        model_config: Configuration du modèle
        metadata: Métadonnées additionnelles

    Returns:
        Chemin du fichier sauvegardé

    Raises:
        IOError: Si la sauvegarde échoue
    """
    save_path = Path(save_path)
    ensure_directory(save_path.parent)

    try:
        import torch

        # Préparer les métadonnées
        save_metadata = {
            "timestamp": datetime.now().isoformat(),
            "model_type": model.__class__.__name__,
            "toolkit_version": "1.0.0",  # Version du toolkit
        }

        if model_config:
            save_metadata["model_config"] = model_config

        if metadata:
            save_metadata.update(metadata)

        # Sauvegarder le modèle
        if hasattr(model, "state_dict"):
            # Modèle PyTorch
            torch.save(
                {"model_state_dict": model.state_dict(), "metadata": save_metadata},
                save_path,
            )
        else:
            # Autre type de modèle
            with open(save_path, "wb") as f:
                pickle.dump({"model": model, "metadata": save_metadata}, f)

        logger.info(f"Modèle sauvegardé: {save_path}")
        return str(save_path)

    except Exception as e:
        logger.error(f"Erreur lors de la sauvegarde: {e}")
        raise IOError(f"Impossible de sauvegarder le modèle: {e}")


def load_model(load_path: Union[str, Path], model_class: Optional[Any] = None) -> tuple:
    """
    Charge un modèle T4Rec depuis un fichier.

    Args:
        load_path: Chemin vers le modèle sauvegardé
        model_class: Classe du modèle (pour reconstruction)

    Returns:
        Tuple (modèle, métadonnées)

    Raises:
        IOError: Si le chargement échoue
    """
    load_path = Path(load_path)

    if not load_path.exists():
        raise IOError(f"Fichier modèle non trouvé: {load_path}")

    try:
        import torch

        # Tenter de charger comme modèle PyTorch
        try:
            checkpoint = torch.load(load_path, map_location="cpu")

            if "model_state_dict" in checkpoint:
                # Modèle PyTorch avec state_dict
                if model_class is None:
                    logger.warning(
                        "model_class non fourni, retour du state_dict seulement"
                    )
                    return checkpoint["model_state_dict"], checkpoint.get(
                        "metadata", {}
                    )

                model = model_class()
                model.load_state_dict(checkpoint["model_state_dict"])
                return model, checkpoint.get("metadata", {})
            else:
                # Format pickle dans un fichier torch
                return checkpoint["model"], checkpoint.get("metadata", {})

        except:
            # Tenter de charger comme pickle
            with open(load_path, "rb") as f:
                checkpoint = pickle.load(f)
                return checkpoint["model"], checkpoint.get("metadata", {})

    except Exception as e:
        logger.error(f"Erreur lors du chargement: {e}")
        raise IOError(f"Impossible de charger le modèle: {e}")


def save_config(
    config: Dict[str, Any], save_path: Union[str, Path], pretty_print: bool = True
) -> str:
    """
    Sauvegarde une configuration en JSON.

    Args:
        config: Configuration à sauvegarder
        save_path: Chemin de sauvegarde
        pretty_print: Formater le JSON

    Returns:
        Chemin du fichier sauvegardé
    """
    save_path = Path(save_path)
    ensure_directory(save_path.parent)

    # Ajouter métadonnées
    config_with_meta = {
        "config": config,
        "timestamp": datetime.now().isoformat(),
        "toolkit_version": "1.0.0",
    }

    try:
        with open(save_path, "w", encoding="utf-8") as f:
            if pretty_print:
                json.dump(config_with_meta, f, indent=2, ensure_ascii=False)
            else:
                json.dump(config_with_meta, f, ensure_ascii=False)

        logger.info(f"Configuration sauvegardée: {save_path}")
        return str(save_path)

    except Exception as e:
        logger.error(f"Erreur lors de la sauvegarde de config: {e}")
        raise IOError(f"Impossible de sauvegarder la configuration: {e}")


def load_config(load_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Charge une configuration depuis un fichier JSON.

    Args:
        load_path: Chemin vers le fichier de configuration

    Returns:
        Configuration chargée

    Raises:
        IOError: Si le chargement échoue
    """
    load_path = Path(load_path)

    if not load_path.exists():
        raise IOError(f"Fichier configuration non trouvé: {load_path}")

    try:
        with open(load_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Extraire la configuration si elle est encapsulée
        if "config" in data:
            config = data["config"]
            logger.info(
                f"Configuration chargée: {load_path} (timestamp: {data.get('timestamp', 'unknown')})"
            )
        else:
            config = data
            logger.info(f"Configuration chargée: {load_path}")

        return config

    except Exception as e:
        logger.error(f"Erreur lors du chargement de config: {e}")
        raise IOError(f"Impossible de charger la configuration: {e}")


def get_model_info(model_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Extrait les informations d'un modèle sauvegardé sans le charger.

    Args:
        model_path: Chemin vers le modèle

    Returns:
        Dictionnaire des informations du modèle
    """
    model_path = Path(model_path)

    if not model_path.exists():
        return {"error": "Fichier non trouvé"}

    try:
        import torch

        # Charger seulement les métadonnées
        checkpoint = torch.load(model_path, map_location="cpu")

        info = {
            "file_size_mb": model_path.stat().st_size / (1024 * 1024),
            "modified_time": datetime.fromtimestamp(
                model_path.stat().st_mtime
            ).isoformat(),
        }

        if "metadata" in checkpoint:
            info.update(checkpoint["metadata"])

        return info

    except Exception as e:
        return {"error": str(e)}


def create_experiment_directory(
    base_path: Union[str, Path], experiment_name: Optional[str] = None
) -> Path:
    """
    Crée un répertoire d'expérience avec timestamp.

    Args:
        base_path: Répertoire de base
        experiment_name: Nom de l'expérience (optionnel)

    Returns:
        Path vers le répertoire créé
    """
    base_path = Path(base_path)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if experiment_name:
        exp_dir = base_path / f"{experiment_name}_{timestamp}"
    else:
        exp_dir = base_path / f"experiment_{timestamp}"

    ensure_directory(exp_dir)

    # Créer sous-répertoires standard
    (exp_dir / "models").mkdir(exist_ok=True)
    (exp_dir / "configs").mkdir(exist_ok=True)
    (exp_dir / "logs").mkdir(exist_ok=True)
    (exp_dir / "results").mkdir(exist_ok=True)

    logger.info(f"Répertoire d'expérience créé: {exp_dir}")
    return exp_dir


def backup_file(
    file_path: Union[str, Path], backup_suffix: str = "_backup"
) -> Optional[Path]:
    """
    Crée une sauvegarde d'un fichier existant.

    Args:
        file_path: Chemin vers le fichier à sauvegarder
        backup_suffix: Suffixe pour la sauvegarde

    Returns:
        Path vers la sauvegarde ou None si pas de fichier original
    """
    file_path = Path(file_path)

    if not file_path.exists():
        return None

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = (
        file_path.parent
        / f"{file_path.stem}{backup_suffix}_{timestamp}{file_path.suffix}"
    )

    try:
        import shutil

        shutil.copy2(file_path, backup_path)
        logger.info(f"Sauvegarde créée: {backup_path}")
        return backup_path
    except Exception as e:
        logger.error(f"Erreur lors de la sauvegarde: {e}")
        return None


def cleanup_old_files(
    directory: Union[str, Path],
    pattern: str = "*",
    max_age_days: int = 30,
    dry_run: bool = True,
) -> List[Path]:
    """
    Nettoie les anciens fichiers dans un répertoire.

    Args:
        directory: Répertoire à nettoyer
        pattern: Pattern des fichiers à considérer
        max_age_days: Age maximum en jours
        dry_run: Mode simulation (ne supprime pas vraiment)

    Returns:
        Liste des fichiers qui seraient/ont été supprimés
    """
    directory = Path(directory)

    if not directory.exists():
        return []

    from datetime import timedelta

    cutoff_time = datetime.now() - timedelta(days=max_age_days)
    old_files = []

    for file_path in directory.glob(pattern):
        if file_path.is_file():
            file_time = datetime.fromtimestamp(file_path.stat().st_mtime)
            if file_time < cutoff_time:
                old_files.append(file_path)

                if not dry_run:
                    try:
                        file_path.unlink()
                        logger.info(f"Fichier supprimé: {file_path}")
                    except Exception as e:
                        logger.error(f"Erreur suppression {file_path}: {e}")

    if dry_run and old_files:
        logger.info(f"Mode simulation: {len(old_files)} fichiers seraient supprimés")

    return old_files
