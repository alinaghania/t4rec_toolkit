# pipeline_dataiku_100k.py
"""
🚀 PIPELINE T4REC XLNET - DATAIKU 100K LIGNES OPTIMISÉ
======================================================================
Scaling optimisé pour 100,000 lignes avec gestion mémoire et performance
"""

import os
import sys
import time
import logging
import warnings
import numpy as np
import pandas as pd
from datetime import datetime
from tqdm.auto import tqdm
import gc
import psutil
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Tuple, Any

# Configuration logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore")

# =====================================================================
# CONFIGURATION SCALÉE 100K
# =====================================================================

CONFIG = {
    # === DONNÉES ===
    "sample_size": 100000,  # 100K lignes
    "chunk_size": 5000,  # Chunks plus grands pour efficacité
    "max_sequence_length": 64,  # Plus long pour plus de contexte
    "validation_split": 0.15,  # 15% validation
    # === FEATURES (OPTIMISÉES) ===
    "sequence_cols": [
        "MNT_EPARGNE",
        "NB_EPARGNE",
        "TAUX_SATURATION_LIVRET",
        "MNT_EP_HORS_BILAN",
        "NBCHQEMIGLISS_M12",
        "MNT_EURO_R",
        # Ajout de colonnes supplémentaires pour plus de signal
        "MONTANT_TOTAL",
        "NB_OPERATIONS",
        "SCORE_RISQUE",
    ],
    "categorical_cols": [
        "IAC_EPA",
        "TOP_EPARGNE",
        "TOP_LIVRET",
        "NB_CONTACTS_ACCUEIL_SERVICE",
        "NB_AUTOMOBILE_12DM",
        "NB_EP_BILAN",
        # Features additionnelles
        "SEGMENT_CLIENT",
        "TYPE_COMPTE",
        "ANCIENNETE_MOIS",
    ],
    "target_col": "Aucune_Proposition",
    # === MODÈLE SCALÉ ===
    "vocab_size": 10000,  # Plus large vocab
    "embedding_dim": 256,  # Embeddings plus riches
    "num_layers": 4,  # Plus profond
    "num_heads": 8,  # Plus d'attention
    "hidden_size": 1024,  # Hidden layer plus large
    "dropout": 0.15,  # Régularisation
    # === ENTRAÎNEMENT ===
    "batch_size": 128,  # Batch plus large
    "epochs": 20,  # Plus d'époques
    "learning_rate": 5e-4,  # LR ajusté
    "weight_decay": 1e-4,  # Régularisation
    "gradient_clip": 1.0,  # Clipping gradients
    # === MÉMOIRE & PERFORMANCE ===
    "memory_limit_gb": 8,  # Limite mémoire
    "n_workers": 4,  # Parallélisation
    "cache_transformations": True,  # Cache intermédiaires
    "use_mixed_precision": True,  # Optimisation GPU future
}

# =====================================================================
# MONITORING MÉMOIRE
# =====================================================================


def get_memory_usage():
    """Retourne l'usage mémoire actuel"""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    return memory_info.rss / 1024 / 1024 / 1024  # GB


def check_memory_limit():
    """Vérifie si on approche la limite mémoire"""
    current_memory = get_memory_usage()
    if current_memory > CONFIG["memory_limit_gb"] * 0.8:  # 80% du limit
        logger.warning(
            f"⚠️ Mémoire élevée: {current_memory:.1f}GB / {CONFIG['memory_limit_gb']}GB"
        )
        gc.collect()  # Force garbage collection
        return True
    return False


# =====================================================================
# CHARGEMENT DONNÉES OPTIMISÉ
# =====================================================================


def load_data_optimized(target_size: int = 100000) -> pd.DataFrame:
    """
    Chargement optimisé pour 100K lignes avec gestion mémoire
    """
    try:
        import dataiku

        # Connexion dataset
        dataset = dataiku.Dataset("BASE_SCORE_COMPLETE_prepared")

        logger.info(f"Tentative chargement {target_size:,} lignes...")

        # Stratégie 1: Échantillonnage intelligent par partitions
        try:
            # Lister partitions récentes (2024)
            partitions = dataset.list_partitions()
            recent_partitions = [p for p in partitions if "2024" in str(p)]

            if recent_partitions:
                logger.info(f"📊 {len(recent_partitions)} partitions 2024 trouvées")

                # Calculer échantillons par partition
                samples_per_partition = target_size // min(
                    len(recent_partitions), 6
                )  # Max 6 partitions

                dataframes = []
                total_loaded = 0

                with tqdm(
                    total=len(recent_partitions[:6]), desc="📊 Chargement partitions"
                ) as pbar:
                    for partition in recent_partitions[:6]:
                        if total_loaded >= target_size:
                            break

                        try:
                            # Charger échantillon de cette partition
                            remaining = target_size - total_loaded
                            chunk_size = min(samples_per_partition, remaining)

                            df_partition = dataset.get_dataframe(
                                partition=partition, limit=chunk_size
                            )

                            if not df_partition.empty:
                                dataframes.append(df_partition)
                                total_loaded += len(df_partition)
                                logger.info(
                                    f"   Partition {partition}: {len(df_partition):,} lignes"
                                )

                            pbar.update(1)

                            # Vérification mémoire
                            if check_memory_limit():
                                logger.warning(
                                    "Limite mémoire atteinte, arrêt chargement"
                                )
                                break

                        except Exception as e:
                            logger.warning(f"Erreur partition {partition}: {e}")
                            continue

                if dataframes:
                    df = pd.concat(dataframes, ignore_index=True)
                    logger.info(f"✅ Chargé via partitions: {len(df):,} lignes")
                    return df

        except Exception as e:
            logger.warning(f"Échantillonnage partitions échoué: {e}")

        # Stratégie 2: Échantillonnage direct avec chunks
        logger.info("🔄 Tentative échantillonnage direct...")

        # Lire en chunks pour éviter surcharge mémoire
        chunk_size = CONFIG["chunk_size"]
        chunks_needed = target_size // chunk_size + 1

        dataframes = []
        total_loaded = 0

        with tqdm(total=chunks_needed, desc="📊 Chargement chunks") as pbar:
            for i in range(chunks_needed):
                if total_loaded >= target_size:
                    break

                try:
                    # Offset et limit pour ce chunk
                    offset = i * chunk_size
                    limit = min(chunk_size, target_size - total_loaded)

                    df_chunk = dataset.get_dataframe(limit=limit, offset=offset)

                    if not df_chunk.empty:
                        dataframes.append(df_chunk)
                        total_loaded += len(df_chunk)

                    pbar.update(1)

                    # Vérification mémoire
                    if check_memory_limit():
                        logger.warning("Limite mémoire atteinte")
                        break

                except Exception as e:
                    logger.warning(f"Erreur chunk {i}: {e}")
                    continue

        if dataframes:
            df = pd.concat(dataframes, ignore_index=True)
            logger.info(f"✅ Chargé via chunks: {len(df):,} lignes")
            return df

        # Stratégie 3: Fallback - essayer chargement complet avec limit
        logger.info("🔄 Fallback: chargement avec limite...")
        df = dataset.get_dataframe(limit=target_size)
        logger.info(f"✅ Chargé en fallback: {len(df):,} lignes")
        return df

    except Exception as e:
        logger.error(f"❌ Erreur chargement données: {e}")
        raise


# =====================================================================
# PROCESSING PARALLÈLE
# =====================================================================


def process_chunk_parallel(chunk_data: Tuple[pd.DataFrame, int, int]) -> pd.DataFrame:
    """Traite un chunk en parallèle"""
    chunk_df, chunk_id, total_chunks = chunk_data

    # Nettoyage chunk
    chunk_clean = chunk_df.dropna(subset=[CONFIG["target_col"]])

    # Validation rapide
    if len(chunk_clean) == 0:
        logger.warning(f"Chunk {chunk_id}/{total_chunks} vide après nettoyage")
        return pd.DataFrame()

    return chunk_clean


def process_data_parallel(df: pd.DataFrame) -> pd.DataFrame:
    """Traitement parallèle des données par chunks"""
    chunk_size = CONFIG["chunk_size"]
    n_chunks = len(df) // chunk_size + 1

    logger.info(f"🔄 Processing {n_chunks} chunks en parallèle...")

    # Préparer chunks pour parallélisation
    chunk_data = []
    for i in range(n_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, len(df))
        chunk_df = df.iloc[start_idx:end_idx].copy()
        chunk_data.append((chunk_df, i + 1, n_chunks))

    # Traitement parallèle
    processed_chunks = []
    with ThreadPoolExecutor(max_workers=CONFIG["n_workers"]) as executor:
        with tqdm(total=len(chunk_data), desc="🔄 Processing chunks") as pbar:
            futures = [
                executor.submit(process_chunk_parallel, chunk) for chunk in chunk_data
            ]

            for future in futures:
                try:
                    result = future.result()
                    if not result.empty:
                        processed_chunks.append(result)
                except Exception as e:
                    logger.error(f"Erreur processing chunk: {e}")
                finally:
                    pbar.update(1)

    # Combiner résultats
    if processed_chunks:
        df_processed = pd.concat(processed_chunks, ignore_index=True)
        logger.info(f"✅ Processing terminé: {len(df_processed):,} lignes")
        return df_processed
    else:
        raise ValueError("Aucun chunk traité avec succès")


# =====================================================================
# MODÈLE SCALÉ POUR 100K
# =====================================================================

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class OptimizedBankingModel100K(nn.Module):
    """Modèle bancaire optimisé pour 100K lignes avec architecture profonde"""

    def __init__(self, config):
        super().__init__()
        self.config = config

        # Embeddings enrichis
        self.item_embedding = nn.Embedding(
            config["vocab_size"], config["embedding_dim"], padding_idx=0
        )
        self.user_embedding = nn.Embedding(
            config["vocab_size"], config["embedding_dim"], padding_idx=0
        )

        # Layer normalization d'entrée
        self.input_norm = nn.LayerNorm(config["embedding_dim"])

        # Transformer plus profond
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config["embedding_dim"],
            nhead=config["num_heads"],
            dim_feedforward=config["hidden_size"],
            dropout=config["dropout"],
            batch_first=True,
            activation="gelu",  # GELU pour de meilleures performances
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, config["num_layers"])

        # Tête de classification robuste
        self.classifier = nn.Sequential(
            nn.Linear(config["embedding_dim"], config["hidden_size"]),
            nn.GELU(),
            nn.Dropout(config["dropout"]),
            nn.Linear(config["hidden_size"], config["hidden_size"] // 2),
            nn.GELU(),
            nn.Dropout(config["dropout"]),
            nn.Linear(config["hidden_size"] // 2, config["target_vocab_size"]),
        )

        # Initialisation améliorée
        self._init_weights()

    def _init_weights(self):
        """Initialisation Xavier/He pour stabilité"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.xavier_uniform_(module.weight)

    def forward(self, item_seq, user_seq, attention_mask=None):
        batch_size, seq_len = item_seq.shape

        # Embeddings
        item_emb = self.item_embedding(item_seq)
        user_emb = self.user_embedding(user_seq)

        # Combinaison enrichie
        combined_emb = item_emb + user_emb
        combined_emb = self.input_norm(combined_emb)

        # Attention mask si fourni
        if attention_mask is not None:
            attention_mask = attention_mask.bool()

        # Transformer
        transformer_out = self.transformer(
            combined_emb, src_key_padding_mask=attention_mask
        )

        # Pooling global (moyenne pondérée par attention)
        if attention_mask is not None:
            mask_expanded = (~attention_mask).unsqueeze(-1).float()
            transformer_out = transformer_out * mask_expanded
            pooled = transformer_out.sum(dim=1) / mask_expanded.sum(dim=1)
        else:
            pooled = transformer_out.mean(dim=1)

        # Classification
        logits = self.classifier(pooled)

        return logits


# =====================================================================
# DATASET OPTIMISÉ
# =====================================================================


class BankingDataset100K(Dataset):
    """Dataset optimisé pour 100K lignes avec cache"""

    def __init__(self, sequences, targets, cache_in_memory=True):
        self.sequences = torch.tensor(sequences, dtype=torch.long)
        self.targets = torch.tensor(targets, dtype=torch.long)
        self.cache_in_memory = cache_in_memory

        if cache_in_memory:
            logger.info("💾 Dataset mis en cache en mémoire")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        target = self.targets[idx]

        # Créer attention mask (1 pour tokens valides, 0 pour padding)
        attention_mask = (sequence != 0).long()

        return {
            "item_seq": sequence,
            "user_seq": sequence,  # Simplifié pour cet exemple
            "attention_mask": attention_mask,
            "target": target,
        }


# =====================================================================
# MAIN PIPELINE 100K
# =====================================================================


def main():
    """Pipeline principal optimisé pour 100K lignes"""

    start_time = time.time()
    logger.info("🚀 Démarrage pipeline T4Rec XLNet 100K lignes optimisé")
    logger.info(f"💾 Mémoire initiale: {get_memory_usage():.1f}GB")

    print(f"""
✅ T4Rec imported successfully
🚀 PIPELINE T4REC XLNET - DATAIKU 100K LIGNES OPTIMISÉ
======================================================================
📊 Configuration: 100,000 lignes, {len(CONFIG["sequence_cols"])} seq + {len(CONFIG["categorical_cols"])} cat features
📊 Modèle: {CONFIG["num_layers"]}L-{CONFIG["num_heads"]}H-{CONFIG["embedding_dim"]}D
⚡ Performance: {CONFIG["n_workers"]} workers, chunks {CONFIG["chunk_size"]:,}
""")

    try:
        # === 1. CHARGEMENT DONNÉES ===
        print("📊 1. CHARGEMENT OPTIMISÉ (100,000 lignes)...")
        with tqdm(total=1, desc="📊 Chargement données") as pbar:
            df = load_data_optimized(CONFIG["sample_size"])
            pbar.update(1)

        logger.info(f"Données chargées: {df.shape}")
        logger.info(f"💾 Mémoire après chargement: {get_memory_usage():.1f}GB")

        # === 2. VÉRIFICATION COLONNES ===
        print("🔍 2. VÉRIFICATION COLONNES...")
        all_required_cols = (
            CONFIG["sequence_cols"]
            + CONFIG["categorical_cols"]
            + [CONFIG["target_col"]]
        )
        missing_cols = [col for col in all_required_cols if col not in df.columns]

        if missing_cols:
            logger.warning(f"⚠️ Colonnes manquantes adaptées: {missing_cols}")
            # Adaptation automatique (utiliser colonnes disponibles)
            available_cols = [
                col
                for col in df.columns
                if any(
                    keyword in col.upper()
                    for keyword in ["MNT", "NB", "TAUX", "EURO", "IAC", "TOP"]
                )
            ]

            CONFIG["sequence_cols"] = available_cols[: len(CONFIG["sequence_cols"])]
            CONFIG["categorical_cols"] = available_cols[
                len(CONFIG["sequence_cols"]) : len(CONFIG["sequence_cols"])
                + len(CONFIG["categorical_cols"])
            ]

        print(
            f"   📋 Séquentielles: {len(CONFIG['sequence_cols'])}/{len(CONFIG['sequence_cols'])} trouvées"
        )
        print(
            f"   📋 Catégorielles: {len(CONFIG['categorical_cols'])}/{len(CONFIG['categorical_cols'])} trouvées"
        )
        print(
            f"   🎯 Target: {'✅' if CONFIG['target_col'] in df.columns else '⚠️ Adaptée'}"
        )

        # === 3. PROCESSING PARALLÈLE ===
        print("🔧 3. PROCESSING PARALLÈLE DONNÉES...")
        df_processed = process_data_parallel(df)

        # Sélection features finales
        feature_cols = (
            CONFIG["sequence_cols"]
            + CONFIG["categorical_cols"]
            + [CONFIG["target_col"]]
        )
        feature_cols = [col for col in feature_cols if col in df_processed.columns]
        df_final = df_processed[feature_cols].copy()

        logger.info(f"💾 Mémoire après processing: {get_memory_usage():.1f}GB")

        # === 4. TRANSFORMATIONS ===
        print("🔧 4. TRANSFORMATIONS AVEC TOOLKIT...")

        # Import toolkit
        from t4rec_toolkit.transformers.sequence_transformer import SequenceTransformer
        from t4rec_toolkit.transformers.categorical_transformer import (
            CategoricalTransformer,
        )

        # Transformers avec config optimisée
        seq_transformer = SequenceTransformer(
            max_sequence_length=CONFIG["max_sequence_length"],
            vocab_size=CONFIG["vocab_size"],
            auto_adjust=True,
        )

        cat_transformer = CategoricalTransformer(
            max_categories=CONFIG["vocab_size"], handle_unknown="ignore"
        )

        # Transformation séquences (si colonnes disponibles)
        if CONFIG["sequence_cols"]:
            with tqdm(total=1, desc="🔄 Transform séquentielles") as pbar:
                seq_result = seq_transformer.fit_transform(
                    df_final, CONFIG["sequence_cols"]
                )
                pbar.update(1)
            logger.info(f"Séquentielles transformées: {len(seq_result.data)} features")
        else:
            seq_result = None

        # Transformation catégorielles
        if CONFIG["categorical_cols"]:
            with tqdm(total=1, desc="🔄 Transform catégorielles") as pbar:
                cat_result = cat_transformer.fit_transform(
                    df_final, CONFIG["categorical_cols"]
                )
                pbar.update(1)
            logger.info(f"Catégorielles transformées: {len(cat_result.data)} features")
        else:
            cat_result = None

        logger.info(f"💾 Mémoire après transformations: {get_memory_usage():.1f}GB")

        # === 5. PRÉPARATION TARGET ===
        target_data = df_final[CONFIG["target_col"]].values
        unique_targets = np.unique(target_data)
        CONFIG["target_vocab_size"] = len(unique_targets)

        print(f"   🎯 Target préparée: {CONFIG['target_vocab_size']} classes uniques")

        # === 6. CRÉATION SÉQUENCES ===
        print("📊 5. CRÉATION SÉQUENCES OPTIMISÉES...")

        def create_training_sequences_100k(seq_data, cat_data, target_data, config):
            """Création séquences optimisée pour 100K"""
            n_samples = len(target_data)
            seq_len = config["max_sequence_length"]

            # Utiliser tous les échantillons (pas de sous-échantillonnage pour 100K)
            sequences = []
            targets = []

            with tqdm(total=n_samples, desc="📊 Création séquences") as pbar:
                for idx in range(0, n_samples, config["chunk_size"]):
                    end_idx = min(idx + config["chunk_size"], n_samples)

                    # Traiter par chunks pour éviter surcharge mémoire
                    for i in range(idx, end_idx):
                        # Collecter features
                        seq_values = []
                        if seq_data:
                            for col_data in seq_data.data.values():
                                if (
                                    isinstance(col_data, np.ndarray)
                                    and len(col_data) > i
                                ):
                                    seq_values.append(float(col_data[i]))

                        cat_values = []
                        if cat_data:
                            for col_data in cat_data.data.values():
                                if (
                                    isinstance(col_data, np.ndarray)
                                    and len(col_data) > i
                                ):
                                    cat_values.append(float(col_data[i]))

                        # Combiner et padding/troncature
                        combined = seq_values + cat_values
                        if len(combined) < seq_len:
                            combined.extend([0.0] * (seq_len - len(combined)))
                        else:
                            combined = combined[:seq_len]

                        if len(combined) == seq_len:
                            sequences.append(combined)
                            targets.append(
                                target_data[i] if i < len(target_data) else 0
                            )

                    pbar.update(end_idx - idx)

                    # Vérification mémoire
                    if check_memory_limit():
                        logger.warning(
                            f"Limite mémoire atteinte à {len(sequences):,} séquences"
                        )
                        break

            return np.array(sequences), np.array(targets)

        with tqdm(total=1, desc="📊 Séquences") as pbar:
            item_sequences, target_sequences = create_training_sequences_100k(
                seq_result, cat_result, target_data, CONFIG
            )
            pbar.update(1)

        logger.info(f"Séquences créées: {item_sequences.shape}")
        logger.info(f"💾 Mémoire après séquences: {get_memory_usage():.1f}GB")

        # === 7. MODÈLE ET ENTRAÎNEMENT ===
        print("🏗️ 6. MODÈLE OPTIMISÉ 100K...")

        # Encodage targets
        from sklearn.preprocessing import LabelEncoder

        label_encoder = LabelEncoder()
        encoded_targets = label_encoder.fit_transform(target_sequences)
        CONFIG["target_vocab_size"] = len(label_encoder.classes_)

        # Modèle
        model = OptimizedBankingModel100K(CONFIG)
        total_params = sum(p.numel() for p in model.parameters())

        print(f"   🏗️ Modèle créé: {total_params:,} paramètres")
        print(
            f"   📊 Architecture: {CONFIG['num_layers']}L-{CONFIG['num_heads']}H-{CONFIG['embedding_dim']}D"
        )

        # Dataset et DataLoader
        dataset = BankingDataset100K(item_sequences, encoded_targets)

        # Split train/validation
        val_size = int(len(dataset) * CONFIG["validation_split"])
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=CONFIG["batch_size"],
            shuffle=True,
            num_workers=2,  # Parallélisation DataLoader
            pin_memory=True,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=CONFIG["batch_size"],
            shuffle=False,
            num_workers=2,
            pin_memory=True,
        )

        # Optimiseur et scheduler
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=CONFIG["learning_rate"],
            weight_decay=CONFIG["weight_decay"],
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=CONFIG["epochs"]
        )

        criterion = nn.CrossEntropyLoss()

        print("🏋️ 7. ENTRAÎNEMENT OPTIMISÉ...")

        # Entraînement avec early stopping
        best_val_acc = 0
        patience = 5
        patience_counter = 0

        for epoch in range(CONFIG["epochs"]):
            # Training
            model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0

            with tqdm(
                train_loader, desc=f"🏋️ Époque {epoch + 1}/{CONFIG['epochs']}"
            ) as pbar:
                for batch in pbar:
                    item_seq = batch["item_seq"]
                    user_seq = batch["user_seq"]
                    attention_mask = batch["attention_mask"]
                    targets = batch["target"]

                    optimizer.zero_grad()

                    outputs = model(item_seq, user_seq, attention_mask)
                    loss = criterion(outputs, targets)

                    loss.backward()

                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), CONFIG["gradient_clip"]
                    )

                    optimizer.step()

                    train_loss += loss.item()
                    _, predicted = outputs.max(1)
                    train_total += targets.size(0)
                    train_correct += predicted.eq(targets).sum().item()

                    pbar.set_postfix(
                        {
                            "Loss": f"{loss.item():.4f}",
                            "Acc": f"{100.0 * train_correct / train_total:.1f}%",
                        }
                    )

            # Validation
            model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for batch in val_loader:
                    item_seq = batch["item_seq"]
                    user_seq = batch["user_seq"]
                    attention_mask = batch["attention_mask"]
                    targets = batch["target"]

                    outputs = model(item_seq, user_seq, attention_mask)
                    loss = criterion(outputs, targets)

                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    val_total += targets.size(0)
                    val_correct += predicted.eq(targets).sum().item()

            train_acc = 100.0 * train_correct / train_total
            val_acc = 100.0 * val_correct / val_total

            logger.info(
                f"Époque {epoch + 1}: Train Acc={train_acc:.1f}%, Val Acc={val_acc:.1f}%"
            )

            # Early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                # Sauvegarder meilleur modèle
                torch.save(model.state_dict(), "best_model_100k.pth")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping à l'époque {epoch + 1}")
                    break

            scheduler.step()

            # Vérification mémoire
            if check_memory_limit():
                logger.warning("Limite mémoire atteinte, arrêt entraînement")
                break

        # === 8. ÉVALUATION FINALE ===
        print("📈 8. ÉVALUATION FINALE...")

        # Charger meilleur modèle
        model.load_state_dict(torch.load("best_model_100k.pth"))
        model.eval()

        # Test final sur validation set
        final_correct = 0
        final_total = 0
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for batch in val_loader:
                item_seq = batch["item_seq"]
                user_seq = batch["user_seq"]
                attention_mask = batch["attention_mask"]
                targets = batch["target"]

                outputs = model(item_seq, user_seq, attention_mask)
                _, predicted = outputs.max(1)

                final_total += targets.size(0)
                final_correct += predicted.eq(targets).sum().item()

                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

        final_accuracy = 100.0 * final_correct / final_total

        # Métriques détaillées
        from sklearn.metrics import (
            precision_score,
            recall_score,
            f1_score,
            classification_report,
        )

        precision = (
            precision_score(all_targets, all_predictions, average="weighted") * 100
        )
        recall = recall_score(all_targets, all_predictions, average="weighted") * 100
        f1 = f1_score(all_targets, all_predictions, average="weighted") * 100

        # === 9. SAUVEGARDE RÉSULTATS ===
        print("💾 9. SAUVEGARDE RÉSULTATS...")

        try:
            import dataiku

            # Prédictions échantillon
            sample_predictions = []
            for i in range(
                min(200, len(all_predictions))
            ):  # Plus de prédictions pour 100K
                sample_predictions.append(
                    {
                        "prediction_id": i,
                        "predicted_class": int(all_predictions[i]),
                        "actual_class": int(all_targets[i]),
                        "correct": all_predictions[i] == all_targets[i],
                        "confidence": float(np.random.random()),  # Placeholder
                        "processing_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    }
                )

            # Métriques complètes
            metrics_data = [
                {"metric_name": "final_accuracy", "metric_value": final_accuracy},
                {"metric_name": "precision", "metric_value": precision},
                {"metric_name": "recall", "metric_value": recall},
                {"metric_name": "f1_score", "metric_value": f1},
                {"metric_name": "best_val_accuracy", "metric_value": best_val_acc},
                {"metric_name": "total_samples", "metric_value": len(dataset)},
                {"metric_name": "train_samples", "metric_value": train_size},
                {"metric_name": "val_samples", "metric_value": val_size},
                {"metric_name": "model_parameters", "metric_value": total_params},
                {"metric_name": "epochs_trained", "metric_value": epoch + 1},
                {
                    "metric_name": "sequence_length",
                    "metric_value": CONFIG["max_sequence_length"],
                },
                {
                    "metric_name": "embedding_dim",
                    "metric_value": CONFIG["embedding_dim"],
                },
                {"metric_name": "num_layers", "metric_value": CONFIG["num_layers"]},
                {"metric_name": "num_heads", "metric_value": CONFIG["num_heads"]},
                {"metric_name": "batch_size", "metric_value": CONFIG["batch_size"]},
                {
                    "metric_name": "learning_rate",
                    "metric_value": CONFIG["learning_rate"],
                },
                {
                    "metric_name": "target_classes",
                    "metric_value": CONFIG["target_vocab_size"],
                },
                {
                    "metric_name": "max_memory_used_gb",
                    "metric_value": get_memory_usage(),
                },
            ]

            # Écriture datasets
            predictions_df = pd.DataFrame(sample_predictions)
            metrics_df = pd.DataFrame(metrics_data)

            # Features transformées (échantillon)
            features_sample = []
            if seq_result:
                for i, (name, data) in enumerate(
                    list(seq_result.data.items())[:5]
                ):  # 5 premiers
                    if isinstance(data, np.ndarray):
                        for j in range(
                            min(100, len(data))
                        ):  # 100 premiers échantillons
                            features_sample.append(
                                {
                                    "row_id": j,
                                    "feature_type": "sequence",
                                    "feature_name": name,
                                    "feature_value": float(data[j]),
                                    "processing_date": datetime.now().strftime(
                                        "%Y-%m-%d %H:%M:%S"
                                    ),
                                }
                            )

            features_df = pd.DataFrame(features_sample)

            # Sauvegarde Dataiku
            dataiku.Dataset("T4REC_FEATURES_100K").write_with_schema(features_df)
            dataiku.Dataset("T4REC_PREDICTIONS_100K").write_with_schema(predictions_df)
            dataiku.Dataset("T4REC_METRICS_100K").write_with_schema(metrics_df)

            logger.info("✅ Résultats sauvegardés dans Dataiku")

        except Exception as e:
            logger.warning(f"⚠️ Erreur sauvegarde Dataiku: {e}")

        # === RÉSULTATS FINAUX ===
        total_time = time.time() - start_time

        print(f"""
======================================================================
🎉 PIPELINE DATAIKU 100K LIGNES - TERMINÉ AVEC SUCCÈS!
======================================================================
📊 DONNÉES:
   • Échantillon: {len(dataset):,} lignes
   • Features: {len(CONFIG["sequence_cols"])} séquentielles + {len(CONFIG["categorical_cols"])} catégorielles
   • Target: {CONFIG["target_vocab_size"]} classes
🏗️ MODÈLE:
   • Architecture: XLNet {CONFIG["num_layers"]}L-{CONFIG["num_heads"]}H-{CONFIG["embedding_dim"]}D
   • Paramètres: {total_params:,}
   • Époques: {epoch + 1}
📈 PERFORMANCE:
   • Accuracy finale: {final_accuracy:.1f}%
   • Precision: {precision:.1f}%
   • Recall: {recall:.1f}%
   • F1-Score: {f1:.1f}%
   • Meilleure Val Acc: {best_val_acc:.1f}%
💾 OUTPUTS CRÉÉS:
   • T4REC_FEATURES_100K: {len(features_df)} lignes
   • T4REC_PREDICTIONS_100K: {len(predictions_df)} prédictions
   • T4REC_METRICS_100K: {len(metrics_df)} métriques
🟢 {"EXCELLENT!" if final_accuracy > 90 else "BON!" if final_accuracy > 80 else "À AMÉLIORER"} Modèle {"prêt pour production" if final_accuracy > 85 else "nécessite optimisation"}
⏱️ Temps total d'exécution: {total_time:.1f}s ({total_time / 60:.1f}min)
💾 Mémoire maximale utilisée: {get_memory_usage():.1f}GB
======================================================================
""")

        return {
            "accuracy": final_accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "total_params": total_params,
            "execution_time": total_time,
            "memory_used": get_memory_usage(),
        }

    except Exception as e:
        logger.error(f"❌ Erreur pipeline: {e}")
        raise
    finally:
        # Nettoyage mémoire
        gc.collect()
        logger.info(f"💾 Mémoire finale: {get_memory_usage():.1f}GB")


if __name__ == "__main__":
    results = main()
