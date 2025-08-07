# === PHASE 1 : EXPLORATION ET VALIDATION - DATAIKU PARTITIONNÉ ===
# Analyse de BASE_SCORE_COMPLETE_prepared pour déploiement T4Rec sur 2024

import dataiku
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import warnings

warnings.filterwarnings("ignore")

print("🔍 PHASE 1 : EXPLORATION BASE_SCORE_COMPLETE_prepared")
print("=" * 70)

# === CONNEXION DATASETS DATAIKU ===
print("\n📊 1. CONNEXION AUX DATASETS...")

# Input : Table partitionnée source
input_dataset = dataiku.Dataset("BASE_SCORE_COMPLETE_prepared")

# Output : Résultats d'analyse (à créer dans Dataiku)
output_dataset = dataiku.Dataset("T4REC_ANALYSIS_PHASE1")

# === ANALYSE PARTITIONS DISPONIBLES ===
print("\n📅 2. ANALYSE DES PARTITIONS...")

try:
    # Récupérer les partitions disponibles
    partitions = input_dataset.list_partitions()
    print(f"✅ Partitions disponibles: {len(partitions)}")

    # Filtrer les partitions 2024
    partitions_2024 = [p for p in partitions if "2024" in p]
    print(f"✅ Partitions 2024: {len(partitions_2024)}")
    print(f"📋 Détail 2024: {partitions_2024}")

except Exception as e:
    print(f"⚠️ Erreur partitions: {e}")
    # Fallback : prendre un échantillon
    partitions_2024 = ["2024-01", "2024-02", "2024-03"]  # À adapter selon votre naming

# === ÉCHANTILLONNAGE INTELLIGENT ===
print("\n🔬 3. ÉCHANTILLONNAGE POUR ANALYSE...")


def sample_partition_data(partition_id, sample_size=10000):
    """Échantillonne une partition de manière optimisée"""
    try:
        # Lire avec limite pour éviter l'overflow mémoire
        df_sample = input_dataset.get_dataframe(
            partition=partition_id, limit=sample_size
        )
        return df_sample, len(df_sample)
    except Exception as e:
        print(f"⚠️ Erreur partition {partition_id}: {e}")
        return None, 0


# Analyser 3 partitions représentatives
sample_partitions = (
    partitions_2024[:3] if len(partitions_2024) >= 3 else partitions_2024
)
analysis_results = {}

total_estimated_size = 0
df_structure_sample = None

for partition in sample_partitions:
    print(f"   📊 Analyse partition: {partition}")
    df_sample, size = sample_partition_data(partition, sample_size=5000)

    if df_sample is not None:
        analysis_results[partition] = {
            "rows_sampled": size,
            "columns_count": len(df_sample.columns),
            "memory_mb": df_sample.memory_usage(deep=True).sum() / 1024 / 1024,
        }

        # Garder un échantillon pour analyse structure
        if df_structure_sample is None:
            df_structure_sample = df_sample.copy()

        # Estimation taille totale (approximation)
        if size > 0:
            # Estimer le ratio échantillon/réel basé sur la taille
            estimated_full_size = size * 20  # Hypothèse : échantillon = 5% du total
            total_estimated_size += estimated_full_size
            analysis_results[partition]["estimated_total_rows"] = estimated_full_size

print(f"✅ Échantillonnage terminé sur {len(analysis_results)} partitions")

# === ANALYSE STRUCTURE DES DONNÉES ===
print("\n🏗️ 4. ANALYSE STRUCTURE DES DONNÉES...")

if df_structure_sample is not None:
    print(f"📊 Dataset échantillon: {df_structure_sample.shape}")

    # === COLONNES DISPONIBLES ===
    all_columns = list(df_structure_sample.columns)
    print(f"📋 Total colonnes: {len(all_columns)}")

    # === COMPARAISON AVEC LE TEST PRÉCÉDENT ===
    # Colonnes utilisées dans votre test pipeline_t4rec_training.py
    expected_sequence_cols = [
        "nbchqemigliss_m12",
        "nb_automobile_12dm",
        "mntecscrdimm",
        "mnt_euro_r_3m",
        "nb_contacts_accueil_service",
    ]
    expected_categorical_cols = [
        "dummy:iac_epa:03",
        "dummy:iac_epa:01",
        "dummy:iac_epa:02",
    ]
    expected_target = "souscription_produit_1m"

    print("\n🔍 5. COMPARAISON AVEC LE TEST PRÉCÉDENT...")

    # Vérifier présence des colonnes
    missing_seq_cols = [col for col in expected_sequence_cols if col not in all_columns]
    missing_cat_cols = [
        col for col in expected_categorical_cols if col not in all_columns
    ]
    target_present = expected_target in all_columns

    # Chercher des colonnes similaires si manquantes
    similar_cols = {}
    for missing_col in missing_seq_cols + missing_cat_cols:
        # Recherche par substring
        similar = [
            col
            for col in all_columns
            if any(part in col.lower() for part in missing_col.lower().split("_")[:2])
        ]
        if similar:
            similar_cols[missing_col] = similar[:3]  # Top 3 similaires

    # === ANALYSE TYPES ET QUALITÉ ===
    print("\n📊 6. ANALYSE TYPES ET QUALITÉ...")

    column_analysis = {}
    for col in all_columns[:50]:  # Analyser les 50 premières colonnes
        try:
            col_info = {
                "dtype": str(df_structure_sample[col].dtype),
                "non_null_count": int(df_structure_sample[col].count()),
                "null_percentage": round(
                    (df_structure_sample[col].isnull().sum() / len(df_structure_sample))
                    * 100,
                    2,
                ),
                "unique_values": int(df_structure_sample[col].nunique())
                if df_structure_sample[col].nunique() < 1000
                else "1000+",
            }

            # Ajouter des stats selon le type
            if pd.api.types.is_numeric_dtype(df_structure_sample[col]):
                col_info["mean"] = (
                    round(float(df_structure_sample[col].mean()), 2)
                    if df_structure_sample[col].mean()
                    == df_structure_sample[col].mean()
                    else "NaN"
                )
                col_info["std"] = (
                    round(float(df_structure_sample[col].std()), 2)
                    if df_structure_sample[col].std() == df_structure_sample[col].std()
                    else "NaN"
                )

            column_analysis[col] = col_info

        except Exception as e:
            column_analysis[col] = {"error": str(e)}

    # === ESTIMATION VOLUMÉTRIE 2024 ===
    print("\n📈 7. ESTIMATION VOLUMÉTRIE 2024...")

    estimated_total_rows_2024 = total_estimated_size
    estimated_total_cols = len(all_columns)
    estimated_memory_gb = (estimated_total_rows_2024 * estimated_total_cols * 8) / (
        1024**3
    )  # 8 bytes par valeur approximatif

    volumetry_estimation = {
        "estimated_total_rows_2024": estimated_total_rows_2024,
        "estimated_total_columns": estimated_total_cols,
        "estimated_memory_gb": round(estimated_memory_gb, 2),
        "estimated_processing_time_hours": round(
            estimated_total_rows_2024 / 50000, 1
        ),  # Approximation basée sur votre test
        "recommended_chunk_size": min(10000, estimated_total_rows_2024 // 10),
    }

else:
    print("❌ Aucun échantillon disponible pour analyse")
    column_analysis = {}
    volumetry_estimation = {}

# === CONSOLIDATION RÉSULTATS ===
print("\n📋 8. CONSOLIDATION DES RÉSULTATS...")

final_analysis = {
    "analysis_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "partitions_analysis": analysis_results,
    "columns_comparison": {
        "total_columns_found": len(all_columns)
        if df_structure_sample is not None
        else 0,
        "expected_sequence_cols": expected_sequence_cols,
        "expected_categorical_cols": expected_categorical_cols,
        "expected_target": expected_target,
        "missing_sequence_cols": missing_seq_cols
        if df_structure_sample is not None
        else [],
        "missing_categorical_cols": missing_cat_cols
        if df_structure_sample is not None
        else [],
        "target_present": target_present if df_structure_sample is not None else False,
        "similar_columns_found": similar_cols
        if df_structure_sample is not None
        else {},
    },
    "data_quality": column_analysis,
    "volumetry_2024": volumetry_estimation,
    "recommendations": [],
}

# === GÉNÉRATION RECOMMANDATIONS ===
print("\n💡 9. GÉNÉRATION RECOMMANDATIONS...")

recommendations = []

# Recommandations basées sur l'analyse
if df_structure_sample is not None:
    if missing_seq_cols:
        recommendations.append(
            f"⚠️ Colonnes séquentielles manquantes: {missing_seq_cols}. Vérifier noms ou utiliser alternatives."
        )

    if missing_cat_cols:
        recommendations.append(
            f"⚠️ Colonnes catégorielles manquantes: {missing_cat_cols}. Adapter le pipeline."
        )

    if not target_present:
        recommendations.append(
            f"❌ Target '{expected_target}' non trouvée. Pipeline non applicable sans adaptation."
        )

    if estimated_memory_gb > 16:
        recommendations.append(
            f"💾 Volume important ({estimated_memory_gb:.1f}GB). Recommandé: processing par chunks."
        )

    if estimated_total_rows_2024 > 100000:
        recommendations.append(
            f"⏱️ Volume important ({estimated_total_rows_2024:,} lignes). Temps d'entraînement estimé: {volumetry_estimation.get('estimated_processing_time_hours', '?')}h."
        )

    if len(all_columns) > 500:
        recommendations.append(
            f"📊 Beaucoup de colonnes ({len(all_columns)}). Recommandé: sélection features pertinentes."
        )

final_analysis["recommendations"] = recommendations

# === CRÉATION DATASET DE SORTIE ===
print("\n💾 10. SAUVEGARDE RÉSULTATS...")

# Convertir en DataFrame pour Dataiku
results_rows = []

# Résumé global
results_rows.append(
    {
        "analysis_type": "SUMMARY",
        "partition": "ALL_2024",
        "metric": "estimated_total_rows",
        "value": estimated_total_rows_2024,
        "details": json.dumps(volumetry_estimation),
    }
)

# Détails par partition
for partition, details in analysis_results.items():
    results_rows.append(
        {
            "analysis_type": "PARTITION",
            "partition": partition,
            "metric": "rows_analyzed",
            "value": details.get("rows_sampled", 0),
            "details": json.dumps(details),
        }
    )

# Colonnes manquantes
for col in missing_seq_cols + missing_cat_cols:
    results_rows.append(
        {
            "analysis_type": "MISSING_COLUMN",
            "partition": "ALL",
            "metric": col,
            "value": 0,
            "details": json.dumps(similar_cols.get(col, [])),
        }
    )

# Recommandations
for i, rec in enumerate(recommendations):
    results_rows.append(
        {
            "analysis_type": "RECOMMENDATION",
            "partition": "ALL",
            "metric": f"recommendation_{i + 1}",
            "value": 1,
            "details": rec,
        }
    )

# DataFrame final
df_results = pd.DataFrame(results_rows)

# Ajouter métadonnées
df_results["analysis_date"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
df_results["compatible_with_test"] = target_present and len(missing_seq_cols) == 0

print(f"✅ Résultats préparés: {len(df_results)} lignes d'analyse")

# === ÉCRITURE OUTPUT DATAIKU ===
try:
    output_dataset.write_with_schema(df_results)
    print("✅ Résultats sauvegardés dans T4REC_ANALYSIS_PHASE1")
except Exception as e:
    print(f"⚠️ Erreur sauvegarde: {e}")
    print("📊 Résultats disponibles en mémoire:")
    print(df_results.head())

# === RÉSUMÉ FINAL ===
print("\n" + "=" * 70)
print("🎯 RÉSUMÉ PHASE 1 - EXPLORATION TERMINÉE")
print("=" * 70)

if df_structure_sample is not None:
    print(
        f"📊 Dataset: {len(all_columns)} colonnes, ~{estimated_total_rows_2024:,} lignes 2024"
    )
    print(
        f"🎯 Compatibilité test: {'✅ OUI' if target_present and len(missing_seq_cols) == 0 else '❌ NON'}"
    )
    print(f"💾 Mémoire estimée: {estimated_memory_gb:.1f}GB")
    print(
        f"⏱️ Temps estimé: {volumetry_estimation.get('estimated_processing_time_hours', '?')}h"
    )

    if recommendations:
        print(f"\n⚠️ {len(recommendations)} recommandations importantes:")
        for rec in recommendations[:3]:
            print(f"   • {rec}")

    print(
        f"\n📋 Prochaine étape: {'Phase 2 - Adaptation pipeline' if target_present else 'Corriger colonnes manquantes'}"
    )

else:
    print("❌ Échec analyse - Vérifier connexion dataset")

print("=" * 70)
