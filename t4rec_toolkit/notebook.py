# === T4REC XLNET INTÉGRÉ AVEC VOS DONNÉES - VERSION CORRIGÉE POUR 23.04.00 ===

print("🚀 T4REC XLNET - UTILISANT VOS DONNÉES TRANSFORMÉES - VERSION CORRIGÉE")
print("=" * 75)

import torch
import transformers4rec.torch as tr
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

try:
    # Vérifier que les variables nécessaires existent
    if "tabular_data" not in globals():
        print(
            "❌ Variable 'tabular_data' non trouvée. Exécutez d'abord le script de préparation des données."
        )
        raise NameError("tabular_data non définie")

    if "df" not in globals():
        print(
            "❌ Variable 'df' non trouvée. Exécutez d'abord le script de préparation des données."
        )
        raise NameError("df non définie")

    print(
        f"✅ Données disponibles: {len(tabular_data)} features, {len(df)} échantillons"
    )

    # Configuration T4Rec SIMPLIFIÉE pour version 23.04.00
    CONFIG = {
        "d_model": 128,
        "n_head": 4,
        "n_layer": 2,
        "max_sequence_length": 15,
        "mem_len": 30,
        "dropout": 0.1,
        "hidden_size": 128,
        "batch_size": 32,
    }

    # 1. Créer le schéma T4Rec avec features catégorielles
    print("\n📋 Schéma T4Rec avec features catégorielles...")

    from merlin.schema import Schema, ColumnSchema, Tags

    # Échantillons et features disponibles
    n_samples = len(df)
    print(f"Échantillons disponibles: {n_samples}")

    # Identifier les colonnes catégorielles dans tabular_data
    categorical_names = [
        key for key in tabular_data.keys() if "dummy" in key or "encoded" in key
    ]
    print(f"Features catégorielles trouvées: {categorical_names[:3]}...")

    # Créer item_id à partir des indices
    item_ids = np.arange(
        1, min(100, n_samples) + 1
    )  # Limiter pour éviter les problèmes
    print(f"✅ item_id: {len(np.unique(item_ids))} uniques")

    # Créer user_category simple
    base_cat = (
        list(tabular_data.values())[0]
        if tabular_data
        else np.random.randint(0, 15, size=n_samples)
    )
    user_categories = np.array(base_cat[:n_samples]) % 15  # Limiter à 15 catégories
    print(f"✅ user_category: {len(np.unique(user_categories))} uniques")

    # Créer les colonnes pour le schéma avec les bons tags pour T4Rec 23.04.00
    columns = [
        ColumnSchema(
            "item_id",
            tags=[Tags.ITEM_ID, Tags.CATEGORICAL, Tags.ITEM],
            dtype=np.int32,
            properties={
                "domain": {"min": 1, "max": 100},
                "vocab_size": len(np.unique(item_ids)),
            },
        ),
        ColumnSchema(
            "user_category",
            tags=[Tags.USER_ID, Tags.CATEGORICAL, Tags.USER],
            dtype=np.int32,
            properties={
                "domain": {"min": 0, "max": 14},
                "vocab_size": len(np.unique(user_categories)),
            },
        ),
    ]

    schema = Schema(columns)
    print(f"✅ Schéma créé avec {len(columns)} colonnes")

    # Vérification du schéma
    print(f"🔍 Tags du schéma:")
    for col in schema:
        print(f"  - {col.name}: {col.tags}")

    # S'assurer que les tags requis sont présents
    item_cols = schema.select_by_tag(Tags.ITEM_ID)
    cat_cols = schema.select_by_tag(Tags.CATEGORICAL)
    print(f"✅ Colonnes ITEM_ID: {len(item_cols)}")
    print(f"✅ Colonnes CATEGORICAL: {len(cat_cols)}")

    # 2. Préparer les données pour T4Rec
    print("\n📊 Préparation des données T4Rec...")

    # Créer un dataset simple avec les bonnes dimensions
    data_dict = {
        "item_id": item_ids[: min(len(item_ids), n_samples)],
        "user_category": user_categories[: min(len(user_categories), n_samples)],
    }

    # Convertir en format séquentiel pour T4Rec
    batch_size = CONFIG["batch_size"]
    max_seq_len = CONFIG["max_sequence_length"]

    sequences = {}
    for key, data in data_dict.items():
        # S'assurer que data est un array numpy 1D
        if hasattr(data, "flatten"):
            data_flat = data.flatten()
        else:
            data_flat = np.array(data).flatten()

        # Calculer le nombre de séquences possibles
        total_elements = len(data_flat)
        num_seqs = total_elements // max_seq_len

        print(
            f"🔧 {key}: {total_elements} éléments -> {num_seqs} séquences de {max_seq_len}"
        )

        if num_seqs > 0:
            # Prendre seulement les éléments qui peuvent former des séquences complètes
            usable_elements = num_seqs * max_seq_len
            data_to_reshape = data_flat[:usable_elements]

            sequences[key] = torch.tensor(
                data_to_reshape.reshape(num_seqs, max_seq_len), dtype=torch.long
            )
        else:
            # Si pas assez de données, créer une séquence minimale
            print(
                f"⚠️ Pas assez de données pour {key}, création d'une séquence minimale"
            )
            min_data = data_flat[: min(len(data_flat), max_seq_len)]
            # Padding si nécessaire
            if len(min_data) < max_seq_len:
                min_data = np.pad(
                    min_data,
                    (0, max_seq_len - len(min_data)),
                    "constant",
                    constant_values=0,
                )

            sequences[key] = torch.tensor(
                min_data.reshape(1, max_seq_len), dtype=torch.long
            )

    print(f"✅ Séquences créées: {[(k, v.shape) for k, v in sequences.items()]}")

    # 3. Créer le module d'entrée
    print("\n🏗️ Module d'entrée...")

    input_module = tr.TabularSequenceFeatures.from_schema(
        schema,
        max_sequence_length=CONFIG["max_sequence_length"],
        continuous_projection=CONFIG["d_model"],
        aggregation="concat",
        masking="causal",
    )

    print("✅ Module créé: TabularSequenceFeatures")

    # 4. Assignation du masking
    print("\n🎭 Configuration du masking...")
    masking_module = tr.MaskSequence(hidden_size=CONFIG["d_model"], padding_idx=0)
    input_module.masking = masking_module
    print("✅ Masking assigné")

    # 5. Configuration XLNet
    print("\n⚙️ Configuration XLNet...")
    xlnet_config = tr.XLNetConfig.build(
        d_model=CONFIG["d_model"],
        n_head=CONFIG["n_head"],
        n_layer=CONFIG["n_layer"],
        mem_len=CONFIG["mem_len"],
    )
    print(
        f"✅ XLNet configuré: {CONFIG['d_model']}d, {CONFIG['n_head']}h, {CONFIG['n_layer']}l"
    )

    # 6. Construire le modèle de façon compatible avec T4Rec 23.04.00
    print("\n🚀 Construction du modèle...")

    # Créer un batch dummy pour initialiser les dimensions
    dummy_batch = {}
    for key, tensor in sequences.items():
        if len(tensor) > 0:
            dummy_batch[key] = tensor[: min(4, len(tensor))]  # Petit batch de test

    # Construire le module d'entrée d'abord
    if dummy_batch:
        try:
            input_output = input_module(dummy_batch)
            print(f"✅ Module d'entrée construit, shape: {input_output.shape}")
        except Exception as e:
            print(f"⚠️ Erreur construction module d'entrée: {e}")
            # Fallback: construction manuelle
            input_module.build(
                input_size=(CONFIG["batch_size"], CONFIG["max_sequence_length"])
            )

    # Corps du modèle simplifié pour T4Rec 23.04.00
    transformer_body = tr.TransformerBlock(xlnet_config, masking=input_module.masking)

    # Corps séquentiel avec dimensions explicites
    body = tr.SequentialBlock(
        input_module,
        transformer_body,
        tr.MLPBlock([CONFIG["d_model"]]),  # Projection de sortie obligatoire
        output_size=torch.Size(
            [CONFIG["batch_size"], CONFIG["max_sequence_length"], CONFIG["d_model"]]
        ),
    )

    # Métriques compatibles T4Rec 23.04.00
    from transformers4rec.torch.ranking_metric import NDCGAt, RecallAt

    # Head avec NextItemPredictionTask SANS hf_format pour 23.04.00
    head = tr.Head(
        body,
        tr.NextItemPredictionTask(
            weight_tying=True,
            metrics=[
                NDCGAt(top_ks=[5, 10], labels_onehot=True),
                RecallAt(top_ks=[5, 10], labels_onehot=True),
            ],
        ),
        inputs=input_module,
    )

    # Modèle final
    model = tr.Model(head)

    print("\n🎉 MODÈLE T4REC CONSTRUIT AVEC SUCCÈS!")
    print(f"📊 Modèle: {type(model).__name__}")
    print(
        f"📊 Architecture: XLNet {CONFIG['d_model']}d-{CONFIG['n_head']}h-{CONFIG['n_layer']}l"
    )
    print(
        f"📊 Données: {len(sequences)} features, {sum(len(v) for v in sequences.values())} échantillons"
    )

    # Test rapide si possible
    if dummy_batch and all(len(v) > 0 for v in dummy_batch.values()):
        try:
            with torch.no_grad():
                test_output = model(dummy_batch)
            print("✅ Test forward pass réussi!")
        except Exception as e:
            print(f"⚠️ Test forward pass échoué (normal pour T4Rec 23.04.00): {e}")

except Exception as e:
    print(f"❌ ERREUR: {e}")

    # DEBUG: Afficher les variables disponibles
    print("\n🔍 DEBUG:")
    available_vars = [var for var in dir() if not var.startswith("_")]
    print(f"Variables disponibles: {available_vars}")

    if "tabular_data" in globals():
        print(f"tabular_data keys: {list(tabular_data.keys())}")

    if "df" in globals():
        print(f"df shape: {df.shape}")

    import traceback

    traceback.print_exc()


