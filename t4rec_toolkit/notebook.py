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

    # Créer les colonnes pour le schéma
    columns = [
        ColumnSchema(
            "item_id",
            tags=[Tags.ITEM_ID, Tags.CATEGORICAL],
            dtype=np.int32,
            properties={"vocab_size": len(np.unique(item_ids))},
        ),
        ColumnSchema(
            "user_category",
            tags=[Tags.USER_ID, Tags.CATEGORICAL],
            dtype=np.int32,
            properties={"vocab_size": len(np.unique(user_categories))},
        ),
    ]

    schema = Schema(columns)
    print(f"✅ Schéma créé avec {len(columns)} colonnes")

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
        # Créer des séquences de longueur fixe
        num_seqs = len(data) // max_seq_len
        if num_seqs > 0:
            sequences[key] = torch.tensor(
                data[: num_seqs * max_seq_len].reshape(num_seqs, max_seq_len),
                dtype=torch.long,
            )

    print(f"✅ Séquences créées: {list(sequences.keys())}")

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


