# 🚀 T4REC XLNET - FEATURES CATÉGORIELLES UNIQUEMENT - STABLE
print("🚀 T4REC XLNET - FEATURES CATÉGORIELLES UNIQUEMENT - STABLE")
print("=" * 65)

import torch
import transformers4rec.torch as tr
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

try:
    # === CONFIGURATION ===
    CONFIG = {
        'd_model': 128,
        'n_head': 4,
        'n_layer': 2,
        'max_sequence_length': 15,
        'mem_len': 30,
        'dropout': 0.1,
        'hidden_size': 128  # Pour MaskSequence
    }

    # === SCHÉMA T4REC ===
    print("📋 Création du schéma catégoriel...")

    from merlin.schema import Schema, ColumnSchema, Tags

    columns = [
        ColumnSchema("item_id", tags=[Tags.CATEGORICAL, Tags.ITEM, Tags.ID],
                     dtype=np.int32, properties={"domain": {"min": 1, "max": 100}}),
        ColumnSchema("user_category", tags=[Tags.CATEGORICAL, Tags.USER],
                     dtype=np.int32, properties={"domain": {"min": 1, "max": 20}})
    ]

    schema = Schema(columns)
    print(f"✅ Schéma défini avec {len(schema)} colonnes: {[col.name for col in schema]}")

    # === DONNÉES DUMMY ===
    print("\n📊 Génération de données d'exemple...")

    n_samples = 560
    item_ids = np.arange(n_samples) % 99 + 1
    user_categories = np.random.randint(1, 15, n_samples)

    t4rec_data = {
        "item_id": item_ids.astype(np.int64),
        "user_category": user_categories.astype(np.int64)
    }

    print(f"✅ Données prêtes: item_id ({len(np.unique(item_ids))} uniques), user_category ({len(np.unique(user_categories))} uniques)")

    # === MODULE D'ENTRÉE ===
    print("\n🏗️ Construction du module d'entrée...")

    input_module = tr.TabularSequenceFeatures.from_schema(
        schema=schema,
        max_sequence_length=CONFIG['max_sequence_length'],
        aggregation="concat",
        masking=None,
        automatic_build=False
    )
    print(f"✅ Module d'entrée: {type(input_module).__name__}")

    # === MASKING ===
    print("\n🎭 Configuration du masking...")

    from transformers4rec.torch.masking import MaskSequence

    try:
        masking_module = MaskSequence(
            schema=schema.select_by_tag(Tags.ITEM),
            hidden_size=CONFIG['hidden_size'],
            max_sequence_length=CONFIG['max_sequence_length'],
            masking_prob=0.2,
            padding_idx=0
        )
        input_module.masking = masking_module
        print(f"✅ Masking activé: {type(masking_module).__name__}")
    except Exception as err:
        print(f"⚠️ Erreur masking: {err}")
        input_module.masking = None

    # === CONFIGURATION XLNET ===
    print("\n⚙️ Configuration du modèle XLNet...")

    xlnet_config = tr.XLNetConfig.build(
        d_model=CONFIG['d_model'],
        n_head=CONFIG['n_head'],
        n_layer=CONFIG['n_layer'],
        total_seq_length=CONFIG['max_sequence_length'],
        mem_len=CONFIG['mem_len'],
        dropout=CONFIG['dropout'],
        attn_type='bi',
        initializer_range=0.02,
        hidden_act='gelu',
        layer_norm_eps=1e-12
    )

    print(f"✅ XLNet configuré: {CONFIG['d_model']}d, {CONFIG['n_head']} heads, {CONFIG['n_layer']} layers")

    # === CORPS DU MODÈLE ===
    print("\n🧱 Construction du corps du modèle...")

    transformer_block = tr.Block(
        tr.TransformerBlock(xlnet_config, masking=input_module.masking),
        output_size=CONFIG['d_model']
    )

    body = tr.SequentialBlock(
        input_module,
        transformer_block
    )

    print(f"✅ Corps du modèle prêt: {type(body).__name__}")

    # === TÊTE DU MODÈLE ===
    print("\n🧠 Ajout de la tête NextItemPredictionTask...")

    from transformers4rec.torch.ranking_metric import NDCGAt, RecallAt

    task = tr.NextItemPredictionTask(
        weight_tying=True,
        metrics=[
            NDCGAt(top_ks=[5, 10], labels_onehot=True),
            RecallAt(top_ks=[5, 10], labels_onehot=True)
        ],
        loss_function="cross_entropy"
    )

    head = tr.Head(
        body=body,
        task=task,
        inputs=input_module
    )

    # === MODÈLE FINAL ===
    print("\n🚀 Initialisation du modèle T4Rec...")

    model = tr.Model(head)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"✅ Modèle T4Rec construit avec {n_params:,} paramètres")

    # === TEST AVEC BATCH DUMMY ===
    print("\n🧪 Test du modèle sur un batch...")

    batch_size = 16
    sequenced_batch = {}

    for key, data in t4rec_data.items():
        tensor = torch.tensor(data[:batch_size], dtype=torch.long)
        seq_tensor = tensor.unsqueeze(1).expand(-1, CONFIG['max_sequence_length'])
        sequenced_batch[key] = seq_tensor

    print(f"📦 Batch de test: {[(k, v.shape) for k, v in sequenced_batch.items()]}")

    model.eval()
    with torch.no_grad():
        try:
            output = model(sequenced_batch)
            print(f"✅ Test réussi ! Output shape: {output.shape}")
        except Exception as e:
            print(f"❌ Test échoué : {e}")
            import traceback
            traceback.print_exc()

except Exception as e:
    print(f"\n❌ ERREUR GÉNÉRALE : {e}")
    import traceback
    traceback.print_exc()
    if 'schema' in locals():
        try:
            print(f"📑 Schéma avec {len(schema)} colonnes : {[col.name for col in schema]}")
        except Exception as debug_err:
            print(f"⚠️ Impossible d'accéder aux colonnes : {debug_err}")

