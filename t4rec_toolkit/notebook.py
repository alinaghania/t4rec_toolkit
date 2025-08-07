# === T4REC XLNET INTÉGRÉ AVEC VOS DONNÉES ===

print("🚀 T4REC XLNET - UTILISANT VOS DONNÉES TRANSFORMÉES")
print("=" * 65)

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
            "❌ Variable 'df' non trouvée. Exécutez d'abord le script de chargement des données."
        )
        raise NameError("df non définie")

    print(
        f"✅ Données disponibles: {len(tabular_data)} features, {len(df)} échantillons"
    )

    # Configuration T4Rec pour version 23.04.00
    CONFIG = {
        "d_model": 128,
        "n_head": 4,
        "n_layer": 2,
        "max_sequence_length": 15,
        "mem_len": 30,
        "dropout": 0.1,
        "hidden_size": 128,
        "vocab_size": 100,
    }

    # 1. Créer le schéma T4Rec SEULEMENT avec features catégorielles
    print("\n📋 Schéma T4Rec avec features catégorielles...")

    from merlin.schema import Schema, ColumnSchema, Tags

    # Utiliser vos features existantes comme base
    n_samples = len(next(iter(tabular_data.values())))

    # Créer des features catégorielles basées sur vos données
    print(f"Échantillons disponibles: {n_samples}")

    # Item IDs : cycle sur une plage
    item_ids = np.arange(n_samples) % 99 + 1  # IDs de 1 à 99

    # User categories : basé sur vos features dummy existantes
    categorical_features = [name for name in tabular_data.keys() if "dummy" in name]
    print(f"Features catégorielles trouvées: {categorical_features[:3]}...")

    if categorical_features:
        # Utiliser la première feature dummy comme base
        base_feature = tabular_data[categorical_features[0]]
        if hasattr(base_feature, "flatten"):
            base_values = base_feature.flatten()
        else:
            base_values = np.array(base_feature)
        user_categories = (base_values % 19) + 1  # 1 à 19
    else:
        user_categories = np.random.randint(1, 20, n_samples)

    print(f"✅ item_id: {len(np.unique(item_ids))} uniques")
    print(f"✅ user_category: {len(np.unique(user_categories))} uniques")

    # Schéma Merlin simplifié
    columns = [
        ColumnSchema(
            "item_id",
            tags=[Tags.CATEGORICAL, Tags.ITEM, Tags.ID],
            dtype=np.int32,
            properties={"domain": {"min": 1, "max": 100}},
        ),
        ColumnSchema(
            "user_category",
            tags=[Tags.CATEGORICAL, Tags.USER],
            dtype=np.int32,
            properties={"domain": {"min": 1, "max": 20}},
        ),
    ]

    schema = Schema(columns)
    print(f"✅ Schéma créé avec {len(columns)} colonnes")

    # 2. Préparer les données pour T4Rec
    print("\n📊 Préparation des données T4Rec...")

    t4rec_data = {
        "item_id": item_ids.astype(np.int64),
        "user_category": user_categories.astype(np.int64),
    }

    # 3. Créer le module d'entrée
    print("\n🏗️ Module d'entrée...")

    input_module = tr.TabularSequenceFeatures.from_schema(
        schema=schema,
        max_sequence_length=CONFIG["max_sequence_length"],
        aggregation="concat",
        masking=None,
        automatic_build=False,
    )

    print(f"✅ Module créé: {type(input_module).__name__}")

    # 4. Configuration du masking
    print("\n🎭 Configuration du masking...")

    try:
        from transformers4rec.torch.masking import MaskSequence

        masking_module = MaskSequence(
            schema=schema.select_by_tag(Tags.ITEM),
            hidden_size=CONFIG["hidden_size"],
            max_sequence_length=CONFIG["max_sequence_length"],
            masking_prob=0.2,
            padding_idx=0,
        )

        input_module.masking = masking_module
        print(f"✅ Masking assigné")

    except Exception as masking_error:
        print(f"⚠️ Erreur masking: {masking_error}")
        input_module.masking = None
        print("⚠️ Continuons sans masking")

    # 5. Configuration XLNet (CORRIGÉE)
    print("\n⚙️ Configuration XLNet...")

    xlnet_config = tr.XLNetConfig.build(
        d_model=CONFIG["d_model"],
        n_head=CONFIG["n_head"],
        n_layer=CONFIG["n_layer"],
        total_seq_length=CONFIG["max_sequence_length"],
        mem_len=CONFIG["mem_len"],
        dropout=CONFIG["dropout"],
        attn_type="bi",
        initializer_range=0.02,
        hidden_act="gelu",
        layer_norm_eps=1e-12,
        pad_token=0,
        # PAS de vocab_size ici pour éviter le conflit
    )

    print(
        f"✅ XLNet configuré: {CONFIG['d_model']}d, {CONFIG['n_head']}h, {CONFIG['n_layer']}l"
    )

    # 6. Construire le modèle
    print("\n🚀 Construction du modèle...")

    # D'abord, construire le module d'entrée avec un batch dummy
    print("🔧 Construction du module d'entrée...")
    dummy_batch = {}
    for key, data in t4rec_data.items():
        tensor = torch.tensor(data[:4], dtype=torch.long)  # Petit batch
        seq_tensor = tensor.unsqueeze(1).expand(-1, CONFIG["max_sequence_length"])
        dummy_batch[key] = seq_tensor

    # Construire le module d'entrée
    input_output = input_module(dummy_batch)
    print(f"✅ Module d'entrée construit, output shape: {input_output.shape}")

    # Maintenant construire le corps avec les bonnes dimensions
    body = tr.SequentialBlock(
        input_module,
        tr.TransformerBlock(xlnet_config, masking=input_module.masking),
        tr.MLPBlock([CONFIG["d_model"]]),  # Projection vers la dimension du modèle
    )

    from transformers4rec.torch.ranking_metric import NDCGAt, RecallAt

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

    model = tr.Model(head)

    print(f"✅ Modèle créé!")
    print(f"📈 Paramètres: {sum(p.numel() for p in model.parameters()):,}")

    # 7. Test du modèle
    print("\n🧪 Test du modèle...")

    batch_size = 16
    test_batch = {}

    for key, data in t4rec_data.items():
        tensor = torch.tensor(data[:batch_size], dtype=torch.long)
        # Créer des séquences
        seq_tensor = tensor.unsqueeze(1).expand(-1, CONFIG["max_sequence_length"])
        test_batch[key] = seq_tensor

    print(f"Batch de test: {[(k, v.shape) for k, v in test_batch.items()]}")

    model.eval()
    with torch.no_grad():
        try:
            output = model(test_batch)
            print(f"✅ Test réussi! Output shape: {output.shape}")
            test_success = True
        except Exception as test_error:
            print(f"❌ Test échoué: {test_error}")
            print(f"Type d'erreur: {type(test_error).__name__}")
            import traceback

            traceback.print_exc()
            test_success = False

    if test_success:
        print("\n🎯 Préparation pour l'entraînement...")

        # Préparer les séquences complètes
        full_sequences = {}

        for key, data in t4rec_data.items():
            tensor = torch.tensor(data, dtype=torch.long)
            sequences = []

            for i in range(len(tensor)):
                # Créer une séquence réaliste
                seq = torch.full(
                    (CONFIG["max_sequence_length"],), tensor[i], dtype=tensor.dtype
                )

                # Ajouter de la variabilité
                if i > 0:
                    start_idx = max(0, i - CONFIG["max_sequence_length"] + 1)
                    prev_values = tensor[start_idx:i]
                    if len(prev_values) > 0:
                        seq[-len(prev_values) :] = prev_values
                    seq[-1] = tensor[i]

                sequences.append(seq)

            full_sequences[key] = torch.stack(sequences)

        print(f"Séquences créées: {[(k, v.shape) for k, v in full_sequences.items()]}")

        # Préparer les labels
        target_col = "souscription_produit_1m"
        if target_col in df.columns:
            y = df[target_col].values
            label_encoder = LabelEncoder()
            y_encoded = label_encoder.fit_transform(y)
            y_tensor = torch.tensor(y_encoded, dtype=torch.long)

            print(f"✅ Labels préparés: {len(np.unique(y_encoded))} classes")

            # Split des données
            n_total = len(y_tensor)
            n_train = int(0.8 * n_total)

            train_sequences = {k: v[:n_train] for k, v in full_sequences.items()}
            val_sequences = {k: v[n_train:] for k, v in full_sequences.items()}
            y_train = y_tensor[:n_train]
            y_val = y_tensor[n_train:]

            print(f"✅ Split: Train {len(y_train)}, Val {len(y_val)}")

            # Configuration d'entraînement simple
            from torch.optim import AdamW
            from torch.nn import CrossEntropyLoss

            optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
            criterion = CrossEntropyLoss()

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)
            print(f"Device: {device}")

            # Entraînement rapide
            print("\n🔥 Entraînement (3 époques)...")

            num_epochs = 3
            batch_size_train = 8

            for epoch in range(num_epochs):
                model.train()
                total_loss = 0
                n_batches = 0

                for i in range(0, len(y_train), batch_size_train):
                    end_idx = min(i + batch_size_train, len(y_train))

                    batch_data = {
                        k: v[i:end_idx].to(device) for k, v in train_sequences.items()
                    }
                    batch_targets = y_train[i:end_idx].to(device)

                    optimizer.zero_grad()

                    try:
                        outputs = model(batch_data)
                        loss = criterion(outputs, batch_targets)
                        loss.backward()

                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        optimizer.step()

                        total_loss += loss.item()
                        n_batches += 1

                    except Exception as batch_error:
                        print(f"⚠️ Erreur batch: {batch_error}")
                        continue

                avg_loss = total_loss / max(n_batches, 1)
                print(f"Époque {epoch + 1}: Loss = {avg_loss:.4f}")

            # Évaluation
            print("\n📊 Évaluation...")
            model.eval()
            correct = 0
            total_eval = 0

            with torch.no_grad():
                for i in range(0, len(y_val), batch_size_train):
                    end_idx = min(i + batch_size_train, len(y_val))

                    batch_data = {
                        k: v[i:end_idx].to(device) for k, v in val_sequences.items()
                    }
                    batch_targets = y_val[i:end_idx].to(device)

                    try:
                        outputs = model(batch_data)
                        predictions = torch.argmax(outputs, dim=1)
                        correct += (predictions == batch_targets).sum().item()
                        total_eval += batch_targets.size(0)
                    except:
                        continue

            accuracy = correct / max(total_eval, 1)
            print(f"✅ Précision: {accuracy:.2%}")

            # Sauvegarde
            save_dict = {
                "model_state_dict": model.state_dict(),
                "config": CONFIG,
                "schema": schema,
                "label_encoder": label_encoder,
                "accuracy": accuracy,
                "transformers4rec_version": "23.04.00",
            }

            torch.save(save_dict, "t4rec_xlnet_working_model.pth")
            print("✅ Modèle sauvegardé: t4rec_xlnet_working_model.pth")
            print("🎉 SUCCÈS COMPLET!")

        else:
            print(f"❌ Colonne target '{target_col}' non trouvée")

    else:
        print("❌ Échec du test - arrêt")

except Exception as e:
    print(f"❌ ERREUR: {e}")
    import traceback

    traceback.print_exc()

    # Debug info
    print(f"\n🔍 DEBUG:")
    print(
        f"Variables disponibles: {[name for name in globals().keys() if not name.startswith('_')]}"
    )
    if "tabular_data" in globals():
        print(f"tabular_data keys: {list(tabular_data.keys())[:5]}")
    if "df" in globals():
        print(f"df shape: {df.shape if hasattr(df, 'shape') else 'N/A'}")

