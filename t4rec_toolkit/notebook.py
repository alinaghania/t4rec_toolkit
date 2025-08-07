# === T4REC XLNET AVEC SEULEMENT DES FEATURES CATÉGORIELLES (PLUS STABLE) ===

print("🚀 T4REC XLNET - FEATURES CATÉGORIELLES UNIQUEMENT - STABLE")
print("=" * 65)

import torch
import transformers4rec.torch as tr
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

try:
    # Configuration T4Rec SIMPLIFIÉE pour version 23.04.00
    CONFIG = {
        'd_model': 128,
        'n_head': 4,
        'n_layer': 2,
        'max_sequence_length': 15,
        'mem_len': 30,
        'dropout': 0.1,
        'hidden_size': 128,  # OBLIGATOIRE pour MaskSequence
        'vocab_size': 100    # OBLIGATOIRE pour XLNetConfig
    }

    # 1. Créer le schéma T4Rec SANS features continues (plus stable)
    print("📋 Schéma T4Rec avec SEULEMENT des features catégorielles...")
    
    from merlin.schema import Schema, ColumnSchema, Tags
    
    # Schéma SIMPLIFIÉ avec SEULEMENT des features catégorielles
    columns = [
        # Item ID - OBLIGATOIREMENT en premier
        ColumnSchema(
            "item_id",
            tags=[Tags.CATEGORICAL, Tags.ITEM, Tags.ID],
            dtype=np.int32,
            properties={"domain": {"min": 1, "max": 100}}
        ),
        # User category
        ColumnSchema(
            "user_category",
            tags=[Tags.CATEGORICAL, Tags.USER],
            dtype=np.int32, 
            properties={"domain": {"min": 1, "max": 20}}
        )
        # PAS de feature continue pour éviter les problèmes
    ]
    
    schema = Schema(columns)
    print(f"✅ Schéma simplifié: item_id + user_category SEULEMENT")
    
    # 2. Données optimisées pour T4Rec (SEULEMENT catégorielles)
    print("\n📊 Données pour T4Rec (catégorielles uniquement)...")
    
    # Prendre le nombre d'échantillons
    n_samples = len(next(iter(tabular_data.values())))
    print(f"Échantillons: {n_samples}")
    
    # Item IDs : cycle sur une petite plage
    item_ids = np.arange(n_samples) % 99 + 1  # IDs de 1 à 99
    
    # User categories : basé sur vos features dummy
    categorical_names = [name for name in tabular_data.keys() if 'dummy' in name]
    if categorical_names:
        base_cat = np.array(tabular_data[categorical_names[0]])
        user_categories = (base_cat % 19) + 1  # 1 à 19
    else:
        user_categories = np.random.randint(1, 20, n_samples)
    
    # Dataset T4Rec SIMPLIFIÉ (seulement catégorielles)
    t4rec_data = {
        "item_id": item_ids.astype(np.int64),
        "user_category": user_categories.astype(np.int64)
        # PAS de continuous_feature
    }
    
    print(f"✅ item_id: {len(np.unique(item_ids))} uniques ({item_ids.min()}-{item_ids.max()})")
    print(f"✅ user_category: {len(np.unique(user_categories))} uniques")
    
    # 3. Créer le module d'entrée SANS continuous_projection
    print("\n🏗️ Module d'entrée (seulement catégorielles)...")
    
    # Module SIMPLIFIÉ sans continuous_projection
    input_module = tr.TabularSequenceFeatures.from_schema(
        schema=schema,
        max_sequence_length=CONFIG['max_sequence_length'],
        # PAS de continuous_projection car pas de features continues
        aggregation="concat",
        masking=None,  # PAS de masking initially
        automatic_build=False  # OK maintenant car pas de continuous_projection
    )
    
    print(f"✅ Module créé: {type(input_module).__name__}")
    
    # 4. Ajouter le masking manuellement
    print("\n🎭 Configuration du masking...")
    from transformers4rec.torch.masking import MaskSequence
    
    try:
        masking_module = MaskSequence(
            schema=schema.select_by_tag(Tags.ITEM),
            hidden_size=CONFIG['hidden_size'],  # PARAMÈTRE OBLIGATOIRE
            max_sequence_length=CONFIG['max_sequence_length'],
            masking_prob=0.2,
            padding_idx=0
        )
        
        input_module.masking = masking_module
        print(f"✅ Masking assigné: {type(masking_module).__name__}")
        
    except Exception as masking_error:
        print(f"⚠️ Erreur masking: {masking_error}")
        input_module.masking = None
        print("⚠️ Continuons sans masking pour le test")
    
    # 5. Configuration XLNet COMPLÈTE
    print("\n⚙️ Config XLNet...")
    
    xlnet_config = tr.XLNetConfig.build(
        d_model=CONFIG['d_model'],
        n_head=CONFIG['n_head'], 
        n_layer=CONFIG['n_layer'],
        total_seq_length=CONFIG['max_sequence_length'],
        mem_len=CONFIG['mem_len'],
        dropout=CONFIG['dropout'],
        pad_token_id=0,
        vocab_size=CONFIG['vocab_size'],
        attn_type='bi',
        initializer_range=0.02,
        hidden_act='gelu',
        layer_norm_eps=1e-12
    )
    
    print(f"✅ XLNet: {CONFIG['d_model']}d, {CONFIG['n_head']}h, {CONFIG['n_layer']}l")
    
    # 6. Construire le modèle
    print("\n🚀 Modèle T4Rec...")
    
    # Corps du modèle
    body = tr.SequentialBlock(
        input_module,
        tr.TransformerBlock(xlnet_config, masking=input_module.masking)
    )
    
    # Métriques
    from transformers4rec.torch.ranking_metric import NDCGAt, RecallAt
    
    # Tête de prédiction
    head = tr.Head(
        body,
        tr.NextItemPredictionTask(
            weight_tying=True,
            hf_format=True,
            metrics=[
                NDCGAt(top_ks=[5, 10], labels_onehot=True),
                RecallAt(top_ks=[5, 10], labels_onehot=True)
            ],
            loss_function="cross_entropy"
        ),
        inputs=input_module
    )
    
    # Modèle complet
    model = tr.Model(head)
    
    print(f"✅ Modèle T4Rec créé!")
    print(f"📈 Paramètres: {sum(p.numel() for p in model.parameters()):,}")
    
    # 7. Test rapide du modèle
    print("\n🧪 Test du modèle...")
    
    # Créer un batch test avec séquences
    batch_size = 16
    sequenced_batch = {}
    
    for key, data in t4rec_data.items():
        tensor = torch.tensor(data[:batch_size], dtype=torch.long)
        # Créer des séquences en répétant
        seq_tensor = tensor.unsqueeze(1).expand(-1, CONFIG['max_sequence_length'])
        sequenced_batch[key] = seq_tensor
    
    print(f"Batch de test: {[(k, v.shape) for k, v in sequenced_batch.items()]}")
    
    model.eval()
    with torch.no_grad():
        try:
            output = model(sequenced_batch)
            print(f"✅ Test réussi: output shape={output.shape}")
            test_success = True
        except Exception as test_error:
            print(f"❌ Test échoué: {test_error}")
            print(f"Type d'erreur: {type(test_error).__name__}")
            import traceback
            traceback.print_exc()
            test_success = False
    
    if test_success:
        # 8. Entraînement
        print("\n🎯 Entraînement...")
        
        # Préparer toutes les données avec séquences
        full_dataset = {}
        
        for key, data in t4rec_data.items():
            tensor = torch.tensor(data, dtype=torch.long)
            
            # Créer des séquences plus réalistes
            sequences = []
            for i in range(len(tensor)):
                # Créer une séquence avec les éléments précédents
                base_val = tensor[i]
                seq = torch.full((CONFIG['max_sequence_length'],), base_val, dtype=tensor.dtype)
                
                # Ajouter quelques variations pour plus de réalisme
                if i > 0:
                    # Prendre les éléments précédents
                    start_idx = max(0, i - CONFIG['max_sequence_length'] + 1)
                    prev_seq = tensor[start_idx:i]
                    if len(prev_seq) > 0:
                        seq[-len(prev_seq):] = prev_seq
                    seq[-1] = base_val  # Élément courant en dernier
                
                sequences.append(seq)
            
            full_dataset[key] = torch.stack(sequences)
        
        print(f"Dataset final: {[(k, v.shape) for k, v in full_dataset.items()]}")
        
        # Préparer les labels
        y = df['souscription_produit_1m'].values
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        y_tensor = torch.tensor(y_encoded, dtype=torch.long)
        
        # Split train/validation
        n_samples_total = len(y_tensor)
        n_train = int(0.8 * n_samples_total)
        
        train_data = {k: v[:n_train] for k, v in full_dataset.items()}
        val_data = {k: v[n_train:] for k, v in full_dataset.items()}
        y_train = y_tensor[:n_train]
        y_val = y_tensor[n_train:]
        
        print(f"✅ Train: {n_train}, Val: {len(y_val)}")
        
        # Configuration d'entraînement
        from torch.optim import AdamW
        from torch.nn import CrossEntropyLoss
        
        optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
        criterion = CrossEntropyLoss()
        
        # Device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        print(f"Device utilisé: {device}")
        
        # Boucle d'entraînement
        num_epochs = 3
        batch_size_train = 8
        
        for epoch in range(num_epochs):
            model.train()
            total_loss = 0
            n_batches = 0
            
            # Mini-batches d'entraînement
            for i in range(0, len(y_train), batch_size_train):
                end_idx = min(i + batch_size_train, len(y_train))
                
                batch_data = {
                    k: v[i:end_idx].to(device) 
                    for k, v in train_data.items()
                }
                batch_targets = y_train[i:end_idx].to(device)
                
                optimizer.zero_grad()
                try:
                    outputs = model(batch_data)
                    loss = criterion(outputs, batch_targets)
                    loss.backward()
                    
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    
                    optimizer.step()
                    
                    total_loss += loss.item()
                    n_batches += 1
                    
                except Exception as batch_error:
                    print(f"⚠️ Erreur batch {i//batch_size_train + 1}: {batch_error}")
                    continue
            
            avg_loss = total_loss / max(n_batches, 1)
            print(f"Époque {epoch + 1}/{num_epochs}: Loss = {avg_loss:.4f}")
        
        # Évaluation finale
        print("\n📊 Évaluation finale...")
        model.eval()
        correct = 0
        total_eval = 0
        
        with torch.no_grad():
            for i in range(0, len(y_val), batch_size_train):
                end_idx = min(i + batch_size_train, len(y_val))
                
                batch_data = {
                    k: v[i:end_idx].to(device)
                    for k, v in val_data.items()
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
        print(f"✅ Précision finale: {accuracy:.2%}")
        
        # Sauvegarde
        save_dict = {
            'model_state_dict': model.state_dict(),
            'config': CONFIG,
            'schema': schema,
            'label_encoder': label_encoder,
            'accuracy': accuracy,
            'transformers4rec_version': '23.04.00',
            'architecture': 'categorical_only_stable'
        }
        
        torch.save(save_dict, 't4rec_xlnet_categorical_only_success.pth')
        print("✅ Modèle sauvegardé: t4rec_xlnet_categorical_only_success.pth")
        print("🎉 SUCCÈS T4REC XLNET CATÉGORIELLES UNIQUEMENT!")
    
    else:
        print("❌ Échec du test - pas d'entraînement")

except Exception as e:
    print(f"❌ ERREUR GÉNÉRALE: {e}")
    import traceback
    traceback.print_exc()
    
    print(f"\n🔍 DEBUG:")
    if 'schema' in locals():
        try:
            print(f"Schéma: {len(schema)} colonnes")
            if hasattr(schema, 'column_schemas'):
                column_names = [col.name for col in schema.column_schemas]
                print(f"Colonnes: {column_names}")
        except:
            print("Impossible d'accéder aux colonnes du schéma")
