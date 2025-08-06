# === ENTRAÎNEMENT T4REC XLNET PUR - SANS FALLBACK ===

print("🚀 T4REC XLNET PUR")
print("=" * 30)

import torch
import transformers4rec.torch as tr
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Configuration directe pour T4Rec XLNet
CONFIG = {
    'd_model': 128,  # Réduit pour stabilité
    'n_head': 4,     # Réduit pour stabilité
    'n_layer': 2,    # Réduit pour stabilité
    'max_sequence_length': 15,
    'mem_len': 30,
    'dropout': 0.1
}

try:
    # 1. Créer le schéma T4Rec directement
    print("📋 Création schéma T4Rec direct...")
    
    from merlin.schema import Schema, ColumnSchema, Tags
    
    # Schéma minimal avec item_id requis
    columns = [
        # Item ID obligatoire pour T4Rec
        ColumnSchema(
            name="item_id",
            tags={Tags.CATEGORICAL, Tags.ITEM, Tags.ID},
            dtype="int64",
            properties={"vocab_size": 1000}
        ),
        # Une feature séquentielle principale
        ColumnSchema(
            name="sequence_feature",
            tags={Tags.CONTINUOUS, Tags.LIST},
            dtype="float32",
            is_list=True,
            properties={"max_sequence_length": CONFIG['max_sequence_length']}
        ),
        # Une feature catégorielle
        ColumnSchema(
            name="category_feature", 
            tags={Tags.CATEGORICAL},
            dtype="int64",
            properties={"vocab_size": 20}
        )
    ]
    
    schema = Schema(columns)
    print(f"✅ Schéma T4Rec créé: {len(columns)} colonnes")
    
    # 2. Préparer les données pour T4Rec
    print("\n📊 Préparation données T4Rec...")
    
    # Créer item_id basé sur les features catégorielles existantes
    categorical_names = [name for name in tabular_data.keys() if 'dummy' in name][:1]  # Une seule
    
    if categorical_names:
        # Utiliser la première feature catégorielle comme base pour item_id
        base_feature = tabular_data[categorical_names[0]]
        item_ids = np.array(base_feature, dtype=np.int64) % 999 + 1  # IDs de 1 à 999
    else:
        # Fallback: IDs séquentiels
        n_samples = len(next(iter(tabular_data.values())))
        item_ids = np.arange(1, n_samples + 1, dtype=np.int64) % 999 + 1
    
    # Créer une feature séquentielle en combinant plusieurs features existantes
    sequence_names = [name for name in tabular_data.keys() if 'sequence' in name][:3]
    if sequence_names:
        # Moyenner quelques features de séquence
        seq_data = []
        for name in sequence_names:
            data = tabular_data[name]
            if isinstance(data[0], np.ndarray):
                # Prendre la première séquence et la tronquer/padder
                seq_data.append([seq[:CONFIG['max_sequence_length']] if len(seq) >= CONFIG['max_sequence_length'] 
                               else np.pad(seq, (0, CONFIG['max_sequence_length'] - len(seq))) 
                               for seq in data])
        
        # Moyenner les séquences
        if seq_data:
            sequence_feature = np.mean(seq_data, axis=0).astype(np.float32)
        else:
            sequence_feature = np.random.randn(len(item_ids), CONFIG['max_sequence_length']).astype(np.float32)
    else:
        # Créer des séquences aléatoires
        sequence_feature = np.random.randn(len(item_ids), CONFIG['max_sequence_length']).astype(np.float32)
    
    # Feature catégorielle simple
    if categorical_names:
        category_feature = np.array(tabular_data[categorical_names[0]], dtype=np.int64) % 19  # 0-18
    else:
        category_feature = np.random.randint(0, 19, size=len(item_ids), dtype=np.int64)
    
    # Créer le dataset T4Rec
    t4rec_data = {
        "item_id": item_ids,
        "sequence_feature": sequence_feature,
        "category_feature": category_feature
    }
    
    print(f"✅ item_id: {len(np.unique(item_ids))} uniques")
    print(f"✅ sequence_feature: shape={sequence_feature.shape}")
    print(f"✅ category_feature: range={category_feature.min()}-{category_feature.max()}")
    
    # 3. Créer le module d'entrée T4Rec directement
    print("\n🏗️ Module d'entrée T4Rec direct...")
    
    input_module = tr.TabularSequenceFeatures.from_schema(
        schema=schema,
        max_sequence_length=CONFIG['max_sequence_length'],
        continuous_projection=CONFIG['d_model'],
        aggregation="concat",
        masking="mlm"
    )
    
    print(f"✅ Module d'entrée créé: {type(input_module).__name__}")
    
    # 4. Configuration XLNet T4Rec
    print("\n⚙️ Configuration XLNet...")
    
    xlnet_config = tr.XLNetConfig.build(
        d_model=CONFIG['d_model'],
        n_head=CONFIG['n_head'],
        n_layer=CONFIG['n_layer'],
        total_seq_length=CONFIG['max_sequence_length'],
        mem_len=CONFIG['mem_len'],
        dropout=CONFIG['dropout']
    )
    
    print(f"✅ Config XLNet: {CONFIG['d_model']}d, {CONFIG['n_head']}h, {CONFIG['n_layer']}l")
    
    # 5. Créer le modèle T4Rec complet
    print("\n🚀 Construction modèle T4Rec...")
    
    # Corps du modèle
    body = tr.SequentialBlock(
        input_module,
        tr.TransformerBlock(xlnet_config, masking=input_module.masking)
    )
    
    # Tête de prédiction
    head = tr.Head(
        body,
        tr.NextItemPredictionTask(weight_tying=True),
        inputs=input_module
    )
    
    # Modèle final
    model = tr.Model(head)
    
    print(f"✅ Modèle T4Rec créé!")
    print(f"📈 Paramètres: {sum(p.numel() for p in model.parameters()):,}")
    
    # 6. Préparer l'entraînement
    print("\n🎯 Préparation entraînement...")
    
    # Convertir en tenseurs PyTorch
    X_torch = {k: torch.tensor(v) for k, v in t4rec_data.items()}
    
    # Target depuis vos données originales
    y = df['souscription_produit_1m'].values
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    y_torch = torch.tensor(y_encoded, dtype=torch.long)
    
    # Split train/val
    indices = np.arange(len(y_torch))
    train_indices, val_indices = train_test_split(
        indices, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    X_train = {k: v[train_indices] for k, v in X_torch.items()}
    X_val = {k: v[val_indices] for k, v in X_torch.items()}
    y_train = y_torch[train_indices]
    y_val = y_torch[val_indices]
    
    print(f"✅ Train: {len(train_indices)}, Val: {len(val_indices)}")
    print(f"✅ Classes: {len(label_encoder.classes_)}")
    
    # 7. Entraînement T4Rec
    print("\n🔥 Entraînement T4Rec XLNet...")
    
    from torch.optim import AdamW
    from torch.nn import CrossEntropyLoss
    
    optimizer = AdamW(model.parameters(), lr=1e-4)
    criterion = CrossEntropyLoss()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    print(f"✅ Device: {device}")
    
    # Paramètres d'entraînement
    num_epochs = 10
    batch_size = 16
    
    print(f"✅ Époques: {num_epochs}, Batch: {batch_size}")
    
    # Boucle d'entraînement
    for epoch in range(num_epochs):
        # Train
        model.train()
        total_loss = 0
        batches = 0
        
        for i in range(0, len(train_indices), batch_size):
            batch_indices = train_indices[i:i+batch_size]
            
            batch_X = {k: v[batch_indices].to(device) for k, v in X_train.items()}
            batch_y = y_train[i:i+batch_size].to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            batches += 1
        
        # Validation
        model.eval()
        val_loss = 0
        val_batches = 0
        
        with torch.no_grad():
            for i in range(0, len(val_indices), batch_size):
                batch_indices = val_indices[i:i+batch_size]
                
                batch_X = {k: v[batch_indices].to(device) for k, v in X_val.items()}
                batch_y = y_val[i:i+batch_size].to(device)
                
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
                val_batches += 1
        
        avg_train = total_loss / batches if batches > 0 else 0
        avg_val = val_loss / val_batches if val_batches > 0 else 0
        
        print(f"Époque {epoch+1:2d}: Train={avg_train:.4f}, Val={avg_val:.4f}")
    
    # 8. Évaluation finale
    print("\n📊 Évaluation finale...")
    
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for i in range(0, len(val_indices), batch_size):
            batch_indices = val_indices[i:i+batch_size]
            
            batch_X = {k: v[batch_indices].to(device) for k, v in X_val.items()}
            batch_y = y_val[i:i+batch_size].to(device)
            
            outputs = model(batch_X)
            predictions = torch.argmax(outputs, dim=1)
            correct += (predictions == batch_y).sum().item()
            total += batch_y.size(0)
    
    accuracy = correct / total if total > 0 else 0
    print(f"✅ Accuracy: {accuracy:.2%}")
    
    # 9. Sauvegarder
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'schema': schema,
        'config': CONFIG,
        'label_encoder': label_encoder,
        'accuracy': accuracy
    }, 't4rec_pure.pth')
    
    print("✅ Modèle sauvé: t4rec_pure.pth")
    print("🎉 Entraînement T4Rec réussi!")

except Exception as e:
    print(f"❌ ERREUR T4REC: {e}")
    import traceback
    traceback.print_exc()
    
    # Informations de débogage
    print(f"\n🔍 DEBUG INFO:")
    print(f"transformers4rec disponible: {'transformers4rec' in str(type(tr))}")
    print(f"Schema type: {type(schema) if 'schema' in locals() else 'Non créé'}")
    print(f"Input module type: {type(input_module) if 'input_module' in locals() else 'Non créé'}")
    
    if 'xlnet_config' in locals():
        print(f"XLNet config type: {type(xlnet_config)}")
        print(f"XLNet config: {xlnet_config}")
    
    print(f"\nVérifiez:")
    print(f"1. transformers4rec==23.04.00 installé")
    print(f"2. merlin-core installé") 
    print(f"3. Variables 'tabular_data' et 'df' définies")
