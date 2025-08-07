# === T4REC XLNET AVEC MODULE CATÉGORIEL CORRECT ===

print("🚀 T4REC XLNET - MODULE CATÉGORIEL")
print("=" * 40)

import torch
import transformers4rec.torch as tr
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

try:
    # Configuration T4Rec
    CONFIG = {
        'd_model': 128,
        'n_head': 4,
        'n_layer': 2,
        'max_sequence_length': 15,
        'mem_len': 30,
        'dropout': 0.1
    }

    # 1. Créer le schéma T4Rec avec focus sur les catégorielles
    print("📋 Schéma T4Rec avec catégorielles...")
    
    from merlin.schema import Schema, ColumnSchema, Tags
    
    # Le schéma DOIT avoir l'item_id comme PREMIÈRE colonne catégorielle
    columns = [
        # Item ID - OBLIGATOIREMENT en premier
        ColumnSchema(
            name="item_id",
            tags={Tags.CATEGORICAL, Tags.ITEM, Tags.ID},
            dtype="int64",
            properties={"vocab_size": 100}  # Réduit pour simplifier
        ),
        # Autres features catégorielles (contexte)
        ColumnSchema(
            name="user_category",
            tags={Tags.CATEGORICAL, Tags.USER},
            dtype="int64", 
            properties={"vocab_size": 20}
        ),
        # Features continues EN DERNIER
        ColumnSchema(
            name="continuous_feature",
            tags={Tags.CONTINUOUS},
            dtype="float32",
            properties={}
        )
    ]
    
    schema = Schema(columns)
    print(f"✅ Schéma: item_id + user_category + continuous")
    
    # 2. Données optimisées pour T4Rec
    print("\n📊 Données pour T4Rec...")
    
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
    
    # Feature continue : moyenne de vos features
    sequence_names = [name for name in tabular_data.keys() if 'sequence' in name]
    if sequence_names:
        # Prendre la moyenne des premières valeurs des séquences
        continuous_vals = []
        for name in sequence_names[:2]:  # Max 2 features
            data = tabular_data[name]
            if isinstance(data[0], np.ndarray):
                vals = [seq[0] if len(seq) > 0 else 0.0 for seq in data]
            else:
                vals = data
            continuous_vals.append(vals)
        continuous_feature = np.mean(continuous_vals, axis=0).astype(np.float32)
    else:
        continuous_feature = np.random.randn(n_samples).astype(np.float32)
    
    # Dataset T4Rec
    t4rec_data = {
        "item_id": item_ids.astype(np.int64),
        "user_category": user_categories.astype(np.int64), 
        "continuous_feature": continuous_feature
    }
    
    print(f"✅ item_id: {len(np.unique(item_ids))} uniques ({item_ids.min()}-{item_ids.max()})")
    print(f"✅ user_category: {len(np.unique(user_categories))} uniques") 
    print(f"✅ continuous_feature: shape={continuous_feature.shape}")
    
    # 3. Créer le module d'entrée SANS masking d'abord
    print("\n🏗️ Module d'entrée sans masking...")
    
    # Étape 1: Créer sans masking pour éviter l'erreur
    input_module = tr.TabularSequenceFeatures.from_schema(
        schema=schema,
        max_sequence_length=CONFIG['max_sequence_length'],
        continuous_projection=CONFIG['d_model'],
        aggregation="concat",
        masking=None  # PAS de masking initially
    )
    
    print(f"✅ Module créé: {type(input_module).__name__}")
    
    # Étape 2: Ajouter le masking manuellement
    from transformers4rec.torch.masking import MaskSequence
    
    # Créer le masking manuellement
    masking_module = MaskSequence(
        schema=schema.select_by_tag(Tags.ITEM),  # Seulement item_id
        max_sequence_length=CONFIG['max_sequence_length'],
        masking_prob=0.2
    )
    
    # Assigner le masking au module
    input_module.masking = masking_module
    
    print(f"✅ Masking assigné: {type(masking_module).__name__}")
    
    # 4. Configuration XLNet
    print("\n⚙️ Config XLNet...")
    
    xlnet_config = tr.XLNetConfig.build(
        d_model=CONFIG['d_model'],
        n_head=CONFIG['n_head'], 
        n_layer=CONFIG['n_layer'],
        total_seq_length=CONFIG['max_sequence_length'],
        mem_len=CONFIG['mem_len'],
        dropout=CONFIG['dropout']
    )
    
    print(f"✅ XLNet: {CONFIG['d_model']}d, {CONFIG['n_head']}h, {CONFIG['n_layer']}l")
    
    # 5. Construire le modèle
    print("\n🚀 Modèle T4Rec...")
    
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
    
    # Modèle complet
    model = tr.Model(head)
    
    print(f"✅ Modèle T4Rec créé!")
    print(f"📈 Paramètres: {sum(p.numel() for p in model.parameters()):,}")
    
    # 6. Test rapide du modèle
    print("\n🧪 Test du modèle...")
    
    # Créer un batch test
    test_batch = {
        k: torch.tensor(v[:16]) for k, v in t4rec_data.items()  # 16 premiers échantillons
    }
    
    model.eval()
    with torch.no_grad():
        try:
            output = model(test_batch)
            print(f"✅ Test réussi: output shape={output.shape}")
            test_success = True
        except Exception as test_error:
            print(f"❌ Test échoué: {test_error}")
            test_success = False
    
    if test_success:
        # 7. Entraînement
        print("\n🎯 Entraînement...")
        
        # Préparer les données
        X_torch = {k: torch.tensor(v) for k, v in t4rec_data.items()}
        
        # Target
        y = df['souscription_produit_1m'].values
        label_encoder = LabelEncoder() 
        y_encoded = label_encoder.fit_transform(y)
        y_torch = torch.tensor(y_encoded, dtype=torch.long)
        
        # Split
        indices = np.arange(len(y_torch))
        train_indices, val_indices = train_test_split(
            indices, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        X_train = {k: v[train_indices] for k, v in X_torch.items()}
        X_val = {k: v[val_indices] for k, v in X_torch.items()}
        y_train = y_torch[train_indices] 
        y_val = y_torch[val_indices]
        
        print(f"✅ Train: {len(train_indices)}, Val: {len(val_indices)}")
        
        # Optimiseur
        from torch.optim import AdamW
        from torch.nn import CrossEntropyLoss
        
        optimizer = AdamW(model.parameters(), lr=1e-4)
        criterion = CrossEntropyLoss()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        
        # Entraînement
        num_epochs = 8
        batch_size = 32
        
        print(f"Device: {device}, Époques: {num_epochs}")
        
        for epoch in range(num_epochs):
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
            
            avg_loss = total_loss / batches if batches > 0 else 0
            print(f"Époque {epoch+1}: Loss={avg_loss:.4f}")
        
        # Évaluation
        print("\n📊 Évaluation...")
        
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
        
        # Sauvegarde
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'schema': schema,
            'config': CONFIG,
            'label_encoder': label_encoder,
            'accuracy': accuracy,
            't4rec_data_keys': list(t4rec_data.keys())
        }, 't4rec_xlnet_success.pth')
        
        print("✅ Modèle sauvé: t4rec_xlnet_success.pth")
        print("🎉 SUCCÈS T4REC XLNET!")
    
    else:
        print("❌ Test du modèle échoué - pas d'entraînement")

except Exception as e:
    print(f"❌ ERREUR: {e}")
    import traceback
    traceback.print_exc()
    
    # Debug détaillé
    print(f"\n🔍 DEBUG:")
    if 'input_module' in locals():
        print(f"Input module: {type(input_module)}")
        print(f"Input module hasattr masking: {hasattr(input_module, 'masking')}")
        if hasattr(input_module, 'masking'):
            print(f"Masking type: {type(input_module.masking)}")
    
    if 'xlnet_config' in locals():
        print(f"XLNet config: {type(xlnet_config)}")
    
    print(f"Schéma colonnes: {[col.name for col in schema.columns] if 'schema' in locals() else 'Non créé'}")
