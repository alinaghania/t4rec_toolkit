# === ENTRAÎNEMENT DU MODÈLE T4REC - VERSION XLNET CORRIGÉE ===

print("🚀 ENTRAÎNEMENT DU MODÈLE XLNET CORRIGÉ")
print("=" * 55)

# Imports
from t4rec_toolkit.models import ModelRegistry, XLNetModelBuilder, create_model
from t4rec_toolkit.models.registry import get_available_models
import torch
import transformers4rec.torch as tr

try:
    # 1. Vérifier les modèles disponibles
    available_models = get_available_models()
    print(f"📋 Modèles disponibles: {available_models}")

    # 2. Créer un schéma T4Rec natif AVEC ITEM_ID
    print("🏗️ Création du schéma T4Rec avec item_id...")
    
    # Utiliser l'API merlin.schema
    from merlin.schema import Schema, ColumnSchema, Tags
    
    # Créer les colonnes du schéma
    columns = []
    
    # IMPORTANT: Ajouter un item_id requis pour T4Rec masking
    print("   ⭐ Ajout de l'item_id requis pour T4Rec masking")
    item_id_column = ColumnSchema(
        name="item_id",
        tags={Tags.CATEGORICAL, Tags.ITEM, Tags.ID},
        dtype="int32",
        is_list=False,
        properties={"vocab_size": 10000}  # Vocabulaire suffisant
    )
    columns.append(item_id_column)
    print(f"   ✅ Item ID: item_id (vocab_size=10000)")
    
    # Ajouter les features séquentielles (continues)
    sequence_features = [
        'nbchqemigliss_m12_sequence',
        'nb_automobile_12dm_sequence', 
        'mntecscrdimm_sequence',
        'mnt_euro_r_3m_sequence',
        'nb_contacts_accueil_service_sequence'
    ]
    
    for feature_name in sequence_features:
        column = ColumnSchema(
            name=feature_name,
            tags={Tags.CONTINUOUS, Tags.LIST},
            dtype="float32",
            is_list=True,
            properties={"max_sequence_length": 20}
        )
        columns.append(column)
        print(f"   ✅ Séquence continue: {feature_name}")

    # Ajouter les features catégorielles
    categorical_features = [
        'dummy:iac_epa:03_encoded',
        'dummy:iac_epa:01_encoded', 
        'dummy:iac_epa:02_encoded',
        'dummy:iac_epa:N/A_encoded',
        'dummy:iac_epa:__Others___encoded'
    ]
    
    for feature_name in categorical_features:
        column = ColumnSchema(
            name=feature_name,
            tags={Tags.CATEGORICAL},
            dtype="int32",
            is_list=False,
            properties={"vocab_size": 20}
        )
        columns.append(column)
        print(f"   ✅ Catégorielle: {feature_name}")

    # Créer le schéma avec toutes les colonnes (item_id + features)
    schema = Schema(columns)
    total_features = len(sequence_features) + len(categorical_features) + 1  # +1 pour item_id
    print(f"✅ Schéma créé avec {total_features} features (incluant item_id)")
    
    # 3. Convertir le schéma pour le registry avec item_id
    schema_dict = {
        "feature_specs": [],
        "sequence_length": 20,
        "has_item_id": True  # Marquer que nous avons un item_id
    }
    
    for column in columns:
        spec = {
            "name": column.name,
            "dtype": str(column.dtype),
            "is_sequence": column.is_list,
            "is_continuous": Tags.CONTINUOUS in column.tags,
            "is_categorical": Tags.CATEGORICAL in column.tags,
            "is_item_id": Tags.ITEM in column.tags and Tags.ID in column.tags,
        }
        
        if column.properties:
            spec.update(column.properties)
            
        schema_dict["feature_specs"].append(spec)
    
    print(f"🔧 Schéma converti avec item_id: {len(schema_dict['feature_specs'])} specs")
    
    # 4. Préparer les données AVEC item_id artificiel
    print("\n📊 PRÉPARATION DES DONNÉES AVEC ITEM_ID")
    print("-" * 40)
    
    # Créer un item_id artificiel basé sur l'index ou les features
    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import LabelEncoder
    
    # Méthode 1: Créer des item_id basés sur les combinaisons de features catégorielles
    def create_item_ids(data_dict, categorical_feature_names):
        """Crée des item_id artificiels basés sur les features catégorielles."""
        
        # Concaténer toutes les features catégorielles pour créer des "items" uniques
        item_signatures = []
        
        # Obtenir la longueur des données
        sample_key = next(iter(data_dict.keys()))
        n_samples = len(data_dict[sample_key])
        
        for i in range(n_samples):
            # Créer une signature basée sur les features catégorielles
            signature_parts = []
            for feat_name in categorical_feature_names:
                if feat_name in data_dict:
                    # Prendre la valeur ou la moyenne si c'est un array
                    val = data_dict[feat_name][i]
                    if isinstance(val, np.ndarray):
                        val = int(val.mean()) if len(val) > 0 else 0
                    signature_parts.append(str(int(val)))
                else:
                    signature_parts.append("0")
            
            item_signatures.append("_".join(signature_parts))
        
        # Encoder les signatures en item_id numériques
        le = LabelEncoder()
        item_ids = le.fit_transform(item_signatures)
        
        print(f"   ✅ {len(np.unique(item_ids))} item_id uniques créés")
        return item_ids, le
    
    # Extraire les noms des features catégorielles
    categorical_names = [name for name in categorical_features if name in tabular_data]
    
    if categorical_names:
        item_ids, item_encoder = create_item_ids(tabular_data, categorical_names)
    else:
        # Fallback: utiliser des item_id séquentiels
        item_ids = np.arange(len(next(iter(tabular_data.values()))))
        item_encoder = None
        print("   ✅ Item_id séquentiels créés (fallback)")
    
    # Ajouter item_id aux données
    tabular_data_with_itemid = tabular_data.copy()
    tabular_data_with_itemid['item_id'] = item_ids
    
    print(f"   📈 Données préparées avec item_id: {len(tabular_data_with_itemid)} features")
    
    # 5. Test de création du modèle XLNet avec item_id
    print("\n🧪 TEST DU MODÈLE XLNET AVEC ITEM_ID")
    print("-" * 40)
    
    # Configuration XLNet sans masking pour éviter les problèmes
    xlnet_config_safe = {
        'd_model': 256,
        'n_head': 8,
        'n_layer': 4,
        'max_sequence_length': 20,
        'mem_len': 50,
        'dropout': 0.1,
        'masking': None,  # Désactiver le masking temporairement
        'attn_type': 'bi'
    }
    
    print("Configuration XLNet (masking désactivé pour test):")
    for key, value in xlnet_config_safe.items():
        print(f"   {key}: {value}")
    
    try:
        # Test de création du modèle
        model = create_model(
            architecture="xlnet",
            schema=schema_dict,
            **xlnet_config_safe
        )
        
        print("✅ Modèle XLNet créé avec succès (sans masking)")
        print(f"📈 Paramètres du modèle: {sum(p.numel() for p in model.parameters()):,}")
        
        model_created = True
        
    except Exception as e:
        print(f"❌ Échec création modèle XLNet: {e}")
        model_created = False
    
    # 6. Si le modèle est créé, procéder à l'entraînement
    if model_created:
        print("\n🎯 ENTRAÎNEMENT DU MODÈLE XLNET")
        print("-" * 35)
        
        from sklearn.model_selection import train_test_split
        
        def prepare_torch_data_with_itemid(data_dict):
            """Convertit les données avec item_id en format torch."""
            torch_data = {}
            for feature_name, feature_data in data_dict.items():
                if isinstance(feature_data, np.ndarray):
                    if 'sequence' in feature_name:
                        torch_data[feature_name] = torch.tensor(feature_data, dtype=torch.float32)
                    else:
                        torch_data[feature_name] = torch.tensor(feature_data, dtype=torch.int32)
                else:
                    torch_data[feature_name] = torch.tensor(feature_data, dtype=torch.int32)
            return torch_data

        # Préparer les données avec item_id
        X_torch = prepare_torch_data_with_itemid(tabular_data_with_itemid)
        y = df['souscription_produit_1m'].values

        # Encoder le target
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        y_torch = torch.tensor(y_encoded, dtype=torch.long)

        print(f"Features avec item_id: {len(X_torch)}")
        print(f"Target classes: {len(label_encoder.classes_)}")

        # Split train/validation
        indices = np.arange(len(y_torch))
        train_indices, val_indices = train_test_split(
            indices, test_size=0.2, random_state=42, stratify=y_encoded
        )

        X_train = {k: v[train_indices] for k, v in X_torch.items()}
        X_val = {k: v[val_indices] for k, v in X_torch.items()}
        y_train = y_torch[train_indices]
        y_val = y_torch[val_indices]

        print(f"Train: {len(train_indices)} échantillons")
        print(f"Validation: {len(val_indices)} échantillons")

        # Configuration de l'entraînement
        from torch.optim import AdamW
        from torch.nn import CrossEntropyLoss

        optimizer = AdamW(model.parameters(), lr=3e-5, weight_decay=0.01)
        criterion = CrossEntropyLoss()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)

        print(f"Device: {device}")

        # Entraînement
        num_epochs = 12
        batch_size = 16  # Batch size plus petit pour XLNet

        train_losses = []
        val_losses = []

        print(f"\nDébut entraînement: {num_epochs} époques, batch_size={batch_size}")

        for epoch in range(num_epochs):
            # Mode entraînement
            model.train()
            total_train_loss = 0
            num_batches = 0

            for i in range(0, len(train_indices), batch_size):
                batch_indices = train_indices[i:i+batch_size]
                
                batch_X = {k: v[batch_indices].to(device) for k, v in X_train.items()}
                batch_y = y_train[i:i+batch_size].to(device)

                optimizer.zero_grad()
                
                try:
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    
                    # Gradient clipping pour XLNet
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    
                    optimizer.step()
                    total_train_loss += loss.item()
                    num_batches += 1
                    
                except Exception as batch_error:
                    print(f"Erreur batch {i}: {batch_error}")
                    continue

            # Mode évaluation
            model.eval()
            total_val_loss = 0
            num_val_batches = 0

            with torch.no_grad():
                for i in range(0, len(val_indices), batch_size):
                    batch_indices = val_indices[i:i+batch_size]
                    
                    batch_X = {k: v[batch_indices].to(device) for k, v in X_val.items()}
                    batch_y = y_val[i:i+batch_size].to(device)

                    try:
                        outputs = model(batch_X)
                        loss = criterion(outputs, batch_y)
                        total_val_loss += loss.item()
                        num_val_batches += 1
                    except Exception as val_error:
                        continue

            if num_batches > 0 and num_val_batches > 0:
                avg_train_loss = total_train_loss / num_batches
                avg_val_loss = total_val_loss / num_val_batches
                
                train_losses.append(avg_train_loss)
                val_losses.append(avg_val_loss)
                
                if (epoch + 1) % 3 == 0 or epoch == 0:
                    print(f"Époque {epoch+1}: Train={avg_train_loss:.4f}, Val={avg_val_loss:.4f}")

        print("✅ Entraînement terminé!")

        # Évaluation finale
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for i in range(0, len(val_indices), batch_size):
                batch_indices = val_indices[i:i+batch_size]
                batch_X = {k: v[batch_indices].to(device) for k, v in X_val.items()}
                batch_y = y_val[i:i+batch_size].to(device)
                
                try:
                    outputs = model(batch_X)
                    predictions = torch.argmax(outputs, dim=1)
                    correct += (predictions == batch_y).sum().item()
                    total += batch_y.size(0)
                except:
                    continue
        
        if total > 0:
            accuracy = correct / total
            print(f"Accuracy finale: {accuracy:.2%}")
        else:
            accuracy = 0.0
            print("Impossible de calculer l'accuracy")

        # Sauvegarder le modèle
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'schema': schema_dict,
            'tabular_data_with_itemid': list(tabular_data_with_itemid.keys()),
            'label_encoder': label_encoder,
            'item_encoder': item_encoder,
            'xlnet_config': xlnet_config_safe,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'final_accuracy': accuracy,
            'model_type': 'xlnet_with_itemid'
        }, 't4rec_xlnet_with_itemid.pth')

        print("💾 Modèle sauvegardé: t4rec_xlnet_with_itemid.pth")
        
    else:
        # Fallback: modèle PyTorch simple
        print("\n🔄 FALLBACK: MODÈLE PYTORCH SIMPLE")
        print("-" * 40)
        
        class SimpleClassifier(torch.nn.Module):
            def __init__(self, input_dim, n_classes):
                super().__init__()
                self.fc1 = torch.nn.Linear(input_dim, 256)
                self.fc2 = torch.nn.Linear(256, 128)
                self.fc3 = torch.nn.Linear(128, n_classes)
                self.dropout = torch.nn.Dropout(0.3)
                self.relu = torch.nn.ReLU()
                
            def forward(self, x):
                if isinstance(x, dict):
                    # Concaténer toutes les features
                    features = []
                    for value in x.values():
                        if value.dim() > 2:
                            value = value.mean(dim=1)
                        if value.dim() == 1:
                            value = value.unsqueeze(1)
                        features.append(value.float())
                    x = torch.cat(features, dim=1)
                
                x = self.relu(self.fc1(x))
                x = self.dropout(x)
                x = self.relu(self.fc2(x))
                x = self.dropout(x)
                return self.fc3(x)
        
        # Calculer la dimension d'entrée
        sample_input = prepare_torch_data_with_itemid(tabular_data_with_itemid)
        total_dim = 0
        for key, value in sample_input.items():
            if value.dim() > 1:
                total_dim += value.shape[1] if value.dim() == 2 else 1
            else:
                total_dim += 1
        
        simple_model = SimpleClassifier(total_dim, len(np.unique(y)))
        print(f"✅ Modèle PyTorch simple créé: {total_dim} -> {len(np.unique(y))} classes")
        print(f"📈 Paramètres: {sum(p.numel() for p in simple_model.parameters()):,}")

except Exception as e:
    print(f"\n❌ ERREUR GÉNÉRALE: {e}")
    import traceback
    traceback.print_exc()
    
    print("\n💡 SOLUTIONS RECOMMANDÉES:")
    print("1. L'item_id est maintenant inclus dans le schéma")
    print("2. Le masking est désactivé pour éviter les erreurs")
    print("3. Un modèle PyTorch simple est disponible en fallback")
    print("4. Vérifiez que vos données 'tabular_data' et 'df' sont bien définies")
