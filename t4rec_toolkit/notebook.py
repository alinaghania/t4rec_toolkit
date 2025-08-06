# === ENTRAÎNEMENT DU MODÈLE T4REC - VERSION XLNET ===

print("🚀 ENTRAÎNEMENT DU MODÈLE XLNET")
print("=" * 50)

# Imports
from t4rec_toolkit.models import ModelRegistry, XLNetModelBuilder, create_model
from t4rec_toolkit.models.registry import get_available_models
import torch
import transformers4rec.torch as tr

try:
    # 1. Vérifier les modèles disponibles
    available_models = get_available_models()
    print(f"📋 Modèles disponibles: {available_models}")

    # 2. Créer un schéma T4Rec natif
    print("🏗️ Création du schéma T4Rec...")
    
    # Utiliser l'API merlin.schema
    from merlin.schema import Schema, ColumnSchema, Tags
    
    # Créer les colonnes du schéma
    columns = []
    
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
            properties={"vocab_size": 20}  # Augmenté pour XLNet
        )
        columns.append(column)
        print(f"   ✅ Catégorielle: {feature_name}")

    # Créer le schéma avec toutes les colonnes
    schema = Schema(columns)
    print(f"✅ Schéma créé avec {len(sequence_features) + len(categorical_features)} features")
    
    # 3. Convertir le schéma pour le registry - VERSION XLNET
    schema_dict = {
        "feature_specs": [],
        "sequence_length": 20  # XLNet gère mieux les séquences plus longues
    }
    
    for column in columns:
        spec = {
            "name": column.name,
            "dtype": str(column.dtype),
            "is_sequence": column.is_list,
            "is_continuous": Tags.CONTINUOUS in column.tags,
            "is_categorical": Tags.CATEGORICAL in column.tags,
        }
        
        if column.properties:
            spec.update(column.properties)
            
        schema_dict["feature_specs"].append(spec)
    
    print(f"🔧 Schéma converti pour XLNet: {len(schema_dict['feature_specs'])} specs")
    
    # 4. Test de création du module d'entrée XLNET
    print("\n🧪 TEST DU MODULE D'ENTRÉE XLNET")
    print("-" * 35)
    
    builder = XLNetModelBuilder()
    
    # Configuration optimisée pour XLNet
    xlnet_config = {
        'd_model': 256,         # XLNet supporte mieux des dimensions plus grandes
        'n_head': 8,            # Plus de têtes d'attention
        'n_layer': 4,           # Plus de couches pour XLNet
        'max_sequence_length': 20,  # Séquences plus longues
        'mem_len': 50,          # Mémoire pour XLNet
        'dropout': 0.1,
        'masking': 'mlm',       # MLM pour XLNet
        'attn_type': 'bi'       # Attention bidirectionnelle
    }
    
    # Test de création du module d'entrée
    try:
        test_input_module = builder.build_input_module(
            schema_dict, 
            d_model=xlnet_config['d_model'], 
            max_sequence_length=xlnet_config['max_sequence_length'], 
            masking=xlnet_config['masking']
        )
        
        if test_input_module is not None:
            print("✅ Module d'entrée XLNet créé avec succès")
            print(f"   Type: {type(test_input_module).__name__}")
            print(f"   Masking: {getattr(test_input_module, 'masking', 'NON DÉFINI')}")
        else:
            print("❌ Module d'entrée XLNet retourne None")
            
    except Exception as e:
        print(f"❌ Erreur création module d'entrée XLNet: {e}")
    
    # 5. Test spécifique XLNet avec TabularSequenceFeatures
    print("\n🔍 TEST TABULARSEQUENCEFEATURES AVEC XLNET")
    print("-" * 45)
    
    try:
        # Test direct avec un schéma minimal
        test_schema_simple = Schema([
            ColumnSchema(
                name="item_id", 
                tags={Tags.CATEGORICAL, Tags.ITEM, Tags.ID},
                dtype="int32",
                properties={"vocab_size": 1000}
            )
        ])
        
        direct_module = tr.TabularSequenceFeatures.from_schema(
            schema=test_schema_simple,
            max_sequence_length=20,
            continuous_projection=256,
            aggregation="concat",
            masking="mlm"
        )
        
        if direct_module is not None:
            print("✅ TabularSequenceFeatures direct fonctionne avec schéma minimal")
        else:
            print("❌ TabularSequenceFeatures direct échoue même avec schéma minimal")
            
    except Exception as e:
        print(f"❌ TabularSequenceFeatures direct échoue: {e}")
    
    # 6. Créer le modèle XLNet si les tests sont OK
    if test_input_module is not None:
        print("\n🏗️ CRÉATION DU MODÈLE XLNET COMPLET")
        print("-" * 35)
        
        model = create_model(
            architecture="xlnet",  # Utiliser XLNet au lieu de GPT2
            schema=schema_dict,
            **xlnet_config
        )
        
        print("✅ Modèle XLNet créé via registry")
        print(f"📈 Paramètres du modèle: {sum(p.numel() for p in model.parameters()):,}")
        
        # Configuration spécifique XLNet
        print(f"📊 Configuration XLNet:")
        print(f"   - d_model: {xlnet_config['d_model']}")
        print(f"   - n_head: {xlnet_config['n_head']}")
        print(f"   - n_layer: {xlnet_config['n_layer']}")
        print(f"   - mem_len: {xlnet_config['mem_len']}")
        print(f"   - masking: {xlnet_config['masking']}")
        
        # Continuer avec l'entraînement...
        print("\n📊 PRÉPARATION DES DONNÉES POUR XLNET")
        print("-" * 40)
        
        from sklearn.model_selection import train_test_split
        import numpy as np

        def prepare_torch_data_xlnet(tabular_data):
            """Convertit les données tabulaires en format torch pour XLNet."""
            torch_data = {}
            for feature_name, feature_data in tabular_data.items():
                if isinstance(feature_data, np.ndarray):
                    if 'sequence' in feature_name:
                        # XLNet gère mieux les séquences float32
                        torch_data[feature_name] = torch.tensor(feature_data, dtype=torch.float32)
                    else:
                        # Features catégorielles en int32
                        torch_data[feature_name] = torch.tensor(feature_data, dtype=torch.int32)
                else:
                    torch_data[feature_name] = torch.tensor(feature_data, dtype=torch.int32)
            return torch_data

        # Préparer les features et target
        X_torch = prepare_torch_data_xlnet(tabular_data)
        y = df['souscription_produit_1m'].values

        # Encoder le target
        from sklearn.preprocessing import LabelEncoder
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        y_torch = torch.tensor(y_encoded, dtype=torch.long)

        print(f"   Features: {len(X_torch)}")
        print(f"   Target classes: {len(label_encoder.classes_)}")

        # Split train/validation (80/20)
        indices = np.arange(len(y_torch))
        train_indices, val_indices = train_test_split(
            indices, test_size=0.2, random_state=42, stratify=y_encoded
        )

        X_train = {k: v[train_indices] for k, v in X_torch.items()}
        X_val = {k: v[val_indices] for k, v in X_torch.items()}
        y_train = y_torch[train_indices]
        y_val = y_torch[val_indices]

        print(f"   Train: {len(train_indices)} échantillons")
        print(f"   Validation: {len(val_indices)} échantillons")

        # 7. Configuration de l'entraînement optimisée pour XLNet
        print("\n⚙️ CONFIGURATION DE L'ENTRAÎNEMENT XLNET")
        print("-" * 40)
        
        from torch.optim import AdamW
        from torch.nn import CrossEntropyLoss

        # XLNet bénéficie d'un learning rate légèrement plus faible
        optimizer = AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
        criterion = CrossEntropyLoss()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)

        print(f"Device utilisé: {device}")
        print(f"Learning rate: 5e-5 (optimisé pour XLNet)")

        # 8. Entraînement avec XLNet
        print("\n🎯 ENTRAÎNEMENT XLNET")
        print("-" * 25)
        
        num_epochs = 15  # XLNet peut bénéficier de plus d'époques
        batch_size = 24  # Batch size réduit pour XLNet plus grand

        train_losses = []
        val_losses = []

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
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)

                loss.backward()
                
                # Gradient clipping pour XLNet
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()

                total_train_loss += loss.item()
                num_batches += 1

            # Mode évaluation
            model.eval()
            total_val_loss = 0
            num_val_batches = 0

            with torch.no_grad():
                for i in range(0, len(val_indices), batch_size):
                    batch_indices = val_indices[i:i+batch_size]
                    
                    batch_X = {k: v[batch_indices].to(device) for k, v in X_val.items()}
                    batch_y = y_val[i:i+batch_size].to(device)

                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)

                    total_val_loss += loss.item()
                    num_val_batches += 1

            avg_train_loss = total_train_loss / num_batches
            avg_val_loss = total_val_loss / num_val_batches

            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)

            # Affichage périodique pour XLNet
            if (epoch + 1) % 3 == 0 or epoch == 0:
                print(f"Époque {epoch+1}/{num_epochs}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        print("✅ Entraînement XLNet terminé!")

        # 9. Évaluation finale
        print("\n📈 ÉVALUATION FINALE XLNET")
        print("-" * 30)
        
        model.eval()
        with torch.no_grad():
            val_outputs = model({k: v.to(device) for k, v in X_val.items()})
            val_predictions = torch.argmax(val_outputs, dim=1)
            
            correct = (val_predictions == y_val.to(device)).sum().item()
            accuracy = correct / len(y_val)
            
            print(f"Accuracy finale XLNet: {accuracy:.2%}")
            
            # Métriques additionnelles pour XLNet
            from sklearn.metrics import classification_report, confusion_matrix
            
            y_true = y_val.cpu().numpy()
            y_pred = val_predictions.cpu().numpy()
            
            print("\nRapport de classification:")
            print(classification_report(y_true, y_pred, target_names=label_encoder.classes_))

        # 10. Sauvegarder le modèle XLNet
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'schema': schema_dict,
            'merlin_schema': schema,
            'label_encoder': label_encoder,
            'xlnet_config': xlnet_config,
            'epoch': num_epochs,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'final_accuracy': accuracy,
            'model_type': 'xlnet'
        }, 't4rec_xlnet_model.pth')

        print("💾 Modèle XLNet sauvegardé: t4rec_xlnet_model.pth")
        
        # 11. Visualisation des résultats (optionnel)
        print("\n📊 RÉSUMÉ DE L'ENTRAÎNEMENT")
        print("-" * 35)
        print(f"Architecture: XLNet")
        print(f"Features: {len(schema_dict['feature_specs'])}")
        print(f"Séquences: {len(sequence_features)}")
        print(f"Catégorielles: {len(categorical_features)}")
        print(f"Époques: {num_epochs}")
        print(f"Batch size: {batch_size}")
        print(f"Learning rate: 5e-5")
        print(f"Accuracy finale: {accuracy:.2%}")
        print(f"Loss finale train: {train_losses[-1]:.4f}")
        print(f"Loss finale val: {val_losses[-1]:.4f}")
        
    else:
        print("\n❌ ABANDON - MODULE D'ENTRÉE XLNET ÉCHOUE")
        print("Le module d'entrée XLNet ne peut pas être créé.")
        
        # Test de fallback avec un modèle très simple
        print("\n🔄 TEST FALLBACK AVEC MODÈLE PYTORCH NATIF")
        print("-" * 45)
        
        # Créer un modèle PyTorch simple en fallback
        class SimpleBankingModel(torch.nn.Module):
            def __init__(self, n_features, n_classes):
                super().__init__()
                self.n_features = n_features
                self.n_classes = n_classes
                
                # Couches simples
                self.fc1 = torch.nn.Linear(n_features, 128)
                self.fc2 = torch.nn.Linear(128, 64)
                self.fc3 = torch.nn.Linear(64, n_classes)
                self.dropout = torch.nn.Dropout(0.2)
                self.relu = torch.nn.ReLU()
                
            def forward(self, x):
                # Aplatir toutes les features en un vecteur
                if isinstance(x, dict):
                    # Concaténer toutes les features
                    features = []
                    for key, value in x.items():
                        if value.dim() > 2:
                            value = value.mean(dim=1)  # Moyenner les séquences
                        if value.dim() == 1:
                            value = value.unsqueeze(1)
                        features.append(value.float())
                    x = torch.cat(features, dim=1)
                
                x = self.relu(self.fc1(x))
                x = self.dropout(x)
                x = self.relu(self.fc2(x))
                x = self.dropout(x)
                x = self.fc3(x)
                return x
        
        print("✅ Modèle PyTorch simple créé en fallback")
        print("Ce modèle peut être utilisé si T4Rec ne fonctionne pas.")

except Exception as e:
    print(f"\n❌ ERREUR LORS DE L'ENTRAÎNEMENT XLNET: {e}")
    import traceback
    traceback.print_exc()
    
    print("\n🔍 SUGGESTIONS XLNET:")
    print("1. XLNet nécessite plus de mémoire que GPT2")
    print("2. Réduire batch_size si erreur de mémoire")
    print("3. Réduire d_model si nécessaire")
    print("4. XLNet fonctionne mieux avec MLM masking")
    print("5. Essayer avec CPU si problème GPU: device='cpu'")
