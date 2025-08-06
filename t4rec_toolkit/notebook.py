# === ENTRAÎNEMENT DU MODÈLE T4REC - VERSION AVEC DIAGNOSTICS ===

print("🚀 ENTRAÎNEMENT DU MODÈLE")
print("=" * 50)

# Imports
from t4rec_toolkit.models import ModelRegistry, GPT2ModelBuilder, create_model
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
            properties={"max_sequence_length": 15}
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
            properties={"vocab_size": 15}
        )
        columns.append(column)
        print(f"   ✅ Catégorielle: {feature_name}")

    # Créer le schéma avec toutes les colonnes
    schema = Schema(columns)
    print(f"✅ Schéma créé avec {len(sequence_features) + len(categorical_features)} features")
    
    # 3. Convertir le schéma pour le registry
    schema_dict = {
        "feature_specs": [],
        "sequence_length": 15
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
    
    print(f"🔧 Schéma converti pour le registry: {len(schema_dict['feature_specs'])} specs")
    
    # 4. Test de création du module d'entrée AVANT de créer le modèle complet
    print("\n🧪 TEST DU MODULE D'ENTRÉE")
    print("-" * 30)
    
    builder = GPT2ModelBuilder()
    
    # Test de création du module d'entrée
    try:
        test_input_module = builder.build_input_module(
            schema_dict, 
            d_model=192, 
            max_sequence_length=15, 
            masking="clm"
        )
        
        if test_input_module is not None:
            print("✅ Module d'entrée créé avec succès")
            print(f"   Type: {type(test_input_module).__name__}")
            print(f"   Masking: {getattr(test_input_module, 'masking', 'NON DÉFINI')}")
        else:
            print("❌ Module d'entrée retourne None")
            
    except Exception as e:
        print(f"❌ Erreur création module d'entrée: {e}")
        import traceback
        traceback.print_exc()
    
    # 5. Diagnostic complet en cas de problème
    print("\n🔍 DIAGNOSTIC COMPLET")
    print("-" * 30)
    
    # Diagnostic des imports
    diagnostics = {
        "transformers4rec": False,
        "merlin_schema": False,
        "torch": False
    }
    
    try:
        import transformers4rec.torch as tr
        diagnostics["transformers4rec"] = True
        print("✅ transformers4rec.torch disponible")
        
        # Test TabularSequenceFeatures
        try:
            # Créer un schéma minimal pour test
            test_col = ColumnSchema(
                name="test_item",
                tags={Tags.CATEGORICAL, Tags.ITEM, Tags.ID},
                dtype="int32",
                properties={"vocab_size": 100}
            )
            test_schema = Schema([test_col])
            
            test_module = tr.TabularSequenceFeatures.from_schema(
                schema=test_schema,
                max_sequence_length=10,
                continuous_projection=64,
                aggregation="concat",
                masking="clm"
            )
            
            if test_module is not None:
                print("✅ TabularSequenceFeatures.from_schema fonctionne")
            else:
                print("❌ TabularSequenceFeatures.from_schema retourne None")
                
        except Exception as e:
            print(f"❌ TabularSequenceFeatures.from_schema échoue: {e}")
            
    except ImportError as e:
        print(f"❌ transformers4rec.torch non disponible: {e}")
    
    try:
        from merlin.schema import Schema, ColumnSchema, Tags
        diagnostics["merlin_schema"] = True
        print("✅ merlin.schema disponible")
    except ImportError as e:
        print(f"❌ merlin.schema non disponible: {e}")
    
    try:
        import torch
        diagnostics["torch"] = True
        print("✅ torch disponible")
    except ImportError as e:
        print(f"❌ torch non disponible: {e}")
    
    # 6. Créer le modèle seulement si les tests sont OK
    if test_input_module is not None:
        print("\n🏗️ CRÉATION DU MODÈLE COMPLET")
        print("-" * 30)
        
        model = create_model(
            architecture="gpt2",
            schema=schema_dict,
            d_model=192,
            n_head=6,
            n_layer=3,
            max_sequence_length=15,
            dropout=0.1,
            attn_dropout=0.1,
            resid_dropout=0.1,
            embd_dropout=0.1
        )
        
        print("✅ Modèle GPT2 créé via registry")
        print(f"📈 Paramètres du modèle: {sum(p.numel() for p in model.parameters()):,}")
        
        # Continuer avec l'entraînement...
        print("\n📊 PRÉPARATION DES DONNÉES")
        print("-" * 30)
        
        # 4. Préparer les données d'entraînement
        from sklearn.model_selection import train_test_split
        import numpy as np

        def prepare_torch_data(tabular_data):
            """Convertit les données tabulaires en format torch."""
            torch_data = {}
            for feature_name, feature_data in tabular_data.items():
                if isinstance(feature_data, np.ndarray):
                    if 'sequence' in feature_name:
                        torch_data[feature_name] = torch.tensor(feature_data, dtype=torch.float32)
                    else:
                        torch_data[feature_name] = torch.tensor(feature_data, dtype=torch.long)
                else:
                    torch_data[feature_name] = torch.tensor(feature_data, dtype=torch.long)
            return torch_data

        # Préparer les features et target
        X_torch = prepare_torch_data(tabular_data)
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

        # 5. Configuration de l'entraînement
        print("\n⚙️ CONFIGURATION DE L'ENTRAÎNEMENT")
        print("-" * 30)
        
        from torch.optim import AdamW
        from torch.nn import CrossEntropyLoss

        optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
        criterion = CrossEntropyLoss()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)

        print(f"Device utilisé: {device}")

        # 6. Entraînement
        print("\n🎯 ENTRAÎNEMENT")
        print("-" * 30)
        
        num_epochs = 10
        batch_size = 32

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

            print(f"Époque {epoch+1}/{num_epochs}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        print("✅ Entraînement terminé!")

        # 7. Évaluation finale
        model.eval()
        with torch.no_grad():
            val_outputs = model({k: v.to(device) for k, v in X_val.items()})
            val_predictions = torch.argmax(val_outputs, dim=1)
            
            correct = (val_predictions == y_val.to(device)).sum().item()
            accuracy = correct / len(y_val)
            
            print(f"Accuracy finale: {accuracy:.2%}")

        # 8. Sauvegarder le modèle
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'schema': schema_dict,
            'merlin_schema': schema,
            'label_encoder': label_encoder,
            'epoch': num_epochs,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'final_accuracy': accuracy
        }, 't4rec_model.pth')

        print("💾 Modèle sauvegardé: t4rec_model.pth")
        
    else:
        print("\n❌ ABANDON DE LA CRÉATION DU MODÈLE")
        print("Le module d'entrée ne peut pas être créé.")
        print("\n💡 SOLUTIONS POSSIBLES:")
        print("1. Vérifiez l'installation: pip install transformers4rec==23.04.00 merlin-core")
        print("2. Vérifiez les versions des dépendances")
        print("3. Redémarrez le kernel")

except Exception as e:
    print(f"\n❌ ERREUR LORS DE L'ENTRAÎNEMENT: {e}")
    import traceback
    traceback.print_exc()
    
    # Diagnostic final
    print("\n🔍 DIAGNOSTIC FINAL:")
    print("="*50)
    
    try:
        from t4rec_toolkit.models.gpt2_builder import diagnose_gpt2_creation_failure
        diagnosis = diagnose_gpt2_creation_failure(schema_dict if 'schema_dict' in locals() else {})
        
        print("📊 Analyse du schéma:")
        for key, value in diagnosis["schema_analysis"].items():
            print(f"   {key}: {value}")
            
        print("\n⚙️ Analyse de la configuration:")
        for key, value in diagnosis["config_analysis"].items():
            print(f"   {key}: {value}")
            
        print("\n📦 Vérification des imports:")
        for key, value in diagnosis["import_checks"].items():
            status = "✅" if value else "❌"
            print(f"   {status} {key}: {value}")
            
        print("\n💡 Recommandations:")
        for i, rec in enumerate(diagnosis["recommendations"], 1):
            print(f"   {i}. {rec}")
            
    except ImportError:
        print("Module de diagnostic non disponible")
        
        # Diagnostic manuel basique
        print("\n📦 Vérification manuelle des imports:")
        
        try:
            import transformers4rec
            print(f"✅ transformers4rec version: {transformers4rec.__version__}")
        except ImportError as e:
            print(f"❌ transformers4rec non disponible: {e}")
            
        try:
            from merlin.schema import Schema
            print("✅ merlin.schema disponible")
        except ImportError as e:
            print(f"❌ merlin.schema non disponible: {e}")
            
        try:
            import torch
            print(f"✅ torch version: {torch.__version__}")
        except ImportError as e:
            print(f"❌ torch non disponible: {e}")
            
        print("\n💡 Solutions recommandées:")
        print("1. pip install transformers4rec==23.04.00")
        print("2. pip install merlin-core")
        print("3. pip install torch")
        print("4. Redémarrer le kernel Jupyter")
        print("5. Vérifier les conflits de dépendances: pip list | grep -E '(transformers4rec|merlin|torch)'")
    
    except Exception as diag_error:
        print(f"Erreur lors du diagnostic: {diag_error}")
        
        print("\n🔧 DIAGNOSTIC MANUEL DE BASE:")
        print("-" * 40)
        
        # Vérifications de base
        import sys
        print(f"Python version: {sys.version}")
        
        # Vérifier les variables locales disponibles
        print(f"\nVariables disponibles:")
        local_vars = [var for var in locals().keys() if not var.startswith('_')]
        for var in sorted(local_vars):
            print(f"   - {var}")
            
        # Informations sur l'erreur
        if 'e' in locals():
            print(f"\nDernière erreur: {type(e).__name__}: {e}")
            
        print("\n📋 CHECKLIST DE DÉPANNAGE:")
        print("□ Vérifier que transformers4rec==23.04.00 est installé")
        print("□ Vérifier que merlin-core est installé") 
        print("□ Vérifier que torch est installé")
        print("□ Redémarrer le kernel Jupyter")
        print("□ Vérifier les données d'entrée (variables 'tabular_data' et 'df')")
        print("□ Essayer avec un schéma plus simple")
        print("□ Consulter les logs détaillés ci-dessus")
        
        print("\n🆘 SI LE PROBLÈME PERSISTE:")
        print("1. Copier l'erreur complète")
        print("2. Vérifier la compatibilité des versions")
        print("3. Tester avec un exemple minimal")
        print("4. Consulter la documentation Transformers4Rec 23.04")
