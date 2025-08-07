import transformers4rec.torch as tr
from transformers4rec.torch.model.head import Head
from transformers4rec.config.model import XLNetConfig
from transformers4rec.torch.masking import MaskSequence
from transformers4rec.torch.utils import schema_utils
from transformers4rec.torch.models.sequential import NextItemPredictionTask
from merlin.schema import Schema, Tags, ColumnSchema
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from merlin.io import Dataset
from transformers4rec.torch.utils.examples_utils import generate_schema

from transformers4rec.torch.metrics.ranking import NDCGAt, RecallAt

print("🚀 T4REC XLNET - FEATURES CATÉGORIELLES UNIQUEMENT - STABLE")
print("="*65)

# -----------------------
# 1. Création du schéma
# -----------------------
print("📋 Création du schéma catégoriel...")

schema = Schema([
    ColumnSchema("item_id", tags=[Tags.ITEM_ID, Tags.CATEGORICAL]),
    ColumnSchema("user_category", tags=[Tags.USER, Tags.CATEGORICAL])
])
print(f"✅ Schéma défini avec {len(schema.column_names)} colonnes: {schema.column_names}")

# -----------------------
# 2. Données d'exemple
# -----------------------
print("📊 Génération de données d'exemple...")

num_samples = 560
max_session_len = 20
item_ids = np.random.randint(1, 100, size=(num_samples,))
user_cats = np.random.choice([f"cat_{i}" for i in range(1, 15)], size=(num_samples,))
session_id = np.repeat(np.arange(num_samples // max_session_len), max_session_len)

df = pd.DataFrame({
    "item_id": item_ids,
    "user_category": user_cats,
    "session_id": session_id
})

dataset = Dataset(df)
print(f"✅ Données prêtes: item_id ({df['item_id'].nunique()} uniques), user_category ({df['user_category'].nunique()} uniques)")

# -----------------------
# 3. Module d'entrée
# -----------------------
print("🏗️ Construction du module d'entrée...")

input_module = tr.TabularSequenceFeatures.from_schema(
    schema,
    max_sequence_length=max_session_len,
    continuous_projection=None,
    aggregation="concat",
)

print("✅ Module d'entrée: TabularSequenceFeatures")

# -----------------------
# 4. Masking
# -----------------------
print("🎭 Configuration du masking...")

masking = MaskSequence(max_sequence_length=max_session_len)
input_module.masking = masking

print("✅ Masking activé: MaskSequence")

# -----------------------
# 5. XLNet config
# -----------------------
print("⚙️ Configuration du modèle XLNet...")

xlnet_config = XLNetConfig.build(
    d_model=128,
    n_head=4,
    n_layer=2,
    total_seq_length=max_session_len,
)

print("✅ XLNet configuré: 128d, 4 heads, 2 layers")

# -----------------------
# 6. Body du modèle
# -----------------------
print("🧱 Construction du corps du modèle...")

transformer_block = tr.Block(
    tr.TransformerBlock(xlnet_config, masking=input_module.masking),
    output_size=128  # Important!
)

body = tr.SequentialBlock(
    input_module,
    transformer_block
)

print("✅ Corps du modèle prêt: SequentialBlock")

# -----------------------
# 7. Head
# -----------------------
print("🧠 Ajout de la tête NextItemPredictionTask...")

try:
    head = Head(
        body,
        NextItemPredictionTask(
            weight_tying=True,
            metrics=[
                NDCGAt(top_ks=[5, 10], labels_onehot=True),
                RecallAt(top_ks=[5, 10], labels_onehot=True)
            ],
            loss_function="cross_entropy"
        ),
        inputs=input_module
    )
    print("✅ Tête ajoutée avec succès !")
except Exception as e:
    print("❌ ERREUR GÉNÉRALE :", str(e))
    print("📑 Schéma avec", len(schema.column_names), "colonnes :", schema.column_names)
    raise
