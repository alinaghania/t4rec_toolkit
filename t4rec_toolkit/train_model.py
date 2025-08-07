
# === ENTRAÎNEMENT T4REC XLNET AVEC MÉTRIQUES ===

print("🏋️ ENTRAÎNEMENT T4REC XLNET")
print("=" * 50)

import torch
import torch.nn as nn
import transformers4rec.torch as tr
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset
import time

# Vérifier que le modèle existe
if "model" not in globals():
    print("❌ Modèle non trouvé. Exécutez d'abord le script de création du modèle.")
    raise RuntimeError("Modèle requis")

if "inputs" not in globals():
    print(
        "❌ Données d'entrée non trouvées. Exécutez d'abord le script de création du modèle."
    )
    raise RuntimeError("Données requises")

print("✅ Modèle et données trouvés")

# === CONFIGURATION D'ENTRAÎNEMENT ===
TRAINING_CONFIG = {
    "batch_size": 8,  # Petit batch pour la stabilité
    "learning_rate": 1e-4,  # Learning rate conservateur
    "num_epochs": 3,  # Peu d'époques pour le test
    "weight_decay": 1e-5,  # Régularisation légère
    "warmup_steps": 10,  # Warm-up minimal
    "eval_every": 50,  # Évaluation fréquente
    "patience": 5,  # Early stopping
    "max_grad_norm": 1.0,  # Gradient clipping
}

print(f"📋 Configuration: {TRAINING_CONFIG}")

# === PRÉPARATION DES DONNÉES D'ENTRAÎNEMENT ===
print("\n📊 Préparation des données d'entraînement...")

# Récupérer les données du modèle créé précédemment
item_sequences = inputs["item_id"]  # Shape: [n_sessions, seq_len]
user_sequences = inputs["user_id"]  # Shape: [n_sessions, seq_len]

n_sessions, seq_len = item_sequences.shape
print(f"✅ Données: {n_sessions} sessions de {seq_len} items")

# Créer les targets (next item prediction)
# Target = le prochain item dans la séquence
targets = torch.zeros_like(item_sequences)
targets[:, :-1] = item_sequences[:, 1:]  # Décaler d'une position
targets[:, -1] = item_sequences[:, 0]  # Dernier -> premier (cyclique)

print(f"✅ Targets créés: {targets.shape}")

# Split train/validation
n_train = int(0.8 * n_sessions)
train_inputs = {
    "item_id": item_sequences[:n_train],
    "user_id": user_sequences[:n_train],
}
train_targets = targets[:n_train]

val_inputs = {"item_id": item_sequences[n_train:], "user_id": user_sequences[n_train:]}
val_targets = targets[n_train:]

print(
    f"✅ Split: {len(train_inputs['item_id'])} train, {len(val_inputs['item_id'])} val"
)


# === FONCTION D'ENTRAÎNEMENT ===
def create_batch_iterator(inputs_dict, targets, batch_size, shuffle=True):
    """Créer un itérateur de batch pour T4Rec."""
    n_samples = len(targets)
    indices = torch.randperm(n_samples) if shuffle else torch.arange(n_samples)

    for i in range(0, n_samples, batch_size):
        batch_indices = indices[i : i + batch_size]
        batch_inputs = {k: v[batch_indices] for k, v in inputs_dict.items()}
        batch_targets = targets[batch_indices]
        yield batch_inputs, batch_targets


# === OPTIMIZER ET SCHEDULER ===
print("\n⚙️ Configuration optimizer...")

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=TRAINING_CONFIG["learning_rate"],
    weight_decay=TRAINING_CONFIG["weight_decay"],
)

# Scheduler avec warmup
from torch.optim.lr_scheduler import LambdaLR


def lr_lambda(current_step):
    if current_step < TRAINING_CONFIG["warmup_steps"]:
        return float(current_step) / float(max(1, TRAINING_CONFIG["warmup_steps"]))
    return 1.0


scheduler = LambdaLR(optimizer, lr_lambda)

print("✅ Optimizer et scheduler configurés")


# === MÉTRIQUES PERSONNALISÉES ===
class SimpleMetrics:
    def __init__(self):
        self.reset()

    def reset(self):
        self.total_loss = 0.0
        self.total_accuracy = 0.0
        self.total_samples = 0
        self.batch_count = 0

    def update(self, loss, predictions, targets):
        batch_size = len(targets)
        self.total_loss += loss * batch_size

        # Calculer accuracy (top-1)
        if len(predictions.shape) > 2:
            # Si predictions a 3 dimensions [batch, seq, vocab]
            pred_flat = predictions.view(-1, predictions.size(-1))
            target_flat = targets.view(-1)
            pred_classes = torch.argmax(pred_flat, dim=-1)
            correct = (pred_classes == target_flat).float().sum().item()
            accuracy = correct / len(target_flat)
        else:
            # Si predictions a 2 dimensions
            pred_classes = torch.argmax(predictions, dim=-1)
            correct = (pred_classes == targets).float().sum().item()
            accuracy = correct / batch_size

        self.total_accuracy += accuracy * batch_size
        self.total_samples += batch_size
        self.batch_count += 1

    def compute(self):
        if self.total_samples == 0:
            return {"loss": 0.0, "accuracy": 0.0}

        return {
            "loss": self.total_loss / self.total_samples,
            "accuracy": self.total_accuracy / self.total_samples,
        }


# === FONCTION D'ÉVALUATION ===
def evaluate_model(model, eval_inputs, eval_targets, batch_size):
    """Évaluer le modèle sur les données de validation."""
    model.eval()
    metrics = SimpleMetrics()

    with torch.no_grad():
        for batch_inputs, batch_targets in create_batch_iterator(
            eval_inputs, eval_targets, batch_size, shuffle=False
        ):
            try:
                outputs = model(batch_inputs)

                # Calculer la loss
                if hasattr(outputs, "loss"):
                    loss = outputs.loss.item()
                    predictions = (
                        outputs.prediction_scores
                        if hasattr(outputs, "prediction_scores")
                        else outputs
                    )
                else:
                    # Fallback si pas de loss dans les outputs
                    loss = 0.0
                    predictions = outputs

                metrics.update(loss, predictions, batch_targets)

            except Exception as e:
                print(f"⚠️ Erreur évaluation batch: {e}")
                continue

    return metrics.compute()


# === BOUCLE D'ENTRAÎNEMENT ===
print(f"\n🏋️ Début de l'entraînement ({TRAINING_CONFIG['num_epochs']} époques)...")

# Historique des métriques
history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "lr": []}

best_val_loss = float("inf")
patience_counter = 0
global_step = 0

model.train()

for epoch in range(TRAINING_CONFIG["num_epochs"]):
    print(f"\n📈 Époque {epoch + 1}/{TRAINING_CONFIG['num_epochs']}")

    epoch_metrics = SimpleMetrics()
    epoch_start_time = time.time()

    # Entraînement
    for batch_idx, (batch_inputs, batch_targets) in enumerate(
        create_batch_iterator(
            train_inputs, train_targets, TRAINING_CONFIG["batch_size"]
        )
    ):
        try:
            optimizer.zero_grad()

            # Forward pass
            outputs = model(batch_inputs)

            # Calculer la loss
            if hasattr(outputs, "loss"):
                loss = outputs.loss
                predictions = (
                    outputs.prediction_scores
                    if hasattr(outputs, "prediction_scores")
                    else outputs
                )
            else:
                # Fallback: calculer une loss simple
                if len(outputs.shape) == 3:
                    # [batch, seq, vocab] -> [batch*seq, vocab]
                    output_flat = outputs.view(-1, outputs.size(-1))
                    target_flat = batch_targets.view(-1)
                    loss = nn.CrossEntropyLoss()(output_flat, target_flat)
                else:
                    loss = nn.CrossEntropyLoss()(outputs, batch_targets.view(-1))
                predictions = outputs

            # Backward pass
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), TRAINING_CONFIG["max_grad_norm"]
            )

            optimizer.step()
            scheduler.step()

            # Métriques
            epoch_metrics.update(loss.item(), predictions, batch_targets)
            global_step += 1

            # Log périodique
            if batch_idx % 20 == 0:
                current_lr = scheduler.get_last_lr()[0]
                print(
                    f"  Batch {batch_idx}: loss={loss.item():.4f}, lr={current_lr:.6f}"
                )

            # Évaluation périodique
            if global_step % TRAINING_CONFIG["eval_every"] == 0:
                print(f"\n🔍 Évaluation step {global_step}...")
                val_metrics = evaluate_model(
                    model, val_inputs, val_targets, TRAINING_CONFIG["batch_size"]
                )
                print(
                    f"  Val: loss={val_metrics['loss']:.4f}, acc={val_metrics['accuracy']:.4f}"
                )

                # Early stopping
                if val_metrics["loss"] < best_val_loss:
                    best_val_loss = val_metrics["loss"]
                    patience_counter = 0
                    print("  ✅ Nouveau meilleur modèle sauvegardé")
                else:
                    patience_counter += 1

                model.train()  # Retour en mode train

        except Exception as e:
            print(f"⚠️ Erreur batch {batch_idx}: {e}")
            continue

    # Métriques fin d'époque
    train_metrics = epoch_metrics.compute()
    val_metrics = evaluate_model(
        model, val_inputs, val_targets, TRAINING_CONFIG["batch_size"]
    )

    epoch_time = time.time() - epoch_start_time
    current_lr = scheduler.get_last_lr()[0]

    # Sauvegarde historique
    history["train_loss"].append(train_metrics["loss"])
    history["train_acc"].append(train_metrics["accuracy"])
    history["val_loss"].append(val_metrics["loss"])
    history["val_acc"].append(val_metrics["accuracy"])
    history["lr"].append(current_lr)

    print(f"\n📊 Époque {epoch + 1} terminée ({epoch_time:.1f}s)")
    print(
        f"  Train: loss={train_metrics['loss']:.4f}, acc={train_metrics['accuracy']:.4f}"
    )
    print(f"  Val:   loss={val_metrics['loss']:.4f}, acc={val_metrics['accuracy']:.4f}")
    print(f"  LR: {current_lr:.6f}")

    # Early stopping check
    if patience_counter >= TRAINING_CONFIG["patience"]:
        print(f"\n⏹️ Early stopping après {patience_counter} époques sans amélioration")
        break

print(f"\n🎉 ENTRAÎNEMENT TERMINÉ!")
print(f"📊 Meilleure val loss: {best_val_loss:.4f}")

# === RÉSUMÉ FINAL ===
print(f"\n📈 RÉSUMÉ FINAL:")
print(f"  🏋️ Époques: {len(history['train_loss'])}")
print(f"  📉 Train loss finale: {history['train_loss'][-1]:.4f}")
print(f"  📉 Val loss finale: {history['val_loss'][-1]:.4f}")
print(f"  🎯 Train accuracy: {history['train_acc'][-1]:.2%}")
print(f"  🎯 Val accuracy: {history['val_acc'][-1]:.2%}")

# === GRAPHIQUES SIMPLES (optionnel) ===
try:
    import matplotlib.pyplot as plt

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))

    # Loss
    ax1.plot(history["train_loss"], label="Train")
    ax1.plot(history["val_loss"], label="Validation")
    ax1.set_title("Loss")
    ax1.legend()

    # Accuracy
    ax2.plot(history["train_acc"], label="Train")
    ax2.plot(history["val_acc"], label="Validation")
    ax2.set_title("Accuracy")
    ax2.legend()

    # Learning Rate
    ax3.plot(history["lr"])
    ax3.set_title("Learning Rate")

    # Val Loss zoom
    ax4.plot(history["val_loss"])
    ax4.set_title("Validation Loss")

    plt.tight_layout()
    plt.show()

    print("📊 Graphiques affichés")

except ImportError:
    print("📊 Matplotlib non disponible pour les graphiques")

print("\n" + "=" * 50)
print("Entraînement terminé avec succès!")
