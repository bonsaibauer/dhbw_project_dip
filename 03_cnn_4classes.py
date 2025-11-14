import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms
from torchvision.utils import make_grid
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
import os
import numpy as np
from pathlib import Path

# ==========================================
# TEIL 1: MODELL-DEFINITION
# ==========================================

class SimpleCNN_4Class(nn.Module):
    """
    Ein einfaches CNN, inspiriert von der Cashew-Lösung (Versuch6_3.py),
    aber angepasst an unsere 400x400 Bilder und 4 Klassen.
    """
    def __init__(self, num_classes=4):
        super().__init__()
        
        self.network = nn.Sequential(
            # --- Block 1 ---
            # Input: 3x400x400
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # Output: 16x200x200
            
            # --- Block 2 ---
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # Output: 32x100x100
            
            # --- Block 3 (Extra, da 400x400 so groß ist) ---
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # Output: 64x50x50
            
            # --- Block 4 (Extra, da 400x400 so groß ist) ---
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # Output: 128x25x25
            
            # --- Classifier ---
            nn.Flatten(),
            nn.Linear(128 * 25 * 25, 256), # 128*25*25 = 80000
            nn.ReLU(),
            nn.Linear(256, num_classes) # 4 Klassen: Normal, Bruch, Farbfehler, Rest
        )
    
    def forward(self, xb):
        return self.network(xb)

# ==========================================
# TEIL 2: TRAINING & EVALUATION
# ==========================================

def train_model(model, epochs, lr, train_loader, validation_loader, model_save_path="best_cnn_model.pth"):
    """
    Angepasste Trainings-Schleife.
    HINWEIS: class_weights sind entfernt, da wir Oversampling verwenden.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Standard-Loss, da die Batches durch den Sampler bereits balanciert sind
    criterion = nn.CrossEntropyLoss()
    print("Verwende Standard CrossEntropyLoss (Oversampling balanciert die Batches).")
    
    print(f"\nStarte Training mit {epochs} Epochen...")
    print("Epoche | Train Loss | Train Acc | Val Loss | Val Acc")
    print("-" * 60)
    
    best_val_acc = 0.0
    
    for epoch in range(epochs):
        
        # --- Training Phase ---
        model.train()
        train_losses = []
        train_correct = 0
        train_total = 0
        
        for batch in train_loader:
            images, labels = batch
            optimizer.zero_grad()
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        train_acc = train_correct / train_total
        train_loss = sum(train_losses) / len(train_losses)
        
        # --- Validation Phase ---
        model.eval()
        val_losses = []
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in validation_loader:
                images, labels = batch
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_losses.append(loss.item())
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_acc = val_correct / val_total
        val_loss = sum(val_losses) / len(val_losses)
        
        print(f"{epoch+1:6d} | {train_loss:10.4f} | {train_acc:9.4f} | {val_loss:8.4f} | {val_acc:7.4f}")
        
        # Bestes Modell speichern (basierend auf Validation Accuracy)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), model_save_path)
            print(f"  ↳ Neues bestes Modell gespeichert! (Val Acc: {val_acc:.4f})")
            
    print(f"\nTraining abgeschlossen. Beste Validation Accuracy: {best_val_acc:.4f}")

@torch.no_grad()
def evaluate_on_testset(model, test_dataset, dataset_name="Testset"):
    """
    Evaluiert das Modell auf dem Testset und gibt pro-Klasse Genauigkeit aus.
    (Übernommen aus deinem AdvancedCNN-Entwurf)
    """
    model.eval()
    
    # Bilder und Labels sammeln
    images = []
    labels = []
    for i in range(len(test_dataset)):
        img, label = test_dataset[i]
        images.append(img)
        labels.append(label)
    
    images = torch.stack(images)
    labels = torch.tensor(labels)

    # Inferenz
    outputs = model(images)
    _, predictions = torch.max(outputs, dim=1)

    # Gesamt-Genauigkeit
    accuracy = torch.sum(predictions == labels).item() / len(test_dataset)
    
    # Detaillierte Statistik pro Klasse
    class_names = test_dataset.classes
    class_stats = {}
    
    print(f"\n=== EVALUATION AUF {dataset_name} ===")
    print(f"Gesamt-Genauigkeit: {accuracy:.4f} ({torch.sum(predictions == labels).item()}/{len(test_dataset)})")
    
    for class_idx, class_name in enumerate(class_names):
        class_mask = (labels == class_idx)
        if class_mask.sum() > 0:
            class_accuracy = (predictions[class_mask] == labels[class_mask]).float().mean().item()
            class_stats[class_name] = {
                'accuracy': class_accuracy,
                'total': class_mask.sum().item(),
                'correct': (predictions[class_mask] == labels[class_mask]).sum().item()
            }
            print(f"  {class_name}: {class_stats[class_name]['accuracy']:.4f} ({class_stats[class_name]['correct']}/{class_stats[class_name]['total']})")

    # Visuelle Darstellung
    visualize_test_results(images, labels, predictions, accuracy, class_names)
    
    return accuracy, class_stats

# ==========================================
# TEIL 3: VISUALISIERUNG
# ==========================================

def add_border_to_tensor(tensor_img, border_color, border_width=8):
    """ Hilfsfunktion: Fügt farbigen Rahmen zu einem Bild-Tensor hinzu """
    img = tensor_img.permute(1, 2, 0).numpy()
    color = np.array([0.2, 0.8, 0.2]) if border_color == 'green' else np.array([0.8, 0.2, 0.2]) # Grün / Rot
    
    h, w, c = img.shape
    new_h, new_w = h + 2 * border_width, w + 2 * border_width
    bordered_img = np.ones((new_h, new_w, c))
    
    bordered_img[:, :, :] = color
    bordered_img[border_width:border_width+h, border_width:border_width+w, :] = img
    
    return torch.tensor(bordered_img).permute(2, 0, 1)

def visualize_test_results(images, labels, predictions, accuracy, class_names):
    """
    Zeigt ein 2x2 Grid für alle 4 Klassen mit Ergebnissen.
    (Übernommen aus deinem AdvancedCNN-Entwurf)
    """
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    axes = axes.flatten()
    
    for class_idx, (ax, class_name) in enumerate(zip(axes, class_names)):
        # Bilder für diese Klasse sammeln
        class_images = []
        correct_flags = []
        
        for i in range(len(images)):
            if labels[i] == class_idx:
                class_images.append(images[i])
                correct_flags.append(predictions[i] == labels[i])
        
        if class_images:
            # Bilder mit Rahmen versehen (Grün=Richtig, Rot=Falsch)
            bordered_images = []
            for img, correct in zip(class_images, correct_flags):
                bordered_img = add_border_to_tensor(img, 'green' if correct else 'red')
                bordered_images.append(bordered_img)
            
            # Grid erstellen (max 5 Spalten)
            n_cols = max(1, min(5, len(class_images)))
            grid = make_grid(bordered_images, nrow=n_cols, padding=10)
            
            ax.imshow(grid.permute(1, 2, 0))
            correct_count = sum(correct_flags)
            total_count = len(class_images)
            ax.set_title(f'Klasse: {class_name}\n({correct_count}/{total_count} richtig)', 
                           fontsize=14, fontweight='bold')
        else:
            ax.set_title(f'Klasse: {class_name}\n(Keine Bilder im Testset)', fontsize=14, fontweight='bold')
        
        ax.axis('off')
    
    plt.suptitle(f'4-Klassen-CNN - Test Ergebnisse (Gesamt-Genauigkeit: {accuracy:.2%})', 
                 fontsize=20, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

# ==========================================
# TEIL 4: HAUPTFUNKTION (MAIN)
# ==========================================

def create_weighted_sampler(dataset):
    """
    Erstellt einen WeightedRandomSampler, um die
    unausgeglichenen Trainingsdaten zu balancieren (Oversampling).
    """
    # 1. Zähle Vorkommen jeder Klasse
    class_counts = np.bincount(dataset.targets)
    
    print("Klassenverteilung (Training):")
    for i, count in enumerate(class_counts):
        print(f"  Klasse {i} ({dataset.classes[i]}): {count} Bilder")

    # 2. Berechne Gewicht für jede Klasse (1 / Anzahl)
    # Je seltener die Klasse, desto höher das Gewicht
    class_weights = 1. / torch.tensor(class_counts, dtype=torch.float)

    # 3. Weise jedem einzelnen Bild sein Klassengewicht zu
    # 'dataset.targets' ist eine Liste [2, 2, 0, 1, 2, 3, ...]
    # 'sample_weights' wird [w_2, w_2, w_0, w_1, w_2, w_3, ...]
    sample_weights = class_weights[dataset.targets]

    # 4. Erstelle den Sampler
    # 'replacement=True' ist wichtig, damit Bilder mehrfach gezogen werden können
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights), # Ziehe so viele Bilder wie im Original-Datens.
        replacement=True
    )
    
    print(f"WeightedRandomSampler erstellt.")
    return sampler


def main():
    # Pfade
    base_dir = Path("data_single_cnn")
    train_dir = base_dir / "Lernen"
    validation_dir = base_dir / "Validation"
    test_dir = base_dir / "Test"
    
    # Überprüfen ob Daten existieren
    if not train_dir.exists() or not validation_dir.exists() or not test_dir.exists():
        print("FEHLER: data_single_cnn Ordner nicht gefunden oder unvollständig!")
        print("Führe zuerst 02_data_preparation.py aus!")
        return
    
    # Transformationen
    # Aggressivere Data Augmentation für Trainingsdaten
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5), # NEU: Auch vertikal spiegeln
        transforms.RandomRotation(15),        # NEU: Stärker rotieren
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor(),
    ])
    
    # Einfache Transformation für Validation und Test
    basic_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    # Datensätze laden
    train_dataset = ImageFolder(train_dir, transform=train_transform)
    validation_dataset = ImageFolder(validation_dir, transform=basic_transform)
    test_dataset = ImageFolder(test_dir, transform=basic_transform)
    
    print("=== 4-KLASSEN-CNN TRAINING (MIT OVERSAMPLING) ===")
    print(f"Klassen gefunden: {train_dataset.classes}")
    # Sollte sein: ['Bruch', 'Farbfehler', 'Normal', 'Rest']
    # PyTorch sortiert die Ordnernamen alphabetisch!
    # 'Bruch' = 0, 'Farbfehler' = 1, 'Normal' = 2, 'Rest' = 3
    
    # NEU: Weighted Sampler erstellen
    sampler = create_weighted_sampler(train_dataset)
    
    # DataLoader
    # WICHTIG: Wenn sampler genutzt wird, MUSS shuffle=False sein!
    train_loader = DataLoader(train_dataset, batch_size=16, sampler=sampler, shuffle=False, num_workers=0)
    validation_loader = DataLoader(validation_dataset, batch_size=16, shuffle=False, num_workers=0)
    
    # Modell erstellen
    model = SimpleCNN_4Class(num_classes=4)
    
    # Hyperparameter
    EPOCHS = 25 
    LEARNING_RATE = 0.001
    MODEL_SAVE_PATH = "rad_cnn_best.pth"

    # Training (ohne class_weights, da wir Sampler nutzen)
    train_model(model, EPOCHS, LEARNING_RATE, train_loader, validation_loader, MODEL_SAVE_PATH)
    
    # Finale Evaluation
    # FIX: SyntaxWarning '\L' entfernt
    print("Lade bestes Modell für finale Evaluation auf dem Test-Set...")
    try:
        model.load_state_dict(torch.load(MODEL_SAVE_PATH))
        print("Bestes Modell geladen.")
    except Exception as e:
        print(f"Konnte bestes Modell nicht laden ({e}), verwende letztes Modell.")
    
    # Finale Evaluation auf Testset
    evaluate_on_testset(model, test_dataset, "FINALES TESTSET")

if __name__ == '__main__':
    main()