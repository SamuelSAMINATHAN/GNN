import json
import os
from typing import List, Tuple, Optional

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_auc_score, RocCurveDisplay
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.optim import Adam
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool


DATA_PATH = os.path.join("data", "processed", "bad_graph_gnn.json")
MITRE_MAPPING_PATH = os.path.join("data", "mitre_stix_mapping.json")
MODEL_OUTPUT_DIR = os.path.join("model_output")
MODEL_OUTPUT_PATH = os.path.join(MODEL_OUTPUT_DIR, "gnn_cybrain.pth")
ATTACK_GRAPH_FIG_PATH = os.path.join(MODEL_OUTPUT_DIR, "attack_graph_example.png")
LEARNING_CURVES_FIG_PATH = os.path.join(MODEL_OUTPUT_DIR, "learning_curves.png")
CONFUSION_MATRIX_FIG_PATH = os.path.join(MODEL_OUTPUT_DIR, "confusion_matrix.png")
ROC_AUC_FIG_PATH = os.path.join(MODEL_OUTPUT_DIR, "roc_auc.png")


class GNNDataProcessor:
    """
    Charge le dataset JSON unique et le convertit en un objet Data de PyTorch Geometric.
    Gère l'encodage des binaires et la création des features pour chaque nœud.
    """

    def __init__(self, data_path: str = DATA_PATH) -> None:
        self.data_path = data_path
        self.label_encoder = LabelEncoder()

    def _load_raw_data(self) -> dict:
        with open(self.data_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _fit_label_encoder(self, nodes: List[dict]) -> None:
        binaries: List[str] = [str(node.get("binary_name", "unknown")) for node in nodes]
        self.label_encoder.fit(binaries if binaries else ["unknown"])

    def load_pyg_data(self, test_size: float = 0.2, random_state: int = 42) -> Tuple[List[Data], List[Data]]:
        raw_data = self._load_raw_data()
        nodes = raw_data.get("nodes", [])
        links = raw_data.get("links", [])
        
        self._fit_label_encoder(nodes)

        # Mapping des nœuds pour edge_index
        # On utilise (binary_name, uid) comme clé unique pour identifier les nœuds
        node_to_idx = {}
        feat_list: List[List[float]] = []
        y_list: List[int] = []

        for idx, node in enumerate(nodes):
            binary_name = str(node.get("binary_name", "unknown"))
            uid = float(node.get("uid", 0))
            args_count = float(node.get("arguments_count", 0))
            target = int(node.get("target", 0))

            binary_id = float(self.label_encoder.transform([binary_name])[0])
            feat_list.append([binary_id, uid, args_count])
            y_list.append(target)
            
            # Stocker l'index du nœud
            node_key = (binary_name, int(uid))
            node_to_idx[node_key] = idx

        x = torch.tensor(feat_list, dtype=torch.float32)
        y = torch.tensor(y_list, dtype=torch.float32)

        # Construction de l'edge_index
        edge_indices: List[Tuple[int, int]] = []
        for link in links:
            src = link.get("source", {})
            dst = link.get("target", {})
            
            src_key = (str(src.get("binary_name", "unknown")), int(src.get("uid", 0)))
            dst_key = (str(dst.get("binary_name", "unknown")), int(dst.get("uid", 0)))
            
            if src_key in node_to_idx and dst_key in node_to_idx:
                edge_indices.append((node_to_idx[src_key], node_to_idx[dst_key]))

        if not edge_indices:
            edge_index = torch.empty((2, 0), dtype=torch.long)
        else:
            edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()

        # Création de l'objet Data unique
        data = Data(x=x, edge_index=edge_index, y=y)
        
        # Split des indices pour Node Classification
        indices = list(range(len(nodes)))
        train_idx, test_idx = train_test_split(
            indices, test_size=test_size, random_state=random_state, stratify=y_list
        )
        
        # Création des masques
        train_mask = torch.zeros(len(nodes), dtype=torch.bool)
        test_mask = torch.zeros(len(nodes), dtype=torch.bool)
        train_mask[train_idx] = True
        test_mask[test_idx] = True
        
        data.train_mask = train_mask
        data.test_mask = test_mask

        # On retourne une liste contenant le graphe unique pour rester compatible avec DataLoader
        # ou on peut adapter la boucle d'entraînement.
        return [data], [data]


class CyBrainGNN(nn.Module):
    """
    Modèle GNN pour classification binaire au niveau des nœuds.
    """

    def __init__(self, in_channels: int, hidden_channels: int = 64, dropout: float = 0.3) -> None:
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.lin = nn.Linear(hidden_channels, 1)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        # On retire global_mean_pool pour faire de la Node Classification
        out = self.lin(x).view(-1)
        return out


def load_mitre_mapping(path: str = MITRE_MAPPING_PATH) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def mitre_for_attack(mapping: dict, default_technique: str = "T1003") -> str:
    if not mapping:
        return default_technique
    return next(iter(mapping.keys()), default_technique)


def train_and_evaluate(
    epochs: int = 20,
    batch_size: int = 16,
    lr: float = 1e-3,
    device: Optional[str] = None,
) -> None:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)

    processor = GNNDataProcessor()
    train_graphs, test_graphs = processor.load_pyg_data()

    train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_graphs, batch_size=batch_size, shuffle=False)

    in_channels = train_graphs[0].x.size(1)
    model = CyBrainGNN(in_channels=in_channels).to(device)
    optimizer = Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    mitre_mapping = load_mitre_mapping()

    history = {
        "train_loss": [],
        "test_loss": [],
        "train_acc": [],
        "test_acc": [],
        "train_f1": [],
        "test_f1": [],
    }

    best_test_f1 = 0.0
    best_attack_graph: Optional[Data] = None
    best_attack_scores: Optional[torch.Tensor] = None

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses: List[float] = []
        all_train_labels: List[int] = []
        all_train_preds: List[int] = []

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            logits = model(batch.x, batch.edge_index)
            
            # Node classification : perte uniquement sur le masque d'entraînement
            loss = criterion(logits[batch.train_mask], batch.y[batch.train_mask])
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            probs = torch.sigmoid(logits[batch.train_mask]).detach().cpu().numpy()
            preds = (probs >= 0.5).astype(int)
            labels = batch.y[batch.train_mask].detach().cpu().numpy().astype(int)
            all_train_labels.extend(labels.tolist())
            all_train_preds.extend(preds.tolist())

        train_loss = float(np.mean(train_losses)) if train_losses else 0.0
        train_acc = accuracy_score(all_train_labels, all_train_preds) if all_train_labels else 0.0
        train_f1 = f1_score(all_train_labels, all_train_preds, zero_division=0) if all_train_labels else 0.0

        model.eval()
        test_losses: List[float] = []
        all_test_labels: List[int] = []
        all_test_probs: List[float] = []
        all_test_preds: List[int] = []

        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                logits = model(batch.x, batch.edge_index)
                
                # Perte uniquement sur le masque de test
                loss = criterion(logits[batch.test_mask], batch.y[batch.test_mask])
                test_losses.append(loss.item())

                probs = torch.sigmoid(logits[batch.test_mask]).detach().cpu().numpy()
                preds = (probs >= 0.5).astype(int)
                labels = batch.y[batch.test_mask].detach().cpu().numpy().astype(int)

                all_test_labels.extend(labels.tolist())
                all_test_probs.extend(probs.tolist())
                all_test_preds.extend(preds.tolist())

            test_loss = float(np.mean(test_losses)) if test_losses else 0.0
            test_acc = accuracy_score(all_test_labels, all_test_preds) if all_test_labels else 0.0
            test_f1 = f1_score(all_test_labels, all_test_preds, zero_division=0) if all_test_labels else 0.0

        history["train_loss"].append(train_loss)
        history["test_loss"].append(test_loss)
        history["train_acc"].append(train_acc)
        history["test_acc"].append(test_acc)
        history["train_f1"].append(train_f1)
        history["test_f1"].append(test_f1)

        print(
            f"Epoch {epoch:03d} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} F1: {train_f1:.4f} | "
            f"Test Loss: {test_loss:.4f} Acc: {test_acc:.4f} F1: {test_f1:.4f}"
        )

        if test_f1 > best_test_f1 and all_test_labels:
            best_test_f1 = test_f1
            torch.save(model.state_dict(), MODEL_OUTPUT_PATH)

        # On garde une trace des prédictions pour la visualisation finale
        if all_test_labels:
            best_attack_graph = test_graphs[0] # C'est le même graphe
            best_attack_scores = torch.sigmoid(model(best_attack_graph.x.to(device), best_attack_graph.edge_index.to(device))).detach().cpu()

    print(f"Modèle sauvegardé dans {MODEL_OUTPUT_PATH}")

    y_true = np.array(all_test_labels, dtype=int)
    y_pred = np.array(all_test_preds, dtype=int)
    y_prob = np.array(all_test_probs, dtype=float) if all_test_probs else None

    plot_learning_curves(history)
    plot_confusion_matrix(y_true, y_pred)
    if y_prob is not None and len(np.unique(y_true)) > 1:
        plot_roc_auc(y_true, y_prob)

    if best_attack_graph is not None:
        technique = mitre_for_attack(mitre_mapping)
        print(f"Anomalie détectée, technique MITRE probable : {technique}")
        # On passe les scores de prédiction pour la visualisation
        visualize_attack_graph(best_attack_graph, ATTACK_GRAPH_FIG_PATH, scores=best_attack_scores)


def plot_learning_curves(history: dict) -> None:
    epochs = range(1, len(history["train_loss"]) + 1)
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["test_loss"], label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Courbe de Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["train_acc"], label="Train Acc")
    plt.plot(epochs, history["test_acc"], label="Test Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Courbe d'Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.savefig(LEARNING_CURVES_FIG_PATH)
    plt.close()


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=["Sain", "Attaque"],
        yticklabels=["Sain", "Attaque"],
        ylabel="Vrai label",
        xlabel="Prédiction",
        title="Matrice de confusion",
    )

    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    fig.tight_layout()
    plt.savefig(CONFUSION_MATRIX_FIG_PATH)
    plt.close()


def plot_roc_auc(y_true: np.ndarray, y_prob: np.ndarray) -> None:
    auc = roc_auc_score(y_true, y_prob)
    RocCurveDisplay.from_predictions(y_true, y_prob)
    plt.title(f"ROC-AUC (AUC={auc:.3f})")
    plt.savefig(ROC_AUC_FIG_PATH)
    plt.close()


def visualize_attack_graph(graph: Data, fig_path: str, scores: Optional[torch.Tensor] = None) -> None:
    num_nodes = graph.num_nodes
    G = nx.DiGraph() if graph.edge_index.numel() > 0 else nx.Graph()
    G.add_nodes_from(range(num_nodes))

    if graph.edge_index.numel() > 0:
        edges = graph.edge_index.t().cpu().numpy().tolist()
        G.add_edges_from(edges)

    # Utilisation des labels 'target' s'ils existent, sinon par défaut
    y = graph.y.cpu().numpy() if hasattr(graph, 'y') else np.zeros(num_nodes)
    
    node_colors = []
    for i in range(num_nodes):
        # Si un score est fourni et > 0.5 (prédiction d'attaque)
        if scores is not None and scores[i] >= 0.5:
            node_colors.append("red")
        # Si c'est une attaque réelle (ground truth)
        elif y[i] == 1:
            node_colors.append("orange")
        else:
            node_colors.append("skyblue")

    plt.figure(figsize=(10, 10))
    pos = nx.spring_layout(G, seed=42, k=0.15)
    
    nx.draw(
        G, pos, 
        with_labels=True, 
        node_color=node_colors, 
        node_size=300, 
        arrows=True, 
        font_size=8
    )
    
    plt.title("Graphe des processus : Bleu=Sain, Orange=Cible, Rouge=Détecté")
    plt.savefig(fig_path)
    plt.close()


if __name__ == "__main__":
    train_and_evaluate()

