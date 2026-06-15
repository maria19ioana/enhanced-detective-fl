import flwr as fl
import torch
import numpy as np
import csv
import os
from model import FraudDetectionMLP, load_data
from sklearn.metrics import auc, precision_recall_curve, confusion_matrix, recall_score

RESULTS_FILE = "results/50_rounds/results_attack_40-model-poisoning_r50.csv"

if not os.path.exists(RESULTS_FILE):
    with open(RESULTS_FILE, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Round", "AUPRC", "Recall", "TN", "FP", "FN", "TP"])

def get_evaluate_fn():
    test_loader = load_data('global_test.csv', batch_size=256)
    
    if not os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Round", "AUPRC", "Recall", "TN", "FP", "FN", "TP"])

    def evaluate(server_round, parameters, config):
        model = FraudDetectionMLP()
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        model.load_state_dict(state_dict, strict=True)
        
        model.eval()
        y_true, y_probs = [], []
        
        with torch.no_grad():
            for b_features, b_labels in test_loader:
                outputs = model(b_features)
                y_probs.extend(outputs.numpy())
                y_true.extend(b_labels.numpy())
        
        y_probs = np.nan_to_num(np.array(y_probs), nan=0.0, posinf=1.0, neginf=0.0)
        
        precision, recall_curve_vals, _ = precision_recall_curve(y_true, y_probs)
        auprc = auc(recall_curve_vals, precision)
        
        y_pred = (y_probs > 0.5).astype(int)
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        recall = recall_score(y_true, y_pred)
        
        print(f"\n--- ROUND {server_round} GLOBAL METRICS ---")
        print(f"Global AUPRC: {auprc:.4f} | Recall: {recall:.4f}")
        print(f"Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
        
        with open(RESULTS_FILE, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([server_round, auprc, recall, tn, fp, fn, tp])
            
        return 0.0, {"auprc": auprc}
    
    return evaluate

strategy = fl.server.strategy.FedAvg(
    evaluate_fn=get_evaluate_fn(),
    min_fit_clients=10,
    min_available_clients=10,
)

if __name__ == "__main__":
    print("Starting Attack Server on port 8080...")
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=50),
        strategy=strategy,
    )