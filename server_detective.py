import flwr as fl
import torch
import csv
import os
import numpy as np
from flwr.common import parameters_to_ndarrays
from model import FraudDetectionMLP, load_data
from sklearn.metrics import auc, precision_recall_curve, confusion_matrix, recall_score

RESULTS_FILE = "results/results_detective_40_r50-2.csv"

class DetectiveMedian(fl.server.strategy.FedMedian):
    def aggregate_fit(self, server_round, results, failures):
        print(f"\n DETECTIVE Analyzing Bank Updates for Round {server_round}...")
        
        client_updates = []
        for client, fit_res in results:
            bank_id = fit_res.metrics.get("bank_id", client.cid)
            
            weights = parameters_to_ndarrays(fit_res.parameters)
            flat_weights = np.concatenate([w.flatten() for w in weights])
            
            client_updates.append((bank_id, flat_weights, client, fit_res))
            
        all_weights = np.array([w for _, w, _, _ in client_updates])
        median_vector = np.median(all_weights, axis=0)
        
        distances = []
        for bank_id, w, client, fit_res in client_updates:
            dist = np.linalg.norm(w - median_vector)
            distances.append((bank_id, dist, client, fit_res))
            
        distances.sort(key=lambda x: x[1], reverse=True)
        
        just_dists = [d for _, d, _, _ in distances]
        median_dist = np.median(just_dists)
        mad = np.median([np.abs(d - median_dist) for d in just_dists])
        
        threshold = max(median_dist + (3 * mad), median_dist * 1.5)
        
        print(f"THREAT REPORT (Median Dist: {median_dist:.4f} | Threshold: {threshold:.4f})")
        
        trusted_results = []
        for bank_id, dist, client, fit_res in distances:
            if dist > threshold:
                print(f" SUSPICIOUS: Bank ID {bank_id} (Distance: {dist:.4f}) -> BANNED!")
            else:
                print(f" Normal: Bank ID {bank_id} (Distance: {dist:.4f})")
                trusted_results.append((client, fit_res))
                
        if not trusted_results:
            print("WARNING: All banks flagged! Falling back to accepting all.")
            trusted_results = results
                
        return super().aggregate_fit(server_round, trusted_results, failures)

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
        
        precision, recall_curve, _ = precision_recall_curve(y_true, y_probs)
        auprc = auc(recall_curve, precision)
        y_pred = (np.array(y_probs) > 0.5).astype(int)
        
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        recall = recall_score(y_true, y_pred)
        
        print(f"[ROUND {server_round}] Global Recall: {recall:.4f}")
        
        with open(RESULTS_FILE, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([server_round, auprc, recall, tn, fp, fn, tp])
            
        return 0.0, {"auprc": auprc}
        
    return evaluate

if __name__ == "__main__":
    strategy = DetectiveMedian(
        evaluate_fn=get_evaluate_fn(),
        min_fit_clients=10,
        min_available_clients=10,
    )
    
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=50),
        strategy=strategy,
    )