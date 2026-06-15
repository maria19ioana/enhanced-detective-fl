import torch
import torch.nn as nn
import torch.optim as optim
import csv
import os
import numpy as np
from sklearn.metrics import auc, precision_recall_curve, confusion_matrix, recall_score
from model import FraudDetectionMLP, load_data

NUM_EPOCHS = 50 
#RESULTS_FILE = "results_centralized_baseline.csv"
RESULTS_FILE = "results/50_rounds/results_centralized_baseline_r50.csv"


TRAIN_DATA_PATH = 'C:/Users/ivani/Desktop/Disertatie/notebook/global_train.csv' 
TEST_DATA_PATH = 'C:/Users/ivani/Desktop/Disertatie/notebook/global_test.csv'

def train_and_evaluate():
    print("Loading centralized datasets...")
    train_loader = load_data(TRAIN_DATA_PATH, batch_size=256)
    test_loader = load_data(TEST_DATA_PATH, batch_size=256)
    
    model = FraudDetectionMLP()
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    if not os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Epoch", "Loss", "AUPRC", "Recall", "TN", "FP", "FN", "TP"])

    print(f"Starting Centralized Training for {NUM_EPOCHS} Epochs...")
    
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        for b_features, b_labels in train_loader:
            optimizer.zero_grad()
            outputs = model(b_features)
            
            loss = criterion(outputs, b_labels.unsqueeze(1) if outputs.shape != b_labels.shape else b_labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        avg_loss = epoch_loss / len(train_loader)
        
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
        
        with open(RESULTS_FILE, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([epoch, avg_loss, auprc, recall, tn, fp, fn, tp])
            
        print(f"Epoch {epoch}/{NUM_EPOCHS} | Loss: {avg_loss:.4f} | Recall: {recall:.4f} | AUPRC: {auprc:.4f} | Missed Frauds (FN): {fn}")

if __name__ == "__main__":
    train_and_evaluate()
    print(f"Results saved to {RESULTS_FILE}.")