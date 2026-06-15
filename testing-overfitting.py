import pandas as pd
import matplotlib.pyplot as plt

try:
    df_test = pd.read_csv('global_test.csv')
    total_samples = len(df_test)
    fraud_samples = df_test['Class'].sum()
    print(f"Test Set Total Samples: {total_samples}")
    print(f"Test Set Fraud Samples: {fraud_samples} ({(fraud_samples/total_samples)*100:.2f}%)")
except Exception as e:
    print(f"Error reading global_test.csv: {e}")

try:
    df_base = pd.read_csv('results_baseline_r20.csv')
    
    plt.figure(figsize=(10, 5))
    
    plt.plot(df_base['Round'], df_base['Recall'], label='Recall', marker='o')
    plt.plot(df_base['Round'], df_base['AUPRC'], label='AUPRC', marker='s')
    
    plt.title('Baseline Evaluation Metrics over 20 Rounds (Test Set)')
    plt.xlabel('Round')
    plt.ylabel('Score')
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True)
    plt.show()
    #plt.savefig('learning_curve.png')
    
    print("\nMetrics progression (last 5 rounds):")
    print(df_base[['Round', 'AUPRC', 'Recall']].tail())
except Exception as e:
    print(f"Error analyzing training curve: {e}")