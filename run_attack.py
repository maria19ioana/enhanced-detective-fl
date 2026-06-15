import subprocess
import time
import sys

print("1. Starting Central Server (Attack Phase)...")
server_process = subprocess.Popen([sys.executable, "server_attack.py"])

time.sleep(5)

print("\n2. Starting 10 Bank Clients (6 Honest, 4 Malicious)...")
client_processes = []
for i in range(10):
    p = subprocess.Popen([sys.executable, "client_attack.py", str(i)])
    client_processes.append(p)

server_process.wait()

print("\nAttack Simulation complete. Shutting down banks...")
for p in client_processes:
    p.terminate()
