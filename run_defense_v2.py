import subprocess
import time
import sys

NUM_ROUNDS = 50 
print(f"--- Starting Defense Simulation (IID, {NUM_ROUNDS} Rounds) ---")

print("1. Launching Robust Aggregator Server...")
server_process = subprocess.Popen([sys.executable, "server_enhanced_detective.py"])

time.sleep(5) 

print(f"2. Launching 10 Clients (2 Malicious, 8 Honest)...")
client_processes = []
for i in range(10):
    p = subprocess.Popen([sys.executable, "client_attack.py", str(i)])
    client_processes.append(p)

try:
    server_process.wait()
except KeyboardInterrupt:
    print("\nManual stop detected. Cleaning up...")

print("\n3. Shutting down all bank processes...")
for p in client_processes:
    p.terminate()

server_process.terminate()
print("Simulation complete. All processes closed.")