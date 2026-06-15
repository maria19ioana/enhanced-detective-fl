import subprocess
import time
import sys

print("1. Starting Central Server...")
server_process = subprocess.Popen([sys.executable, "server.py"])

time.sleep(5)

print("\n2. Starting 10 Bank Clients...")
client_processes = []
for i in range(10):
    p = subprocess.Popen([sys.executable, "client.py", str(i)])
    client_processes.append(p)

server_process.wait()

print("\nSimulation complete. Shutting down banks...")
for p in client_processes:
    p.terminate()