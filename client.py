import flwr as fl
import torch
from model import FraudDetectionMLP, load_data

class FraudClient(fl.client.NumPyClient):
    def __init__(self, bank_id):
        self.bank_id = bank_id  
        self.model = FraudDetectionMLP()
        self.train_loader = load_data(f'client_data/bank_{bank_id}.csv')
        self.device = torch.device("cpu") 

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        criterion = torch.nn.BCELoss()
        
        self.model.train()
        for _ in range(1): 
            for images, labels in self.train_loader:
                optimizer.zero_grad()
                criterion(self.model(images), labels).backward()
                optimizer.step()
        
        return self.get_parameters(config={}), len(self.train_loader.dataset), {"bank_id": self.bank_id}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        return 0.0, len(self.train_loader.dataset), {"accuracy": 0.0}
    

if __name__ == "__main__":
    import sys
    bank_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    
    print(f"Bank {bank_id} connecting to Central Server...")
    fl.client.start_client(
        server_address="127.0.0.1:8080", 
        client=FraudClient(bank_id).to_client()
    )