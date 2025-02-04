# Neural Networks Cycle.py

#------------------------------------------------------------------
# Model Definition

import torch
from torch_geometric.nn.models import AttentiveFP
#torch.multiprocessing.set_start_method('spawn')

class LinearReLU(torch.nn.Module):
    def __init__(self, in_size, out_size):
        super().__init__()
        
        self.in_size, self.out_size = in_size, out_size
        
        self.linear = torch.nn.Linear(self.in_size, self.out_size)
        self.relu = torch.nn.ReLU()

        return

    def forward(self, x):
         return self.relu(self.linear(x))
    
    def reset_parameters(self):
        self.linear.reset_parameters()

class OdorModel(torch.nn.Module):
    def __init__(self, in_size, out_size, hidden_size, num_hidden_layer):
        super().__init__()

        self.in_size, self.out_size, self.hidden_size = in_size, out_size, hidden_size
        self.num_hidden_layer = num_hidden_layer

        self.in_layer = LinearReLU(self.in_size, self.hidden_size)
        self.hidden_layer = torch.nn.Sequential(
            *[LinearReLU(self.hidden_size, self.hidden_size) for _ in range(num_hidden_layer)]
        )
        self.out_layer = torch.nn.Linear(self.hidden_size, self.out_size)

    def forward(self, x):
        x = self.in_layer(x)
        for hl in self.hidden_layer:
            x = hl(x)
        return self.out_layer(x)

    def reset_parameters(self):
        self.in_layer.reset_parameters()
        for hl in self.hidden_layer:
            hl.reset_parameters()
        self.out_layer.reset_parameters()
        
#------------------------------------------------------------------
# NN Cycle definition

class Cycle():
    def __init__(self, parameters, loaders, gpuid = 0):

        self.gpuid = gpuid
        
        torch.device(f'cuda:{self.gpuid}')
        self.parameters = parameters 

        self.train_loader, self.validation_loader,  self.test_loader = tuple(loaders)

        self.model = OdorModel(
                        in_size = self.parameters['in_size'],
                        hidden_size = self.parameters['hidden_size'],
                        out_size = self.parameters['out_size'],
                        num_hidden_layer = self.parameters['num_hidden_layer'],
        ).to(f'cuda:{self.gpuid}')
        
        self.optimizer = torch.optim.Adam(
                            self.model.parameters(), 
                            lr = self.parameters['lr']
        )
        self.criterion = torch.nn.BCEWithLogitsLoss(pos_weight = torch.tensor([self.parameters['pos_weight'] for _ in range(138)]).to(f'cuda:{self.gpuid}'))
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                            self.optimizer
        )

        return

    # Training

    def train(self):
        self.model.train()
    
        tot_loss = 0.0
    
        for x, y in self.train_loader:
            self.optimizer.zero_grad()
    
            out = self.model(x.to(f'cuda:{self.gpuid}'))
    
            loss = self.criterion(out, y.to(torch.float).to(f'cuda:{self.gpuid}'))
             
            loss.backward()
    
            self.optimizer.step()
    
            tot_loss += loss.item()
    
        return tot_loss / len(self.train_loader)
    
    # Validation
    
    def validation(self):
        self.model.eval()
    
        tot_loss = 0.0
    
        with torch.no_grad():
            for x, y in self.validation_loader:
                out = self.model(x.to(f'cuda:{self.gpuid}'))
                
                loss = self.criterion(out, y.to(torch.float).to(f'cuda:{self.gpuid}'))
                
                tot_loss += loss.item()
    
        return tot_loss / len(self.validation_loader)
        
        def test(model, loader):
            self.model.eval()
            test_loss = 0.0
            with torch.no_grad():

                self.pred = [] ; self.true = []

                for x, y in test_loader:
                    out = self.model(x.to(f'cuda:{model.gpuid}'))
                    test_loss += self.criterion(out, y.to(torch.float).to(f'cuda:{model.gpuid}'))

                    out = torch.nn.functional.sigmoid(out)
        
                    pred.append(out.cpu().numpy())
                    true.append(y.cpu().numpy())

            test_loss = test_loss.item() / len(self.test_loader)
            pred = np.concatenate(pred)
            true = np.concatenate(true)

            self.pred = pred.astype(int)
            self.true = true.astype(int)

            return self.pred, self.true
    
    def run(self, epochs, verbose = False):
        
        self.trn_loss = []; self.val_loss = []
        
        for e in range(epochs):
            self.trn_loss.append(self.train())
            self.val_loss.append(self.validation())

            if verbose:
                print(f"Epoch : {e:-05d} | Trn. Loss : {self.trn_loss[-1]:.3f} | Val. Loss : {self.val_loss[-1]:.3f}")
    
        print("Training Complete!")

        return self.val_loss[-1]

#------------------------------------------------------------------
# GNN Cycle definition

class GNN():
    def __init__(self, parameters, loaders, gpuid = 1):

        self.gpuid = gpuid

        torch.device(f'cuda:{self.gpuid}')
        self.parameters = parameters

        self.train_loader, self.validation_loader, self.test_loader = tuple(loaders)

        model = AttentiveFP(in_channels = self.parameters['in_channels'],
                            hidden_channels = self.parameters['hidden_channels'],
                            out_channels = self.parameters['out_channels'],
                            edge_dim = self.parameters['edge_dim'],
                            num_layers = self.parameters['num_layers'],
                            num_timesteps = self.parameters['num_timesteps'],
                            dropout = self.parameters['dropout'],
                           ).to(f'cuda:{self.gpuid}')

        for batch, y in self.test_loader:
            batch.x = batch.x.to(torch.float.to(f'cuda:{self.gpuid}'))
            break

        self.optimizer = torch.optim.Adam(model.parameters(), lr = self.parameters['lr'])
        self.criterion = torch.nn.BCEWithLogitsLoss(pos_weight = torch.tensor([self.parameter['pos_weight'] for _ in range(138)]).to(f'cuda:{self.gpuid}')
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer)

        return

        def train(self):
            self.model.train()
            
            tot_loss = 0.0
            
            for batch, y in self.train_loader:
                self.optimizer.zero_grad()
                batch.x = batch.x.to(torch.float.to(f'cuda:{self.gpuid}'))
                out = self.model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)

                loss = criterion(out, y.to(torch.float).to(f'cuda:{self.gpuid}'))
                loss.backward()
                self.optimizer.step()

                tot_loss += loss.item()

        return tot_loss / len(loader)

        def validation(model, loader):
            model.eval()
            tot_loss = 0.0
            with torch.no_grad():
                for batch, y in loader:
                    batch.x = batch.x.to(torch.float.to(f'cuda:{self.gpuid}'))

                    out = self.model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)

                    loss = criterion(out, y.to(torch.float).to(f'cuda:{self.gpuid}'))
    
                    tot_loss += loss.item()
    
            return tot_loss / len(loader)
        
        def run(self, epochs, verbose = False):
            self.trn_loss = []; self.val_loss = []

            for e in range(epochs):
                self.trn_loss.append(self.train())
                self.val_loss.append(self.validation())

                if verbose:
                    print(f"Epoch : {e:05d} | Trn. Loss : {trn_loss[-1]:.3f} | Val. Loss : {val_loss[-1]:.3f}")

            print("Training Complete!")

            return self.val_loss[-1]