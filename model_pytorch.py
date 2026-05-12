import torch
import torch.nn as nn
import torch.optim as optim 


class nl_net(nn.Module):
    def __init__(self,input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim,128),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(128,64),
            nn.ReLU(),
            
            nn.Linear(64,1),
            nn.Sigmoid()
        )
    def forward(self,x):
        return self.net(x)
        
        
        
def FCN_train_model(x_train,y_train,epochs,lr):
    model = nl_net(x_train.shape[1])
    criteria = nn.BCELoss()
    optimizer =  optim.Adam(model.parameters(),lr  = lr)
    x_train = torch.tensor(x_train, dtype=torch.float32)
    y_train = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)

    for _ in range(epochs):
        preds = model(x_train)
        loss = criteria(preds, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return model


def FCN_predict(model, x_test):
    x_test = torch.tensor(x_test, dtype=torch.float32)
    with torch.no_grad():
        return model(x_test).numpy().flatten()