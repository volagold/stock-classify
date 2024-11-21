import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.LazyConv1d(out_channels=32, kernel_size=5, stride=2, padding=0)
        self.conv2 = nn.LazyConv1d(out_channels=16, kernel_size=5, stride=2, padding=0)
        self.conv3 = nn.LazyConv1d(out_channels=8, kernel_size=5, stride=1, padding=0)

        self.fc1 = nn.LazyLinear(out_features=512)
        self.fc2 = nn.LazyLinear(out_features=128)
        self.fc3 = nn.LazyLinear(out_features=2)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv3(x))
        x = x.flatten(start_dim=1)
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = self.fc3(x)
        
        return x

class Classifier(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.cnn = CNN()
        self.save_hyperparameters()

    def forward(self, x):
        return self.cnn(x)

    def training_step(self, batch, batch_idx):
        loss, acc = self._get_loss_acc(batch)
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        loss, acc = self._get_loss_acc(batch)
        self.log("test_loss", loss)
        self.log("test_acc", acc)
        return acc
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        logits = self.forward(batch)
        probs = F.softmax(logits, dim=-1)
        y = torch.argmax(probs, dim=-1)
        return y
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def _get_loss_acc(self, batch):
        x, y = batch
        logits = self.forward(x)
        loss = F.cross_entropy(logits, y)
        probs = F.softmax(logits, dim=-1)
        y_hat = torch.argmax(probs, dim=-1)  # torch.multinomial(probs, 1, replacement=True)
        acc = sum(y_hat == y) / len(y) 
        acc = round(acc.item() * 100, 2)
        return loss, acc