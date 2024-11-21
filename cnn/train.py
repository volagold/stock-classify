import torch
from torch.utils.data import DataLoader, random_split
from data import SeriesData
from model import Classifier
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks.early_stopping import EarlyStopping


logger = WandbLogger(project='stock-cls')
# ----------------------
# load data
# ----------------------
dataset = SeriesData('training_data.csv')
generator = torch.Generator().manual_seed(42)
train, test = random_split(dataset, [0.8, 0.2], generator=generator)
train_loader = DataLoader(dataset=train, batch_size=32, shuffle=True)
test_loader = DataLoader(dataset=test, batch_size=32, shuffle=False)
# ----------------------
# train and evaluate
# ----------------------
classifier = Classifier()

# lightning doesn't have support for lazymodules yet
device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'
classifier.to(device)
for x, y in train_loader:
    x = x.to(device)
    classifier(x)
    break

trainer = L.Trainer(
    logger=logger,
    max_epochs=20, 
    # max_steps=800, 
    deterministic=True,
    enable_checkpointing=True,
    callbacks=EarlyStopping(monitor="train_acc", patience=50)
    )  
trainer.fit(classifier, train_loader)
trainer.test(classifier, test_loader)