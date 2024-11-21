import os
import glob
import json
from torch.utils.data import DataLoader
from data import SeriesData
from model import Classifier
import lightning as L


ckpts = glob.glob('*/*/checkpoints/*.ckpt')
ckpts.sort(key=os.path.getmtime)

dataset = SeriesData('unlabeled_data.csv', train=False)
data_loader = DataLoader(dataset=dataset, batch_size=len(dataset), shuffle=False)
classifier = Classifier()
trainer = L.Trainer() 
pred = trainer.predict(
    classifier, 
    data_loader,
    ckpt_path=ckpts[-1],  # latest
)

is_target = [bool(i) for i in pred[0]]
not_target = [not x for x in is_target]

prediction = {
    "1": dataset.data.columns[is_target].to_list(),
    "0": dataset.data.columns[not_target].to_list(),
}

with open('prediction.json', 'w') as f:
    json.dump(prediction, f, indent=4)