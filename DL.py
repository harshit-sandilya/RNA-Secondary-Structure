from utils.dl_data import DataModule
from utils.dl_model import Model
import pytorch_lightning as pl

dm = DataModule()
model = Model()
trainer = pl.Trainer(max_epochs=5)
trainer.fit(model, dm)
trainer.test(model, dm)
