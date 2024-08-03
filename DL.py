from utils.dl_data import DataModule
from utils.dl_model import Model
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

dm = DataModule()
model = Model()
logger = TensorBoardLogger("logs", name="nn")
trainer = pl.Trainer(max_epochs=1, logger=logger)
trainer.fit(model, dm)
trainer.test(model, dm)
