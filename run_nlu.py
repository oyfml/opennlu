from rasa.nlu.training_data import load_data
from rasa.nlu.config import RasaNLUModelConfig
from rasa.nlu.model import Trainer, Interpreter
from rasa.nlu import config


data_path = "data/bus_data_train.md"
pipeline_path = "data/rasa_pipeline/PRE-CONFIG_pretrained_embeddings_convert.yml"
name = "bus_nlu_convert"
model_path = "data/model/rasa/bus_nlu_convert"

training_data = load_data(data_path)
trainer = Trainer(config.load(pipeline_path))
trainer.train(training_data)
trainer.persist(model_path, fixed_model_name=name, persist_nlu_training_data=training_data)
