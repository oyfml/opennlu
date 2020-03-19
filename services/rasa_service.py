from rasa.nlu.training_data import load_data
from rasa.nlu.config import RasaNLUModelConfig
from rasa.nlu.model import Trainer, Interpreter
from rasa.nlu import config
import json
import numpy as np

# Dict to store rasa models
class RASA_dict(object):
    def __init__(self):
        self.rasa_list = {}

    def create_new(self, name, data, pipeline):
        # create new RASA model
        self.rasa_list[name] = RASA(name, data, pipeline)
        return self.rasa_list[name]

    def get_model(self, name):
        # retrieve RASA model
        return self.rasa_list[name]

    def get_all(self):
        # return list of all RASA models
        return list(self.rasa_list.keys())

    def size(self):
        # get number of RASA models
        return len(self.rasa_list)


# rasa model
class RASA(object):
    def __init__(self, name, data, pipeline):
        self.name = name
        self.data = data #training data
        self.pipeline = pipeline #components file

    # option 1: train new model
    def train(self):
        # loading the nlu training samples
        training_data = load_data(self.data)
        # trainer to educate our pipeline
        trainer = Trainer(config.load(self.pipeline))
        # train the model
        self.interpreter = trainer.train(training_data)
        # store it for future use
        self.model_directory = trainer.persist("opennlu/data/model/rasa", fixed_model_name=self.name, persist_nlu_training_data=training_data)
      
    # option 2: load existing model
    def load(self, model_directory):
        self.model_directory = model_directory
        self.interpreter = Interpreter.load(self.model_directory,None)

    # evaluate single input message
    def results(self, message):
        # get results from interpreter
        result = self.interpreter.parse(message)
        return result


    # evaluates metrics of entire training or test data set, using sklearn
    def evaluate_metrics(self, path): # test data path: "opennlu/data/____.md"
        from rasa.nlu.test import compute_metrics   
        test_data = load_data(path)
        [self.intent_metrics, self.entity_metrics, _,self.intent_results, self.entity_results, __] = compute_metrics(self.interpreter,test_data)
        # process into lists (1D/2D) for flask view

        # intent_metrics: dict with accuracy, f1, precision
        f1 = self.intent_metrics['F1-score'][0]
        p = self.intent_metrics['Precision'][0]
        int_metrics = [self.intent_metrics['Accuracy'][0], self.intent_metrics['F1-score'][0], self.intent_metrics['Precision'][0]] 
        
        # entity_metrics: list of extractors, each: dict with acc, f1, precision
        ex = list(self.entity_metrics.keys()) #list of extractors
        ent_metrics = [[e, self.entity_metrics[e]['Accuracy'][0], self.entity_metrics[e]['F1-score'][0],self.entity_metrics[e]['Precision'][0]] for i,e in enumerate(ex)] 

        # intent_results: list with object:{target, prediction, message, confidence}
        int_target = [obj.intent_target for obj in self.intent_results]
        int_prediction = [obj.intent_prediction for obj in self.intent_results]
        message = [obj.message for obj in self.intent_results]
        confidence = [obj.confidence for obj in self.intent_results]
        int_results = [int_target, int_prediction, message, confidence] 
        
        # entity_results: list with object:{target, prediction}; target & predict: dict with value, entity type, confidence(for predict)
        ent_targets = [obj.entity_targets for obj in self.entity_results]
        et = list()
        for list_ in ent_targets:
            if len(list_) == 0:
                et.append([['','']])
            else:
                temp = list()
                for obj in list_:
                        temp.append([obj['value'],obj['entity']])
                et.append(temp)
        # et: each row = 1 data, column1:entity_value column2:entity_type
        ent_predictions = [obj.entity_predictions for obj in self.entity_results]
        ep = list()
        for list_ in ent_predictions:
            if len(list_) == 0:
                ep.append([['','','','']])
            else:
                temp = list()
                for dic in list_:
                    temp.append([dic['value'],dic['entity'],dic['confidence'],dic['extractor']])
                ep.append(temp)
        # ep: each row = 1 data, each column = 1 extractor, 3rd dim: [value,type,confidence,extractor]
        ent_results = [et,ep] 
        
        return [int_metrics, ent_metrics, int_results, ent_results]


    # get entity extractors of pipeline
    def get_extractor(self):
        from rasa.nlu.test import get_entity_extractors
        return list(get_entity_extractors(self.interpreter))


    # creates confusion matrix after evaluation, with intent results
    def compute_confusion_matrix(self,save_path):
        from sklearn.metrics import confusion_matrix
        from sklearn.utils.multiclass import unique_labels
        import matplotlib.pyplot as plt

        from rasa.nlu.test import _targets_predictions_from
        from rasa.nlu.test import plot_confusion_matrix

        plt.gcf().clear()
        # extract target and predicted labels from results
        self.target_intents, self.predicted_intents = _targets_predictions_from(self.intent_results,"intent_target","intent_prediction")
        # compute confusion matrix
        cnf_matrix = confusion_matrix(self.target_intents, self.predicted_intents)
        # get list of unique labels from target and predicted
        labels = unique_labels(self.target_intents, self.predicted_intents)
        plot_confusion_matrix(
            cnf_matrix,
            classes=labels,
            title="Intent Confusion matrix",
            out=save_path,
        )
    

    # create histogram of confidence distribution
    def compute_histogram(self,save_path):
        import matplotlib.pyplot as plt
        from rasa.nlu.test import plot_histogram
        
        plt.gcf().clear()
        # hits histogram
        pos_hist = [
            r.confidence for r in self.intent_results if r.intent_target == r.intent_prediction
        ]
        # miss histogram
        neg_hist = [
            r.confidence for r in self.intent_results if r.intent_target != r.intent_prediction
        ]
        plot_histogram([pos_hist, neg_hist],save_path)


    # get individual intent and entity type report; only to be executed after evaluate_metrics()
    def evaluation_get_individual_report(self):
        from sklearn import metrics
        intent_report = metrics.classification_report(
            self.target_intents, self.predicted_intents, output_dict=True
        )
        from rasa.nlu.test import align_all_entity_predictions, merge_labels, substitute_labels, get_entity_extractors

        extractors = get_entity_extractors(self.interpreter)

        aligned_predictions = align_all_entity_predictions(self.entity_results, extractors)
        merged_targets = merge_labels(aligned_predictions)
        merged_targets = substitute_labels(merged_targets, "O", "no_entity")

        entity_report = {}

        for extractor in extractors:
            merged_predictions = merge_labels(aligned_predictions, extractor)
            merged_predictions = substitute_labels(merged_predictions, "O", "no_entity")            

            entity_report = metrics.classification_report(
                merged_targets, merged_predictions, output_dict=True
            )

        return [intent_report, entity_report]

