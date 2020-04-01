import argparse
from trainer import Trainer
from utils import init_logger, load_tokenizer, read_prediction_text, MODEL_PATH_MAP, MODEL_CLASSES
from data_loader import load_and_cache_examples
import os


# Dict to store py-torch models
class PT_dict(object):
    def __init__(self):
        self.pt_list = {}

    def create_new(self, name, model_type='bert', batch_size=16, learning_rate=5e-5, num_train_epochs=10.0, max_seq_len=50, use_crf=False):
        # create new PT model
        self.pt_list[name] = PT(
            name=name, 
            model_type=model_type, 
            batch_size=batch_size, 
            learning_rate=learning_rate, 
            num_train_epochs=num_train_epochs, 
            max_seq_len=max_seq_len, 
            use_crf=use_crf
        )
        return self.pt_list[name]

    def get_model(self, name):
        # retrieve PT model
        return self.pt_list[name]

    def get_all(self):
        # return list of all PT models
        return list(self.pt_list.keys())

    def size(self):
        # get number of PT models
        return len(self.pt_list)

# PT model
class PT(object):
    def __init__(self, name, model_type, batch_size, learning_rate, num_train_epochs, max_seq_len, use_crf):
        parser = argparse.ArgumentParser()

        parser.add_argument("--intent_label_file", default="intent_label.txt", type=str, help="Intent Label file")
        parser.add_argument("--slot_label_file", default="slot_label.txt", type=str, help="Slot Label file")

        parser.add_argument('--seed', type=int, default=1234, help="random seed for initialization")
        parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
        parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help="Number of updates steps to accumulate before performing a backward/update pass.")
        parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
        parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
        parser.add_argument("--max_steps", default=-1, type=int, help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
        parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
        parser.add_argument("--dropout_rate", default=0.1, type=float, help="Dropout for fully-connected layers")

        parser.add_argument('--logging_steps', type=int, default=200, help="Log every X updates steps.")
        parser.add_argument('--save_steps', type=int, default=200, help="Save checkpoint every X updates steps.")

        parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")

        parser.add_argument("--ignore_index", default=-100, type=int, help='Specifies a target value that is ignored and does not contribute to the input gradient')

        parser.add_argument('--slot_loss_coef', type=float, default=1.0, help='Coefficient for the slot loss.')

        # CRF option
        parser.add_argument("--slot_pad_label", default="PAD", type=str, help="Pad token for slot label pad (to be ignore when calculate loss)")

        self.args = parser.parse_args()

        # Arguments controllable by interface
        self.args.task = name #The name of the task to train (also data foldername of task model)
        self.args.data_dir = os.path.join(os.getcwd(),'cached_data') #input data directory
        self.args.model_dir = os.path.join(os.path.join(os.getcwd(),'opennlu/data/model/pytorch'),name) #Path to save, load model
        self.args.model_type = model_type #default: bert
        self.args.batch_size = int(batch_size) #Batch size for training and evaluation
        self.args.learning_rate = float(learning_rate) #initial learning rate for Adam
        self.args.num_train_epochs = float(num_train_epochs) #number of training epochs
        self.args.max_seq_len = int(max_seq_len) #maximum total input sequence length after tokenization
        self.args.use_crf = use_crf #option to use crf or not
        self.args.do_pred = False
        self.args.model_name_or_path = MODEL_PATH_MAP[self.args.model_type]

        init_logger()
        self.tokenizer = load_tokenizer(self.args)

    #option 1: train new model
    def train(self):
        self.train_dataset = load_and_cache_examples(self.args, self.tokenizer, mode="train")
        self.valid_dataset = load_and_cache_examples(self.args,self. tokenizer, mode="dev")
        self.trainer = Trainer(args=self.args, train_dataset=self.train_dataset, valid_dataset=self.valid_dataset)
        self.trainer.train()
        self.trainer.load_model()

    #option 2: load existing model
    def load(self):
        self.trainer = Trainer(args=self.args)
        self.trainer.load_model()

    #evaluate single input message
    def predict(self,input_text):
        self.args.do_pred = True
        text = [input_text]
        [intent_result, slot_result, confidence_score] = self.trainer.predict(text, self.tokenizer) #needs edit in predict function
        return [intent_result, slot_result, confidence_score]

    #evaluate test data set
    def evaluate(self):
        self.test_dataset = load_and_cache_examples(self.args, self.tokenizer, mode="test")
        self.trainer.test_dataset = self.test_dataset
        [self.metrics, self.intent_pred, self.intent_true, self.slot_pred, self.slot_true,  self.confidence_score] = self.trainer.evaluate("test") #get intent & slots f1, precision, recall 
        return [self.metrics, self.intent_pred, self.intent_true, self.slot_pred, self.slot_true, self.confidence_score]

    # get individual intent and entity type report; only to be executed after evaluate_metrics()
    def evaluation_get_individual_report(self):
        from sklearn import metrics
        intent_report = metrics.classification_report(
            self.intent_true, self.intent_pred, output_dict=True
        )
        slot_report = metrics.classification_report(
            self.slot_true, self.slot_pred, output_dict=True
        )
        return [intent_report, slot_report]     

    # creates confusion matrix after evaluation, with intent results
    def compute_confusion_matrix(self,save_path):
        from sklearn.metrics import confusion_matrix
        from sklearn.utils.multiclass import unique_labels
        import matplotlib.pyplot as plt
        from rasa.nlu.test import plot_confusion_matrix

        plt.gcf().clear()
        # compute confusion matrix
        cnf_matrix = confusion_matrix(self.intent_true, self.intent_pred)
        # get list of unique labels from target and predicted
        labels = unique_labels(self.intent_true, self.intent_pred)
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
            score for true, pred, score in zip(self.intent_true, self.intent_pred, self.confidence_score) if true == pred
        ]
        # miss histogram
        neg_hist = [
            score for true, pred, score in zip(self.intent_true, self.intent_pred, self.confidence_score) if true != pred
        ]
        plot_histogram([pos_hist, neg_hist],save_path)
        