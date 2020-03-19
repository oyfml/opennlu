from opennlu.services.tensorflow_JointBERT.readers.goo_format_reader import Reader
from opennlu.services.tensorflow_JointBERT.vectorizers.bert_vectorizer import BERTVectorizer
from opennlu.services.tensorflow_JointBERT.vectorizers.tags_vectorizer import TagsVectorizer
from opennlu.services.tensorflow_JointBERT.models.joint_bert import JointBertModel
from opennlu.services.tensorflow_JointBERT.models.joint_bert_crf import JointBertCRFModel
from opennlu.services.tensorflow_JointBERT.utils import flatten, convert_to_slots

from sklearn.preprocessing import LabelEncoder
import numpy as np
import os
import pickle
import tensorflow as tf
import json
from tensorflow.python.keras.backend import set_session


# Dict to store tensorflow models
class TF_dict(object):
    def __init__(self):
        self.tf_list = {}

    def create_new(self,name,model_type='bert',batch_size=64,learning_rate=5e-5,num_train_epochs=5,use_crf=False):
        # create new TF model
        self.tf_list[name] = TF(
            name=name, 
            model_type=model_type, 
            batch_size=batch_size, 
            learning_rate=learning_rate, 
            num_train_epochs=num_train_epochs, 
            use_crf=use_crf
        )
        return self.tf_list[name]

    def get_model(self, name):
        # retrieve TF model
        return self.tf_list[name]

    def get_all(self):
        # return list of all TF models
        return list(self.tf_list.keys())

    def size(self):
        # get number of TF models
        return len(self.tf_list)


# TF model
class TF(object):
    def __init__(self,name,model_type,batch_size,learning_rate,num_train_epochs,use_crf): #default settings
        self.batch_size = int(batch_size)
        self.learning_rate = float(learning_rate)
        self.epochs = int(num_train_epochs)
        self.use_crf = use_crf
        self.type_ = model_type

        self.save_folder_path = os.path.join(os.path.join(os.getcwd(),'opennlu/data/model/tensorflow'),name)

        tf.compat.v1.reset_default_graph()
        self.sess = tf.compat.v1.Session()
        self.graph = tf.get_default_graph()
        set_session(self.sess)
        #tf.compat.v1.random.set_random_seed(123)

    #option 1: train new model
    def train(self, train_data_dir, valid_data_dir):
        
        self.train_data_folder_path = train_data_dir
        self.val_data_folder_path = valid_data_dir 
   

        if self.type_ == 'bert':
            bert_model_hub_path = 'https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1'
            is_bert = True
        else:
            bert_model_hub_path = 'https://tfhub.dev/google/albert_base/1'
            is_bert = False

        #read data
        train_text_arr, train_tags_arr, train_intents = Reader.read(self.train_data_folder_path)
        val_text_arr, val_tags_arr, val_intents = Reader.read(self.val_data_folder_path)
        
        #vectorise data
        self.bert_vectorizer = BERTVectorizer(self.sess, is_bert, bert_model_hub_path)
        train_input_ids, train_input_mask, train_segment_ids, train_valid_positions, train_sequence_lengths = self.bert_vectorizer.transform(train_text_arr)
        val_input_ids, val_input_mask, val_segment_ids, val_valid_positions, val_sequence_lengths = self.bert_vectorizer.transform(val_text_arr)
        
        #vectorise tags
        if self.use_crf: # with crf
            self.tags_vectorizer = TagsVectorizer()
            self.tags_vectorizer.fit(train_tags_arr)
            train_tags = self.tags_vectorizer.transform(train_tags_arr, train_valid_positions)
            train_tags = tf.keras.utils.to_categorical(train_tags)
            val_tags = self.tags_vectorizer.transform(val_tags_arr, val_valid_positions)
            val_tags = tf.keras.utils.to_categorical(val_tags)
            slots_num = len(self.tags_vectorizer.label_encoder.classes_)
        else: # without crf
            self.tags_vectorizer = TagsVectorizer()
            self.tags_vectorizer.fit(train_tags_arr)
            train_tags = self.tags_vectorizer.transform(train_tags_arr, train_valid_positions)
            val_tags = self.tags_vectorizer.transform(val_tags_arr, val_valid_positions)
            slots_num = len(self.tags_vectorizer.label_encoder.classes_)
        
        #encode labels
        self.intents_label_encoder = LabelEncoder()
        train_intents = self.intents_label_encoder.fit_transform(train_intents).astype(np.int32)
        val_intents = self.intents_label_encoder.transform(val_intents).astype(np.int32)
        intents_num = len(self.intents_label_encoder.classes_)

        #train
        if self.use_crf: # with crf
            self.model = JointBertCRFModel(slots_num, intents_num, bert_model_hub_path, self.sess, num_bert_fine_tune_layers=10, is_bert=is_bert, is_crf=True, learning_rate=self.learning_rate)
            
            self.model.fit([train_input_ids, train_input_mask, train_segment_ids, train_valid_positions, train_sequence_lengths], [train_tags, train_intents],
                    validation_data=([val_input_ids, val_input_mask, val_segment_ids, val_valid_positions, val_sequence_lengths], [val_tags, val_intents]),
                    epochs=self.epochs, batch_size=self.batch_size)
        else: # without crf
            self.model = JointBertModel(slots_num, intents_num, bert_model_hub_path, self.sess, num_bert_fine_tune_layers=10, is_bert=is_bert, is_crf=False, learning_rate=self.learning_rate)

            self.model.fit([train_input_ids, train_input_mask, train_segment_ids, train_valid_positions], [train_tags, train_intents],
                validation_data=([val_input_ids, val_input_mask, val_segment_ids, val_valid_positions], [val_tags, val_intents]),
                epochs=self.epochs, batch_size=self.batch_size)
        
        #save
        if not os.path.exists(self.save_folder_path):
            os.makedirs(self.save_folder_path)
        self.model.save(self.save_folder_path)
        with open(os.path.join(self.save_folder_path, 'tags_vectorizer.pkl'), 'wb') as handle:
            pickle.dump(self.tags_vectorizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(os.path.join(self.save_folder_path, 'intents_label_encoder.pkl'), 'wb') as handle:
            pickle.dump(self.intents_label_encoder, handle, protocol=pickle.HIGHEST_PROTOCOL)


    #option 2: load existing model
    def load(self):

        load_folder_path = self.save_folder_path
        with open(os.path.join(load_folder_path, 'params.json'), 'r') as json_file:
            model_params = json.load(json_file)
            
        slots_num = model_params['slots_num'] 
        intents_num = model_params['intents_num']
        bert_model_hub_path = model_params['bert_hub_path']
        num_bert_fine_tune_layers = model_params['num_bert_fine_tune_layers']
        is_bert = model_params['is_bert']
        is_crf = model_params['is_crf']
        
        self.bert_vectorizer = BERTVectorizer(self.sess, is_bert, bert_model_hub_path)

        with open(os.path.join(load_folder_path, 'tags_vectorizer.pkl'), 'rb') as handle:
            self.tags_vectorizer = pickle.load(handle)
            slots_num = len(self.tags_vectorizer.label_encoder.classes_)
        with open(os.path.join(load_folder_path, 'intents_label_encoder.pkl'), 'rb') as handle:
            self.intents_label_encoder = pickle.load(handle)
            intents_num = len(self.intents_label_encoder.classes_)
        
        if is_crf:
            self.model = JointBertCRFModel.load(load_folder_path, self.sess)
        else:
            self.model = JointBertModel.load(load_folder_path, self.sess)


    #evaluate single input message
    def predict(self,utterance):
     with self.graph.as_default():
        set_session(self.sess)
        tokens = utterance.split()
        input_ids, input_mask, segment_ids, valid_positions, data_sequence_lengths = self.bert_vectorizer.transform([utterance])

        if self.use_crf:
            predicted_tags, predicted_intents = self.model.predict_slots_intent(
                    [input_ids, input_mask, segment_ids, valid_positions, data_sequence_lengths], 
                    self.tags_vectorizer, self.intents_label_encoder, remove_start_end=True,
                    include_intent_prob=True)
        else:
            predicted_tags, predicted_intents = self.model.predict_slots_intent(
                    [input_ids, input_mask, segment_ids, valid_positions], 
                    self.tags_vectorizer, self.intents_label_encoder, remove_start_end=True,
                    include_intent_prob=True)

        response = {
                "intent": {
                        "name": predicted_intents[0][0].strip(),
                        "confidence": predicted_intents[0][1]
                        },
                "slots": " ".join(predicted_tags[0])
                }
        return response
        
    #evaluate test data set
    def evaluate(self, test_data_dir):
        from sklearn import metrics

        with self.graph.as_default():
            set_session(self.sess)
            self.test_data_folder_path = test_data_dir
            data_text_arr, data_tags_arr, data_intents = Reader.read(self.test_data_folder_path)
            data_input_ids, data_input_mask, data_segment_ids, data_valid_positions, data_sequence_lengths = self.bert_vectorizer.transform(data_text_arr)
            
            if self.use_crf:
                predicted_tags, predicted_intents = self.model.predict_slots_intent(
                    [data_input_ids, data_input_mask, data_segment_ids, data_valid_positions, data_sequence_lengths], 
                    self.tags_vectorizer, self.intents_label_encoder, remove_start_end=True, include_intent_prob=True)
            else:
                predicted_tags, predicted_intents = self.model.predict_slots_intent(
                    [data_input_ids, data_input_mask, data_segment_ids, data_valid_positions],
                    self.tags_vectorizer, self.intents_label_encoder, remove_start_end=True, include_intent_prob=True)

            gold_tags = [x.split() for x in data_tags_arr]
            #calculate metrics for intent and slots
            slot_acc = metrics.accuracy_score(flatten(gold_tags), flatten(predicted_tags))
            slot_f1 = metrics.f1_score(flatten(gold_tags), flatten(predicted_tags), average='weighted')
            slot_precision = metrics.precision_score(flatten(gold_tags), flatten(predicted_tags), average='weighted')
            slot_recall = metrics.recall_score(flatten(gold_tags), flatten(predicted_tags), average='weighted')

            confidence = [ex[1] for ex in predicted_intents]
            predicted_intents = [ex[0] for ex in predicted_intents]

            intent_acc = metrics.accuracy_score(data_intents, predicted_intents)
            intent_f1 = metrics.f1_score(data_intents, predicted_intents, average='weighted')
            intent_precision = metrics.precision_score(data_intents, predicted_intents, average='weighted')
            intent_recall = metrics.recall_score(data_intents, predicted_intents, average='weighted')
            metrics = {
                'slot_acc':slot_acc,
                'slot_f1':slot_f1,
                'slot_precision':slot_precision,
                'slot_recall':slot_recall,
                'intent_acc':intent_acc,
                'intent_f1':intent_f1,
                'intent_precision':intent_precision,
                'intent_recall':intent_recall
            }

            predicted_tags = [ " ".join(ex)+"\n" for ex in predicted_tags]

            # for report, confusion matrix & histogram
            self.intent_true = data_intents
            self.intent_pred = predicted_intents
            self.slot_true = data_tags_arr
            self.slot_pred = predicted_tags
            self.confidence_score = [float(score) for score in confidence]

            return [metrics, predicted_intents, data_intents, predicted_tags, data_tags_arr, confidence] #confidence score

    
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
        



        
