import os
import random
import logging

import torch
import numpy as np
from seqeval.metrics import accuracy_score, precision_score, recall_score, f1_score
import sklearn.metrics as sklearn_metrics
from transformers import BertTokenizer, BertConfig, DistilBertConfig, DistilBertTokenizer, AlbertConfig, AlbertTokenizer

from model import JointBERT, JointDistilBERT
from itertools import chain

######################################## PyTorch ########################################

MODEL_CLASSES = {
    'bert': (BertConfig, JointBERT, BertTokenizer),
    'distilbert': (DistilBertConfig, JointDistilBERT, DistilBertTokenizer),
    'albert': (AlbertConfig, JointBERT, AlbertTokenizer)
}

MODEL_PATH_MAP = {
    'bert': 'bert-base-uncased',
    'distilbert': 'distilbert-base-uncased',
    'albert': 'albert-xxlarge-v1'
}

def vocab_process(args,data_type):
    data = ""
    data_path = os.path.join(args.data_dir, args.task, 'train')
    if data_type == 'intent':
        # intent
        with open(os.path.join(data_path, 'label'), 'r', encoding='utf-8') as f_r:
            intent_vocab = set()
            for line in f_r:
                line = line.strip()
                intent_vocab.add(line)

            additional_tokens = ["UNK"]
            for token in additional_tokens:
                data = data + token + '\n'

            intent_vocab = sorted(list(intent_vocab))
            for intent in intent_vocab:
                data = data + intent + '\n'

    else: 
        # slot
        with open(os.path.join(data_path, 'seq.out'), 'r', encoding='utf-8') as f_r:
            slot_vocab = set()
            for line in f_r:
                line = line.strip()
                slots = line.split()
                for slot in slots:
                    slot_vocab.add(slot)

            slot_vocab = sorted(list(slot_vocab), key=lambda x: (x[2:], x[:2]))

            # Write additional tokens
            additional_tokens = ["PAD", "UNK"]
            for token in additional_tokens:
                data = data + token + '\n'

            for slot in slot_vocab:
                data = data + slot + '\n'
    return data

def get_intent_labels(args):
    data = vocab_process(args,'intent')
    return [label.strip() for label in data.split('\n')]


def get_slot_labels(args):
    data = vocab_process(args,'slots')
    return [label.strip() for label in data.split('\n')]


def load_tokenizer(args):
    return MODEL_CLASSES[args.model_type][2].from_pretrained(args.model_name_or_path)


def init_logger():
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if not args.no_cuda and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)


def compute_metrics(intent_preds, intent_labels, slot_preds, slot_labels):
    assert len(intent_preds) == len(intent_labels) == len(slot_preds) == len(slot_labels)
    results = {}
    intent_result = get_intent_acc(intent_preds, intent_labels)
    slot_result = get_slot_metrics(slot_preds, slot_labels)
    sementic_result = get_sentence_frame_acc(intent_preds, intent_labels, slot_preds, slot_labels)

    results.update(intent_result)
    results.update(slot_result)
    results.update(sementic_result)

    return results


def get_slot_metrics(preds, labels):
    assert len(preds) == len(labels)
    return {
        "slot_accuracy": accuracy_score(labels, preds),
        "slot_precision": precision_score(labels, preds),
        "slot_recall": recall_score(labels, preds),
        "slot_f1": f1_score(labels, preds)
    }


def get_intent_acc(preds, labels):
    assert len(preds) == len(labels)
    return {
        "intent_accuracy": sklearn_metrics.accuracy_score(labels, preds),
        "intent_precision": sklearn_metrics.precision_score(labels, preds, average='weighted'),
        "intent_recall": sklearn_metrics.recall_score(labels, preds, average='weighted'),
        "intent_f1": sklearn_metrics.f1_score(labels, preds, average='weighted')
    }


def read_prediction_text(args):
    return [text.strip() for text in open(os.path.join(args.pred_dir, args.pred_input_file), 'r', encoding='utf-8')]


def get_sentence_frame_acc(intent_preds, intent_labels, slot_preds, slot_labels):
    """For the cases that intent and all the slots are correct (in one sentence)"""
    # Get the intent comparison result
    intent_result = (intent_preds == intent_labels)

    # Get the slot comparision result
    slot_result = []
    for preds, labels in zip(slot_preds, slot_labels):
        assert len(preds) == len(labels)
        one_sent_result = True
        for p, l in zip(preds, labels):
            if p != l:
                one_sent_result = False
                break
        slot_result.append(one_sent_result)
    slot_result = np.array(slot_result)

    sementic_acc = np.multiply(intent_result, slot_result).mean()
    return {
        "sementic_frame_acc": sementic_acc
    }



# -*- coding: utf-8 -*-
"""
@author: mwahdan
"""
######################################## Tensorflow ########################################


def flatten(y):
    """
    Flatten a list of lists.

    >>> flatten([[1,2], [3,4]])
    [1, 2, 3, 4]
    """
    return list(chain.from_iterable(y))


def convert_to_slots(slots_arr, no_class_tag='O', begin_prefix='B-', in_prefix='I-'):
    previous = None
    slots = []
    start = -1
    end = -1
    def add(name, s, e):
        if e < s:
            e = s
        slots.append((name, s, e))
    for i, slot in enumerate(slots_arr):
        if slot == 'O':
            current = None
            if previous != None:
                add(previous, start, end)
        if slot.startswith(begin_prefix):
            current = slot[len(begin_prefix):]
            start = i
        elif slot.startswith(in_prefix):
            current = slot[len(in_prefix):]
            if current != previous:
                # logical error, so ignore this slot
                current = None
            else:
                end = i
            
        previous = current
        
    if previous is not None:
        add(previous, start, end)
        
    return slots



if __name__ == '__main__':
    result = convert_to_slots(['O', 'B-artist', 'I-artist', 'O', 'O', 'B-playlist', 'I-playlist', 'O'])
    assert result == [('artist', 1, 2), ('playlist', 5, 6)]
    result = convert_to_slots(['O', 'B-artist', 'I-artist', 'O', 'O', 'B-playlist', 'O'])
    assert result == [('artist', 1, 2), ('playlist', 5, 5)]
    result = convert_to_slots(['O', 'B-artist', 'I-artist', 'O', 'O', 'B-playlist'])
    assert result == [('artist', 1, 2), ('playlist', 5, 5)]
    result = convert_to_slots(['O', 'B-artist', 'O', 'O', 'B-playlist', 'I-playlist', 'O'])
    assert result == [('artist', 1, 1), ('playlist', 4, 5)]
    result = convert_to_slots(['O', 'I-artist', 'I-artist', 'O', 'O', 'B-playlist', 'I-playlist', 'O'])
    assert result == [('playlist', 5, 6)]