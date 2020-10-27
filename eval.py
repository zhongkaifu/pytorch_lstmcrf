import argparse
import numpy as np
from config import Reader, Config, ContextEmb, predict_batch_insts, print_eval_results, batching_list_instances
import time
from model import NNCRF, TransformersCRF
import torch
from typing import List
from common import Instance
import os
from config.transformers_util import tokenize_instance
from config import context_models
import pickle
import tarfile


def init():

    global model
    global config

    parser = argparse.ArgumentParser(description="BERT BiLSTM CRF implementation")
    opt = parse_arguments(parser)

    model_archived_file = opt.model

    tar = tarfile.open(model_archived_file)
    tar.extractall()
    folder_name = tar.getnames()[0]
    tar.close()


    model_path = f"{folder_name}/lstm_crf.m"
    config_path = f"{folder_name}/config.conf"


    f = open(config_path, 'rb')
    config = pickle.load(f)  # variables come out in the order you put them in
    f.close()

    config.device="cpu"
    config.test_file = opt.test_file

    model = TransformersCRF(config)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()



def parse_arguments(parser):
    parser.add_argument('--device', type=str, default="cpu", choices=['cpu', 'cuda:0', 'cuda:1', 'cuda:2'], help="GPU/CPU devices")
    parser.add_argument('--model', type=str, default="output.tar.gz", help="The file path of archived model")
    parser.add_argument('--test_file', type=str, default="test.txt", help="The file path for test")

    args = parser.parse_args()
    for k in args.__dict__:
        print(k + ": " + str(args.__dict__[k]))
    return args


def predict(config: Config, model: NNCRF, test_insts: List[Instance]):

    test_insts = tokenize_instance(context_models[config.embedder_type]["tokenizer"].from_pretrained(config.embedder_type), test_insts, None)
    test_batches = batching_list_instances(config, test_insts)

    predict_model(config, model, test_batches, test_insts)
    print_eval_results(test_insts)


def predict_model(config: Config, model: NNCRF, batch_insts_ids, insts: List[Instance]):
    ## evaluation

    batch_id = 0
    batch_size = config.batch_size
    with torch.no_grad():
        for batch in batch_insts_ids:
            one_batch_insts = insts[batch_id * batch_size:(batch_id + 1) * batch_size]
            batch_max_scores, batch_max_ids = model.decode(**batch)
            predict_batch_insts(one_batch_insts, batch_max_ids, batch["word_seq_lens"], config.idx2labels)
            batch_id += 1



def main():

    init()


    reader = Reader(config.digit2zero)
    tests = reader.read_txt(config.test_file, config.test_num)

    predict(config, model, tests)




if __name__ == "__main__":
    main()
