import numpy as np
from config import Reader, Config, ContextEmb, predict_batch_insts, print_predict_results, batching_list_instances
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


def predict(config: Config, model: NNCRF, test_insts: List[Instance]):

    tokenize_instance(context_models[config.embedder_type]["tokenizer"].from_pretrained(config.embedder_type), test_insts, None)
    test_batches = batching_list_instances(config, test_insts)

    predict_model(config, model, test_batches, test_insts)
    print_predict_results(test_insts)


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

    
    model_archived_file = "output.tar.gz"

    tar = tarfile.open(model_archived_file)
    tar.extractall()
    folder_name = tar.getnames()[0]
    tar.close()


    model_path = f"{folder_name}/lstm_crf.m"
    config_path = f"{folder_name}/config.conf"


    f = open(config_path, 'rb')
    config = pickle.load(f)  # variables come out in the order you put them in

    model = TransformersCRF(config)
    model.load_state_dict(torch.load(model_path))
    model.eval()



    reader = Reader(config.digit2zero)
    #tests = reader.read_txt(config.test_file, config.test_num)

    tests = reader.read_line('[BOS] [space] hello [space] world [space] [EOS]')
    predict(config, model, tests)


    tests = reader.read_line('[BOS] [space] how [space] are [space] you [space] doing [space] today [space] [EOS]')
    predict(config, model, tests)


if __name__ == "__main__":
    main()
