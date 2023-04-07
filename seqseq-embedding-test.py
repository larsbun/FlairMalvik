# -*- coding: utf-8 -*-

import os, torch, flair, pickle

from flair.data import MultiCorpus
from flair.datasets import UD_ENGLISH, UD_GERMAN
from flair.embeddings import FlairEmbeddings, StackedEmbeddings, WordEmbeddings, TransformerWordEmbeddings, TokenEmbeddings
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer
from torch.utils.tensorboard import SummaryWriter
from importCorpus_loop import return_corpus, return_ger, return_specific
#flair.device = torch.device('cuda:1')
from newScoring import F05Score

#flair.cache_root = '/cluster/work/users/larsbun/.cache/flair'
print(os.getenv('HF_DATASETS_CACHE'))

#corpus = return_specific('swedish')
corpus = return_corpus()
csc = corpus[0]
en_fce_c = corpus[1]
dec = corpus[2]
itc = corpus[3]
swc = corpus[4]

mc = MultiCorpus(corpus)
# 2. what label do we want to predict?
label_type = 'verdict'

# 3. make the label dictionary from the corpus
label_dict = swc.make_label_dictionary(label_type=label_type,
                                             add_unk=True)

label_dict.multi_label = False
label_dict.remove_item('␤')
print(label_dict)



#4. initialize embeddings
# embedding_types = [
#     # we use multilingual Flair embeddings in this task
#     FlairEmbeddings('multi-forward'),
#     FlairEmbeddings('multi-backward'),
# ]

# embedding_types = [
#    FlairEmbeddings('news-forward'),
#    FlairEmbeddings('news-backward'),
#     TransformerWordEmbeddings('distilbert-base-multilingual-cased',
#                               pooling_operation='first_last',
#                               fine_tune=True,
#                               batch_size=8,
#                               layers='-1',
#                               use_scalar_mix=True,
#                               allow_long_sentences=True)

# ]

embeddings = TransformerWordEmbeddings(
    model='distilbert-base-uncased',
    layers='-1',
    subtoken_pooling='first',
    fine_tune=True,
    use_context=False)

#embeddings = StackedEmbeddings(embeddings=embedding_types)

#embeddings = TransformerWordEmbeddings('bert-base-multilingual-cased')
# embeddings =  TransformerWordEmbeddings('distilbert-base-multilingual-cased',
#                                         pooling_operation='first_last',
#                                         fine_tune=True,
#                                         batch_size=8,
#                                         layers='-1',
#                                         use_scalar_mix=False,
#                                         allow_long_sentences=True,
#                                         model_max_length=512)
# 

# embeddings = TransformerWordEmbeddings('distilbert-base-multilingual-cased',
#                                        layers="-1",
#                                        subtoken_pooling="first",
#                                        fine_tune=True,
#                                        use_context=True,
#                                        allow_long_sentences=True,
#                                        model_max_length=512
#                                        )

#embeddings = WordEmbeddings('de-crawl')
#embeddings = FlairEmbeddings('sv-v0-X')

# embedding_types = [
#     # we use multilingual Flair embeddings in this task
#     FlairEmbeddings('cs-forward'),
#     FlairEmbeddings('cs-backward'),
# ]
# embeddings = StackedEmbeddings(embeddings=embedding_types)

invalid_sentences = []
for lang in corpus:
    print.lang
    for sentence in mc.get_all_sentences():
        try:
            embeddings.embed(sentence)
        except:
            invalid_sentences.append(sentence)
    print("There are", len(invalid_sentences), "invalid sentences")
    print(invalid_sentences[0])

## SETUP

RUNCORP = "csc"
RNNTYPE = "LSTM"
HIDDENSIZE = 256
RNNLAYERS = 10

# 5. initialize sequence tagger
tagger = SequenceTagger(hidden_size=HIDDENSIZE,
                        embeddings=embeddings,
                        rnn_type = RNNTYPE,
                        rnn_layers = RNNLAYERS,
                        tag_dictionary=label_dict,
                        tag_type=label_type,
                        use_crf=False)

# 5.5 Initialize SummaryWriter for Tensorboard logging
writer = SummaryWriter()

#trainer = ModelTrainer(model, corpus, optimizer=optimizer, epoch_scheduler=scheduler, tensorboard_writer=writer)

# 6. initialize trainer
trainer = ModelTrainer(tagger, csc)

# 7. start training
experiment_root = '/cluster/work/users/larsbun/resources/taggers/example-flair-'+ RUNCORP + '-flair-single' + '-f1-mic-' + RNNTYPE + '-' + str(RNNLAYERS) + '-layers-' + str(HIDDENSIZE) + '-hidden'

os.makedirs(experiment_root + '/log')

trainer.train(experiment_root,
              main_evaluation_metric = ("micro avg", "f1-score"),
              learning_rate=0.1,
              mini_batch_size=32,
              max_epochs=150,
              #num_workers=16,
              use_tensorboard=True,
              #train_with_dev = True,
              tensorboard_log_dir = experiment_root + '/log')
