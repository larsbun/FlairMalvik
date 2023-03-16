import os, torch, flair

from flair.data import MultiCorpus
from flair.datasets import UD_ENGLISH, UD_GERMAN
from flair.embeddings import FlairEmbeddings, StackedEmbeddings, WordEmbeddings
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer
from torch.utils.tensorboard import SummaryWriter
from importCorpus_loop import return_corpus, return_ger, return_specific
flair.device = torch.device('cuda:1')
from newScoring import F05Score

#corpus = return_specific('swedish')
corpus = return_corpus()

mc = MultiCorpus(corpus)
# 2. what label do we want to predict?
label_type = 'verdict'

# 3. make the label dictionary from the corpus
label_dict = corpus[4].make_label_dictionary(label_type=label_type,
                                             add_unk=True)

label_dict.multi_label = False
label_dict.remove_item('␤')
print(label_dict)

swc = corpus[4]

# 4. initialize embeddings
embedding_types = [
    # we use multilingual Flair embeddings in this task
    FlairEmbeddings('sv-forward'),
    FlairEmbeddings('sv-backward'),
]

embeddings = StackedEmbeddings(embeddings=embedding_types)

#embeddings = WordEmbeddings('de-crawl')
#embeddings = FlairEmbeddings('sv-v0-X')

# 5. initialize sequence tagger
tagger = SequenceTagger(hidden_size=256,
                        embeddings=embeddings,
                        rnn_layers = 10,
                        tag_dictionary=label_dict,
                        tag_type=label_type,
                        use_crf=False)

# 5.5 Initialize SummaryWriter for Tensorboard logging
writer = SummaryWriter()

#trainer = ModelTrainer(model, corpus, optimizer=optimizer, epoch_scheduler=scheduler, tensorboard_writer=writer)

# 6. initialize trainer
trainer = ModelTrainer(tagger, mc)

# 7. start training
experiment_root = 'resources/taggers/example-swedish-only-one-tag-F05'
os.makedirs(experiment_root + '/log')
  
trainer.train(experiment_root,
              main_evaluation_metric = ("macro avg", "f1-score"),
              learning_rate=0.1,
              mini_batch_size=32,
              max_epochs=150,
              #num_workers=16,
              use_tensorboard=True,
              #train_with_dev = True,
              tensorboard_log_dir = experiment_root + '/log')
