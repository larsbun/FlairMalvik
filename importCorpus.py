import os

from flair.data import Corpus
from flair.datasets import ColumnCorpus

###  This file adapts an example from flair to import a corpus from a vertical format to what flair expects for sequence labeling


# define columns
columns = {0: 'text', 1: 'verdict'}

# this is the folder in which train, test and dev files reside

data_folder = '/lhome/larsbun/git-projects/multiged-2023/german/'
data_root = 'de_falko-merlin_'


data_group = os.path.join(data_folder, data_root)

# init a corpus using column format, data folder and the names of the train, dev and test files
mycorpus_de: Corpus = ColumnCorpus(data_folder, columns,
                              train_file = data_group+'train.tsv',
                              test_file = data_group +'test_unlabelled.tsv',
                              dev_file = data_group +'dev.tsv')
