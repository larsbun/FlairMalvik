import os

from flair.data import Corpus, Sentence
from flair.datasets import ColumnCorpus

###  This file adapts an example from flair to import a corpus from a vertical format to what flair expects for sequence labeling
###  It is rewritten to provide a function, which returns the corpus list for import (to function as a module)

homedir = os.path.expanduser('~')

def return_corpus():
    # define columns
    columns = {0: 'text', 1: 'verdict'}

    # this is the folder in which train, test and dev files reside
    folder_root = homedir + '/git-projects/multiged-2023/'

    training_tuples = [['czech/', 'cs_geccc_'],
                       ['english/', 'en_fce_'],
    #                   ['english/', 'en_realec_']
                       ['german/','de_falko-merlin_'],
                       ['italian/', 'it_merlin_'],
                       ['swedish/', 'sv_swell_']]

    #training_tuples = ['german/', 'de_falko-merlin_']
    corpus = list([x for x in range(len(training_tuples))])

    for i, tup in enumerate(training_tuples):
        print(i,tup)
        langdir, tset = tup
        data_folder = os.path.join(folder_root) + langdir
        data_group = os.path.join(data_folder) + tset
        print(i, data_group)
        corpus[i]: Corpus = ColumnCorpus(data_folder, columns,
                                  train_file = data_group + 'train.tsv',
                                  test_file = data_group +'test_unlabelled.tsv',
                                  dev_file = data_group +'dev.tsv')

    return corpus

def return_ger():
    columns = {0: 'text', 1: 'verdict'}
    data_folder = homedir + '/git-projects/multiged-2023/german/'
    data_group = data_folder + 'de_falko-merlin_'
    corpus: Corpus = ColumnCorpus(data_folder, columns,
                                     train_file=data_group + 'train.tsv',
                                     test_file=data_group + 'test_unlabelled.tsv',
                                     dev_file=data_group + 'dev.tsv')

    return corpus

def return_swfoo():
    columns = {0: 'text', 1: 'verdict'}
    data_folder = '~/git-projects/multiged-2023/swedish/foo/'
    data_group = data_folder + 'sv_swell_'
    corpus: Corpus = ColumnCorpus(data_folder, columns,
                                     train_file=data_group + 'train.tsv-noc.tsv',
                                     test_file=data_group + 'test_unlabelled.tsv-noc.tsv',
                                     dev_file=data_group + 'dev.tsv-noc.tsv')

    return corpus


def return_specific(language):
    columns = {0: 'text', 1: 'verdict'}
    training_tuples = [['czech/', 'cs_geccc_'],
                       ['english/', 'en_fce_'],
    #                   ['english/', 'en_realec_']
                       ['german/','de_falko-merlin_'],
                       ['italian/', 'it_merlin_'],
                       ['swedish/', 'sv_swell_']]

    print(language)
    index = [index for (index, item) in enumerate(training_tuples) if item[0].strip('/') == language][0]

    lang, pref = training_tuples[index]

    data_folder = '~/git-projects/multiged-2023/' + lang
    data_group = data_folder + pref
    corpus: Corpus = ColumnCorpus(data_folder, columns,
                                     train_file=data_group + 'train.tsv',
                                     test_file=data_group + 'test_unlabelled.tsv',
                                     dev_file=data_group + 'dev.tsv')

    return corpus