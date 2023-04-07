from flair.models import SequenceTagger
from importCorpus_loop import return_corpus
from flair.data import Sentence
from flair.data import Corpus
from flair.datasets import ColumnCorpus
from flair.trainers import ModelTrainer
# define the path to the corpus TSV file
data_folder = '/lhome/larsbun/git-projects/multiged-2023/english/'
column_format = {0: 'text', 1: 'verdict'}

# create a column corpus from the TSV file
#mynewcorpora = [ColumnCorpus('/lhome/larsbun/git-projects/multiged-2023/english/realec', column_format)]

mynewcorpora = return_corpus()
# load a previously built model
BASEDIR = '/lhome/larsbun/tmp/pycharm_project_305/resources/taggers/'
tagger = SequenceTagger.load(BASEDIR + 'example-mc-grammar-no-downsample-all-five-dev-explicit-gpu' + '/best-model.pt')

# train with your new corpus
# trainer: ModelTrainer = ModelTrainer(tagger, mynewcorpus)

# 7. start training
# trainer.train('resources/taggers/continued_model-swe-roberta',
#               learning_rate=0.01,
#               mini_batch_size=32,
#               train_with_dev=True,
#               max_epochs=150)

# got through each sentence
for corp in mynewcorpora:
    country = corp.name.split('/')[-2]
    print(country)
    print(corp.test[0])
    tagger.predict(corp.test)

    with open(country + '_test.tsv', 'w') as country_object:

        for sentence in corp.test:

            # go through each token of sentence
            for token in sentence:
                # print what you need (text and NER value)
                country_object.write(f"{token.text}\t{token.tag}\n")

            # print newline at end of each sentence
            country_object.write('\n')

