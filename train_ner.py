from flair.embeddings import *
from typing import List
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

glove_pnf_pnb : List[TokenEmbeddings] = [
          WordEmbeddings('glove'),
          PooledFlairEmbeddings('news-forward'),
          PooledFlairEmbeddings('news-backward')
        ]

glove_nf_nb : List[TokenEmbeddings] = [
          WordEmbeddings('glove'),
          FlairEmbeddings('news-forward'),
          FlairEmbeddings('news-backward')
        ]

elmo : List[TokenEmbeddings] = [
          ELMoEmbeddings()
        ]

roberta : List[TokenEmbeddings] = [
          TransformerWordEmbeddings('roberta-base')
        ]

glove : List[TokenEmbeddings] = [
          WordEmbeddings('glove')
        ]

glove_char : List[TokenEmbeddings] = [
          WordEmbeddings('glove'),
          CharacterEmbeddings()
      ]

bert : List[TokenEmbeddings] = [
          TransformerWordEmbeddings('bert-base-uncased')
        ]

# bert_ner : List[TokenEmbeddings] = [
#           TransformerWordEmbeddings('dslim/bert-base-NER')
#         ]


from flair.data import Corpus
from flair.datasets import ColumnCorpus
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer

data_folder = '/directory_to_dataset' # initializing the corpus

def load_corpus(train_filename, dev_filename, test_filename):
  # define columns
  columns = {0 : 'text', 1 : 'ner'} # directory where the data resides
  
  # encoding can also be cp1252
  corpus: Corpus = ColumnCorpus(data_folder, 
                                columns,
                                train_file = train_filename,
                                test_file = test_filename,
                                dev_file = dev_filename,
                                encoding = 'utf-8')
  
  return corpus

corpus_aug = load_corpus('conll_train_aug_mr_v2_clean.txt','conll_test_v2_clean.txt','conll_dev_v2_clean.txt')
# corpus_base = load_corpus('conll_train_v2_clean.txt','conll_test_v2_clean.txt','conll_dev_v2_clean.txt')

tag_type = 'ner'

corpus_aug.obtain_statistics(label_type=tag_type, pretty_print=False)

# make tag dictionary from the corpus
tag_dictionary = corpus_aug.make_tag_dictionary(tag_type=tag_type)

folder_embedding = {
# 'augmented-mr-glove-char' : glove_char,
# 'augmented-mr-glove-pnf-pnb' : glove_pnf_pnb,
 'augmented-mr-glove-nf-nb' : glove_nf_nb,
 'augmented-mr-elmo' : elmo,
 'augmented-mr-glove' : glove,
# #  'augmented-fasttext' : fasttext,
 'augmented-mr-bert' : bert,
# #   'augmented-glove-bert' : glove_bert,
# #   'augmented-glove-roberta' : glove_roberta,
 'augmented-mr-roberta' : roberta,
# #   'augmented-xlnet' : xlnet,
#   'augmented-bert-ner' : bert_ner,
#   # 'augmented-bert-uncased-ne' : bert_ner_uncased,
#   # 'augmented-bert-large-ner' : bert_ner_large,
  
}

for folder, embedding_types in folder_embedding.items():
  print(folder)
  
  embeddings : StackedEmbeddings = StackedEmbeddings(
                                   embeddings=embedding_types)

  tagger : SequenceTagger = SequenceTagger(hidden_size=256,
                                         embeddings=embeddings,
                                         tag_dictionary=tag_dictionary,
                                         tag_type=tag_type,
                                         use_crf=True)
  print(tagger)

  trainer : ModelTrainer = ModelTrainer(tagger, corpus_aug)

  trainer.train(data_folder+folder,
                learning_rate=0.1,
                mini_batch_size=32,
                checkpoint=True,
                max_epochs=200)
