import sys
from multiprocessing import cpu_count

from gensim.models import Word2Vec, FastText, Doc2Vec, Phrases
from gensim.models.phrases import Phraser
from sklearn import utils
from fse.models import SIF
import logging

from code.utils import get_path
from code.preprocessing import WordCorpus, DocumentCorpus, PhraseCorpus

logging.basicConfig(
    format='%(asctime)s : %(name)s : %(levelname)s : %(message)s', stream=sys.stdout, level=logging.INFO
)

DATAPATH = "/Users/Olamilekan/Desktop/Machine Learning/OpenSource/yoembedding"


class Embeddings(object):
    def __init__(self, **kwargs):
        self.model = kwargs.get('model')
        self.cbow = kwargs.get('cbow', 0)
        self.dbow = kwargs.get('dbow', 0)
        self.hs = kwargs.get('hs', 0)
        self.epoch = kwargs.get('epoch', 5)
        self.deaccent = kwargs.get('deaccent')
        self.dim = kwargs.get('dim', 300)
        self.data = None
        self.total_examples = None
        self.params = {
            'size': 300,
            'window': 10,
            'workers': max(1, cpu_count() - 1),
            'sample': 1E-5,
            'sg': self.cbow,
            'iter': self.epoch,
            'hs': self.hs
        }

    def setup(self):
        """
        Preprocess and get data ready
        :return:
        """
        if self.model in ["word2vec", "phrase2vec", 'sentence2vec', 'fasttext', 'glove']:
            words = WordCorpus(DATAPATH)
        else:
            words = DocumentCorpus(DATAPATH)

        self.data = list(words.get_texts())
        self.total_examples = len(self.data)

        print("Total Examples:  %s" % self.total_examples)

    def word2vec(self, save=True):
        model = Word2Vec(min_count=1, **self.params)
        logging.info("Building Model")
        model.build_vocab(sentences=self.data)
        model.train(sentences=self.data, total_examples=self.total_examples, epochs=self.epoch)
        logging.info("Training complete. Saving model")
        if save:
            model_name = f"word2vec_{'cbow' if not self.cbow else 'skipgram'}_{self.dim}.vec"
            model_path = get_path(f'/models/word2vec/{model_name}')
            return model.save(model_path)
        else:
            return model

    def fasttext(self, save=True):
        model = FastText(min_count=1, **self.params)
        logging.info("Building Model")
        model.build_vocab(self.data)
        model.train(utils.shuffle(self.data), total_examples=self.total_examples, epochs=self.epoch)
        logging.info("Training complete. Saving model")
        if save:
            model_name = f"fasttext_{'cbow' if not self.cbow else 'skipgram'}_{self.dim}.vec"
            model_path = get_path(f'/models/fastext/{model_name}')
            return model.save(model_path)
        else:
            return model

    def doc2vec(self):
        model = Doc2Vec(dm=self.dbow, vector_size=self.dim, min_count=3, window=10,
                        hs=self.hs, iter=self.epoch, workers=max(1, cpu_count() - 1))
        logging.info("Building Model")
        model.build_vocab(self.data)
        model.train(utils.shuffle(self.data), total_examples=self.total_examples, epochs=self.epoch)
        logging.info("Training complete. Saving model")
        model_name = f"doc2vec_{'dbow' if not self.dbow else 'dm'}_{self.dim}.vec"
        model_path = get_path(f'/models/doc2vec/{model_name}')
        return model.save(model_path)

    def phrase2vec(self):
        phrases = Phrases(self.data,
                          min_count=5,
                          threshold=7,
                          progress_per=1000)
        phrase_model = Phraser(phrases)
        training_data = [phrase_model[sentence] for sentence in self.data]

        model = Word2Vec(min_count=1, **self.params)
        logging.info("Building Model")
        model.build_vocab(sentences=training_data)
        model.train(sentences=training_data, total_examples=len(training_data), epochs=self.epoch)
        logging.info("Training complete. Saving model")
        model_name = f"phrase2vec_{'cbow' if not self.cbow else 'skipgram'}_{self.dim}.vec"
        model_path = get_path(f'/models/word2vec/{model_name}')
        return model.save(model_path)

    def glove(self, save=True):
        pass

    def sentence2vec(self, parentmodel=None, save=True):
        model_map = {"word2vec": self.word2vec(save=False), "glove": self.glove(save=False)}
        model = model_map.get(parentmodel, self.fasttext(save=False))
        sentence_model = SIF(model)
        sentence_model = sentence_model.train(self.data)
        logging.info("Training complete. Saving model")
        if save:
            model_name = f"sentence2vec_{'cbow' if not self.cbow else 'skipgram'}_{self.dim}.vec"
            model_path = get_path(f'/models/sentence2vec/{model_name}')
            return sentence_model.save(model_path)
        else:
            return sentence_model
