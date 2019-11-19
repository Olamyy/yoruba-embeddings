import pathlib
import sys
import logging

from gensim.models.doc2vec import TaggedDocument
from gensim.corpora import TextDirectoryCorpus, WikiCorpus, Dictionary
from gensim.utils import simple_preprocess

from code.utils import get_path

logging.basicConfig(
    format='%(asctime)s : %(name)s : %(levelname)s : %(message)s', stream=sys.stdout, level=logging.INFO
)

DATAPATH = "/Users/Olamilekan/Desktop/Machine Learning/OpenSource/yoembedding"

STOP_WORDS = ['wa', 're', 'pupo', 'naa', 'emo', 'ti', 'yii', 'pelu', 'gbogbo', 'ki', 'sugbon', 'kan', 'e', 'lo', 'nitori', 'je', 'nla', 'ba', 'ati',
              'nigba', 'oun', 'mi', 'maa', 'ko', 'ni', 'fun', 'bi', 'pe', 'ojo', 'fe', 'a', 'sinu', 'opolopo', 'ju', 'se', 'si', 'pada', 'won', 'nnkan', 'bere', 'an', 'awon', 'inu', 'mo', 'lati']

data_dir = pathlib.Path(DATAPATH)
all_files = data_dir.iterdir()


class CustomTextDirectoryCorpus(TextDirectoryCorpus):

    @staticmethod
    def save_corpus(fname, corpus, id2word=None, metadata=False):
        pass

    def getstream(self):
        """Generate documents from the underlying plain text collection (of one or more files).

        Yields
        ------
        str
            One document (if lines_are_documents - True), otherwise - each file is one document.

        """
        num_texts = 0
        for path in self.iter_filepaths():
            with open(path, 'rt') as f:
                if self.lines_are_documents:
                    for line in f:
                        yield line.strip(), path.split('/')[-1]
                        num_texts += 1
                else:
                    try:
                        if path.endswith('.DS_Store'):
                            pass
                        else:
                            yield f.read().strip(), path.split('/')[-1]
                            num_texts += 1
                    except UnicodeError:
                        print(path)
                        pass

        self.length = num_texts


class WordCorpus(TextDirectoryCorpus):
    @staticmethod
    def save_corpus(fname, corpus, id2word=None, metadata=False):
        pass

    def get_texts(self, deaccent=True):
        for doc in self.getstream():
            yield [word for word in simple_preprocess(doc, deacc=deaccent, max_len=30) if word not in STOP_WORDS]


class PhraseCorpus(TextDirectoryCorpus):

    @staticmethod
    def save_corpus(fname, corpus, id2word=None, metadata=False):
        pass

    def get_texts(self, deaccent=True):
        for doc in self.getstream():
            docs = [word for word in simple_preprocess(doc, deacc=deaccent, max_len=30) if word not in STOP_WORDS]
            chunks = list(get_phrases(docs, 5))
            for chunk in chunks:
                yield chunk


class DocumentCorpus(CustomTextDirectoryCorpus):
    @staticmethod
    def save_corpus(fname, corpus, id2word=None, metadata=False):
        pass

    def get_texts(self, deaccent=True):
        for doc, path in self.getstream():
            yield TaggedDocument(simple_preprocess(doc, deacc=deaccent, max_len=30), [path])


class EntityCorpus(TextDirectoryCorpus):
    @staticmethod
    def save_corpus(fname, corpus, id2word=None, metadata=False):
        pass


class TopicCorpus(TextDirectoryCorpus):
    @staticmethod
    def save_corpus(fname, corpus, id2word=None, metadata=False):
        pass


def flatten_data(data):
    return [y for x in data for y in x]


def get_phrases(data, count):
    for i in range(0, len(data), count):
        yield data[i:i + count]


def make_glove(deaccent):
    words = WordCorpus(DATAPATH)
    texts = list(words.get_texts(deaccent=deaccent))
    flattened = flatten_data(texts)
    with open(get_path(f"/data/glove_data.txt"), "w+") as fi:
            for entry in flattened:
                if type(entry) != list:
                    fi.write("%s\n" % entry)
                else:
                    fi.write("\n".join(entry))
    return True
