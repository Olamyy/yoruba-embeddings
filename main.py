import click
from code.embeddings.embeddings import Embeddings
from code.preprocessing import make_glove


@click.command()
@click.argument('action', type=click.Choice(['preprocess', 'train', 'evaluate', 'similarity']))
@click.option('--deaccent/--nodeaccent', default=False, help="Remove accents during preprocessing.")
@click.option('--cbow/--skipgram', default=True, help='Skipgram or cbow for word2vec. Defaults to CBOW')
@click.option('--hs/--ns', default=True, help='Skipgram or cbow for word2vec. Defaults to CBOW')
@click.option('--batch', type=click.INT, default=100, help='Batch for trainings. Default is 100.')
@click.option('--epoch', type=click.INT, default=5, help='Epoch for training model. Default is 5.')
@click.option('--dim', type=click.INT, default=300, help='Embedding dimension')
@click.option('--model', type=click.Choice(['doc2vec', 'word2vec', 'phrase2vec', 'fasttext',  'glove', 'sentence2vec']), help='Algorithm to train data on.')
@click.option('--parentmodel', type=click.Choice(['word2vec', 'fasttext', 'glove']), help='')
def main(action, deaccent, batch, epoch, model,  cbow=True, hs=True, dim=300, parentmodel='fasttext'):
    if action == "preprocess":
        make_glove(deaccent)
    if action == "train":
        embedding_model = Embeddings(cbow=cbow, batch=batch, hs=hs, epoch=epoch, deaccent=deaccent, dim=dim, model=model)
        embedding_model.setup()
        if model == "word2vec":
            return embedding_model.word2vec()
        elif model == "fasttext":
            return embedding_model.fasttext()
        elif model == "sentence2vec":
            return model.sentence2vec(parentmodel=parentmodel)
        elif embedding_model == "glove":
            return embedding_model.glove()
        elif model == "doc2vec":
            embedding_model.doc2vec()
        elif model == "phrase2vec":
            embedding_model.phrase2vec()


if __name__ == "__main__":
    main()
