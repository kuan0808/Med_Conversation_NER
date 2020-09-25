import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')# 忽略警告

import logging
import os.path
import sys
import argparse
import multiprocessing

from gensim.corpora import WikiCorpus
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence


if __name__ == '__main__':

    # create parser object
    parser = argparse.ArgumentParser(description="A word2vec model training script!")

    # defining arguments for parser object
    parser.add_argument("-i", "--input", type=str, nargs=1,
                        metavar="filename", default=None,
                        help="Opens the specified text file.")

    parser.add_argument("-o", "--output", type=str, nargs=1,
                        metavar="filename", default=None,
                        help="Output file name")

    parser.add_argument("-d", "--dim", type=int, nargs=1,
                        metavar="dimension", default=None,
                        help="word vector dimension")

    # parse the arguments from standard input
    args = parser.parse_args()



    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)

    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s',level=logging.INFO)
    logger.info("running %s" % ' '.join(sys.argv))

    dimension = args.dim[0]
    inp = args.input[0]
    print(args.input)
    if args.output != None:
        outp1 = args.output[0]+'.model'
        outp2 = args.output[0]+'.txt'
    else:
        outp1 = args.input[0] + '.model'
        outp2 = args.input[0] + '.txt'

    model = Word2Vec(LineSentence(inp), size=dimension, window=5, min_count=5,
                     workers=multiprocessing.cpu_count())

    model.save(outp1)
    model.wv.save_word2vec_format(outp2, binary=False)


