import os

USE_DROPOUT = os.environ.get('METLRNN_USE_DROPOUT', True)
DROPOUT = os.environ.get('METLRNN_DROPOUT', 0.2)

BATCH_SIZE = os.environ.get('METLRNN_BATCH_SIZE', 1028)
