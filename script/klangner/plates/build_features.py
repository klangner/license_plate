import os.path
import time
import numpy as np
import pandas as pd
import tensorflow as tf


IMAGE_PATH='../../../data/classifier/'
MODEL_DIR = '../../../model/inception3/'


class TensorFlowGraph(object):

    def __init__(self):
        with tf.gfile.FastGFile(MODEL_DIR + 'classify_image_graph_def.pb', 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            _ = tf.import_graph_def(graph_def, name='')
        # self._graph = tf.get_default_graph()
        self._session = tf.Session()

    def parse_image(self, image_path):
        image_data = tf.gfile.FastGFile(image_path, 'rb').read()
        pool_tensor = self._session.graph.get_tensor_by_name('pool_3:0')
        features = self._session.run(pool_tensor, {'DecodeJpeg/contents:0': image_data})
        return np.resize(features, features.size)


if __name__ == "__main__":
    # Prepare inception graph
    graph = TensorFlowGraph()
    # Load all images from folder
    start_index = 0
    batch_size = 5000
    end_index = start_index+batch_size
    start_time = time.time()
    folder = IMAGE_PATH + 'positives/'
    features = [graph.parse_image(folder + fp) for fp in os.listdir(folder)[start_index:end_index]]
    df_positives = pd.DataFrame(features)
    df_positives['label'] = 1
    folder = IMAGE_PATH + 'negatives/'
    features = [graph.parse_image(folder + fp) for fp in os.listdir(folder)[start_index:end_index]]
    df_negatives = pd.DataFrame(features)
    df_negatives['label'] = 0
    df = df_positives.append(df_negatives)
    print(df.head())
    total_time = time.time() - start_time
    print('Parsed {} images in {} seconds'.format(len(df), total_time))
    df.to_csv('../../../data/classifier/train-{}.csv'.format(start_index))
