from absl import logging
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Embedding, Dropout, Input
from tensorflow.keras import activations
from tensorflow.keras.regularizers import l2

import collections

from layer.fm import FM
from layer.dnn import DNN
from layer.linear import Linear
from utils.input import input_layer
from tensorflow.python.keras.layers import Layer
from layer.sequence import AttentionSequencePoolingLayer


def _check_fm_columns(feature_columns):
    if isinstance(feature_columns, collections.Iterator):
        feature_columns = list(feature_columns)
    column_num = len(feature_columns)
    if column_num < 2:
        raise ValueError('feature_columns must have as least two elements.')
    dimension = -1
    for column in feature_columns:
        if dimension != -1 and column.dimension != dimension:
            raise ValueError('fm_feature_columns must have the same dimension.')
        dimension = column.dimension
    return column_num, dimension


'''
用组合将模型重新封装,使用函数式 API, 因为model类对输入不灵活，比如使用特征工程时想将Input作为输入
'''


class DINWrapper:
    def __init__(self,
                 feature_spec,
                 wide_columns,  # wide部分输入
                 deep_columns,  # deep部分输入
                 user_columns,  # din 用户属性
                 user_hist,  # din 用户行为历史
                 item_columns,  # din item属性
                 context_columns,  # din context属性
                 linear_model: Layer = None,
                 dnn_model: Layer = None,
                 sequence_model: Layer = None,
                 activation=None,
                 seed=1024
                 ):
        ''''''
        self.feature_spec = feature_spec
        self.wide_columns = wide_columns
        self.deep_columns = deep_columns
        self.user_columns = user_columns
        self.user_hist = user_hist
        self.item_columns = item_columns
        self.context_columns = context_columns
        self.linear_model = linear_model
        self.dnn_model = dnn_model
        self.sequence_model = sequence_model
        self.activation = activations.get(activation)
        self.seed = seed

    '''
    前向传播过程处理，输出outputs
    '''

    def _forward(self, inputs, linear_model, dnn_model, sequence_model, training=None):
        user_hist_inputs, item_feature_inputs, user_feature_inputs, context_feature_inputs, linear_inputs = inputs

        linear_output = linear_model(linear_inputs)

        query_embedding = tf.expand_dims(item_feature_inputs, axis=1)  # (-1, 1, item_field_num * emb_size)
        key_embeddings = user_hist_inputs  # (-1, sequence_length, item_field_num * emb_size)
        hist = sequence_model([query_embedding, key_embeddings])  # (-1, 1, embedding_size)

        dnn_inputs = tf.keras.layers.concatenate([user_feature_inputs, context_feature_inputs, tf.squeeze(hist)])
        dnn_inputs = tf.keras.layers.Flatten()(dnn_inputs)

        # pylint: disable=protected-access
        if dnn_model._expects_training_arg:
            if training is None:
                training = tf.keras.backend.learning_phase()
            dnn_output = dnn_model(dnn_inputs, training=training)
        else:
            dnn_output = dnn_model(dnn_inputs)

        output = tf.nest.map_structure(
            lambda x, y, z: (x + y + z), linear_output, dnn_output)  # dnn输出维度可能不为1，使用
        if self.activation:
            return tf.nest.map_structure(self.activation, output)
        return output

    '''
    功能：拿到input、output，创建模型、编译、打印模型信息
    '''

    def build_model(self):
        inputs = input_layer(self.feature_spec)
        dnn_inputs = tf.keras.layers.DenseFeatures(self.deep_columns)(inputs)
        linear_inputs = tf.keras.layers.DenseFeatures(self.wide_columns)(inputs)

        user_feature_inputs = tf.keras.layers.DenseFeatures(self.user_columns)(
            inputs)  # (-1, user_field_num * emb_size)
        user_hist_inputs, sequence_length = tf.keras.experimental.SequenceFeatures(self.user_hist)(
            inputs)  # (-1, sequence_length, item_field_num * emb_size)
        user_hist_inputs = tf.sequence_mask(sequence_length)
        item_feature_inputs = tf.keras.layers.DenseFeatures(self.item_columns)(
            inputs)  # (-1, item_field_num * emb_size)
        context_feature_inputs = tf.keras.layers.DenseFeatures(self.context_columns)(
            inputs)  # (-1, context_field_num * emb_size)

        if self.linear_model:
            linear_model = self.linear_model
        else:
            linear_model = Linear(mode=1)
            # linear_model = LinearModel()

        if self.dnn_model:
            dnn_model = self.dnn_model
        else:
            dnn_model = DNN(hidden_units=[128, 64, 2], l2_reg=0.01)
            # dnn_model = tf.keras.Sequential([tf.keras.layers.Dense(units=256),
            #                                  tf.keras.layers.Dense(units=128),
            #                                  tf.keras.layers.Dense(units=64),
            #                               tf.keras.layers.Dense(units=2)])

        if self.sequence_model:
            sequence_model = self.sequence_model
        else:
            sequence_model = AttentionSequencePoolingLayer(att_hidden_units=(80, 40),
                                                           att_activation='sigmoid',
                                                           weight_normalization=False,
                                                           return_score=False,
                                                           supports_masking=True)

        outputs = self._forward(
            inputs=(user_hist_inputs, item_feature_inputs, user_feature_inputs, context_feature_inputs, linear_inputs),
            linear_model=linear_model,
            dnn_model=dnn_model,
            sequence_model=sequence_model)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)

        model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
        logging.info(model.summary())

        return model

    def train_model(self):
        # 暂时不实现
        pass
