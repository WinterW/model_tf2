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
from deepctr.layers.interaction import FM


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


class TowerWrapper:
    def __init__(self,
                 feature_spec,
                 user_columns,
                 item_columns,
                 interaction_columns,
                 user_model: Layer = None,  # 输出一个用户向量
                 item_model: Layer = None,  # 输出一个item向量
                 activation=None,
                 ):
        ''''''
        self.feature_spec = feature_spec
        self.user_columns = user_columns
        self.item_columns = item_columns
        self.interaction_columns = interaction_columns
        self.user_model = user_model
        self.item_model = item_model
        self.activation = activations.get(activation)

    '''
    前向传播过程处理，输出outputs
    '''

    def _forward(self, inputs, linear_model, dnn_model, fm_model, training=None):
        if not isinstance(inputs, (tuple, list)) or len(inputs) != 2:
            linear_inputs = dnn_inputs = inputs
        else:
            linear_inputs, dnn_inputs = inputs

        linear_output = linear_model(linear_inputs)
        # pylint: disable=protected-access
        if dnn_model._expects_training_arg:
            if training is None:
                training = tf.keras.backend.learning_phase()
            dnn_output = dnn_model(dnn_inputs, training=training)
        else:
            dnn_output = dnn_model(dnn_inputs)

        column_num, dimension = _check_fm_columns(self.deep_columns)
        fm_inputs = tf.reshape(dnn_inputs, (-1, column_num, dimension))  # (batch_size,column_num, embedding_size)
        fm_output = fm_model(fm_inputs)  # (batch_size, feature_num, embedding_size)
        output = tf.nest.map_structure(
            lambda x, y, z: (x + y + z), linear_output, dnn_output, fm_output)  # dnn输出维度可能不为1，使用
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

        if self.fm_model:
            fm_model = self.fm_model
        else:
            fm_model = FM()

        outputs = self._forward(inputs=(linear_inputs, dnn_inputs),
                                linear_model=linear_model,
                                dnn_model=dnn_model,
                                fm_model=fm_model)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)

        model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
        logging.info(model.summary())

        return model

    def train_model(self):
        # 暂时不实现
        pass
