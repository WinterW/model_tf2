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
from layer.crossnet import CrossNet
from layer.crossnet import CrossNetMix

'''
用组合将模型重新封装,使用函数式 API, 因为model类对输入不灵活，比如使用特征工程时想将Input作为输入
'''


class DCNWrapper:
    def __init__(self,
                 feature_spec,
                 wide_columns,
                 deep_columns,
                 linear_model: Layer = None,
                 dnn_model: Layer = None,
                 dcn_model: Layer = None,
                 dcn_type='dcn-mix',  # dcn-v dcn-m dcn-mix
                 activation=None,
                 ):
        ''''''
        self.feature_spec = feature_spec
        self.wide_columns = wide_columns
        self.deep_columns = deep_columns
        self.linear_model = linear_model
        self.dnn_model = dnn_model
        self.dcn_model = dcn_model
        self.dcn_type = dcn_type
        self.activation = activations.get(activation)

    '''
    前向传播过程处理，输出outputs
    '''

    def _forward(self, inputs, linear_model, dnn_model, dcn_model, training=None):
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

        crossnet_output = dcn_model(dnn_inputs)  # (batch_size, units)
        crossnet_output = tf.keras.layers.Dense(1,
                                                kernel_initializer=tf.keras.initializers.glorot_normal(self.seed)
                                                )(crossnet_output)
        output = tf.nest.map_structure(
            lambda x, y, z: (x + y + z), linear_output, dnn_output, crossnet_output)  # dnn输出维度可能不为1，使用
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

        if self.dcn_model:
            dcn_model = self.dcn_model
        else:
            if self.dcn_type == 'dcn-v':
                dcn_model = CrossNet(layer_num=2, parameterization='vector')
            elif self.dcn_type == 'dcn-m':
                dcn_model = CrossNet(layer_num=2, parameterization='matrix')
            elif self.dcn_type == 'dcn-mix':
                dcn_model = CrossNetMix(low_rank=32, num_experts=4, layer_num=2)

        outputs = self._forward(inputs=(linear_inputs, dnn_inputs),
                                linear_model=linear_model,
                                dnn_model=dnn_model,
                                dcn_model=dcn_model)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)

        model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
        logging.info(model.summary())

        return model

    def train_model(self):
        # 暂时不实现
        pass
