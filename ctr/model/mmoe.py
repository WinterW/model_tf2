from typing import Dict, Text, List
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


class MMOEWrapper:
    def __init__(self,
                 feature_spec,
                 wide_columns,
                 deep_columns,
                 tower_model: Layer = None,
                 share_bottom_model: Layer = None,
                 expert_models: List[Layer] = None,
                 num_experts=4,
                 expert_dim=8,
                 num_tasks=2,
                 task_type='binary',  # ['binary', 'regression']
                 activation=None,
                 ):
        ''''''
        self.feature_spec = feature_spec
        self.wide_columns = wide_columns
        self.deep_columns = deep_columns
        self.tower_model = tower_model  # tower层
        self.share_bottom_model = share_bottom_model  # 共享层
        self.expert_models = expert_models  # Multi-gate Mixture-of-Experts
        self.num_experts = num_experts
        self.expert_dim = expert_dim
        self.num_tasks = num_tasks
        self.task_type = task_type
        self.activation = activations.get(activation)

    '''
    前向传播过程处理，输出outputs
    '''

    def _forward(self, inputs, tower_model, expert_models: List[Layer], share_bottom_model, training=None):

        share_bottom_outputs = share_bottom_model(inputs)

        expert_outputs = [tf.expand_dims(expert_model(share_bottom_outputs), axis=2) for expert_model in
                          expert_models]  # List (batch_size, self.expert_dim, 1) * self.num_experts
        gate_outputs = [tf.keras.layers.Dense(units=self.num_experts, activation='softmax', name='gate_' + str(i)) for i
                        in range(self.num_tasks)]  # (batch_size, num_experts)

        expert_outputs = tf.concat(expert_outputs, 2)  # (batch_size, self.expert_dim, self.num_experts)
        mmoe_outputs = []
        for gate_output in gate_outputs:
            expanded_gate_output = tf.expand_dims(gate_output, axis=1)  # (batch_size, 1, num_experts)
            weighted_expert_output = expert_outputs * tf.keras.backend.repeat_elements(expanded_gate_output,
                                                                                       self.expert_dim,
                                                                                       axis=1)  # (batch_size, self.expert_dim, self.num_experts)
            mmoe_outputs.append(tf.keras.backend.sum(weighted_expert_output, axis=2))

        outputs = []
        for index, mmoe_output in enumerate(mmoe_outputs):
            tower_output = tower_model(mmoe_output)
            outputs.append(tower_output)
        return outputs

    '''
    功能：拿到input、output，创建模型、编译、打印模型信息
    '''

    def build_model(self):
        inputs = input_layer(self.feature_spec)
        dnn_inputs = tf.keras.layers.DenseFeatures(self.deep_columns)(inputs)
        linear_inputs = tf.keras.layers.DenseFeatures(self.wide_columns)(inputs)

        if self.tower_model:
            tower_model = self.tower_model
        else:
            tower_model = tf.keras.Sequential([DNN(hidden_units=[128, 64], l2_reg=0.01),
                                               tf.keras.layers.Dense(units=1, activation='sigmoid')])

        if self.expert_models:
            expert_models = self.expert_models
        else:
            expert_models = [DNN(hidden_units=[64, 64, self.expert_dim], l2_reg=0.01, name='expert_' + i) for i in
                             range(self.num_experts)]

        if self.share_bottom_model:
            share_bottom_model = self.share_bottom_model
        else:
            share_bottom_model = DNN(hidden_units=[128, 64], l2_reg=0.01)
            # dnn_model = tf.keras.Sequential([tf.keras.layers.Dense(units=256),
            #                                  tf.keras.layers.Dense(units=128),
            #                                  tf.keras.layers.Dense(units=64),
            #                               tf.keras.layers.Dense(units=2)])

        outputs = self._forward(inputs=dnn_inputs,
                                tower_model=tower_model,
                                expert_models=expert_models,
                                share_bottom_model=share_bottom_model)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)

        model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      loss_weights=[0.5, 0.5],
                      metrics=['accuracy'])
        logging.info(model.summary())

        return model

    def train_model(self):
        # 暂时不实现
        pass
