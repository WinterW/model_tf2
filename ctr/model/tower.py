from absl import logging
import collections
from typing import Dict, Text
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Embedding, Dropout, Input
from tensorflow.keras import activations
from tensorflow.keras.regularizers import l2
import tensorflow_recommenders as tfrs

from layer.fm import FM
from layer.dnn import DNN
from layer.linear import Linear
from utils.input import input_layer
from tensorflow.python.keras.layers import Layer
from deepctr.layers.interaction import FM


'''
用组合将模型重新封装,使用函数式 API, 因为model类对输入不灵活，比如使用特征工程时想将Input作为输入
'''


# # 自定义一个层次，用于计算损失，输出不使用，使用add_loss收集loss
# class MultiTaskLossLayer(Layer):
#     def __init__(self, rating_weight, retrieval_weight, name="MultiTaskLossLayer"):
#         super().__init__(name=name)
#         self.rating_weight = rating_weight
#         self.retrieval_weight = retrieval_weight
#         self.rating_task: tf.keras.layers.Layer = tfrs.tasks.Ranking(
#             loss=tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.SUM),
#             # 使用tf.distribute时，需要明确损失https://www.tensorflow.org/tutorials/distribute/custom_training
#             metrics=[tf.keras.metrics.RootMeanSquaredError()],
#         )
#         self.retrieval_task: tf.keras.layers.Layer = tfrs.tasks.Retrieval(
#             metrics=None
#         )
#
#     def call(self, inputs, training=None, **kwargs):
#         ratings = inputs[0]
#         user_embeddings, movie_embeddings, rating_predictions = inputs[1]
#         rating_loss = self.rating_task(
#             labels=ratings,
#             predictions=rating_predictions,
#         )
#         retrieval_loss = self.retrieval_task(user_embeddings, movie_embeddings)
#
#         # add_loss本用于收集layer的正则化损失，这里使用add_loss收集损失，方便灵活定义损失，不局限于(y_true,y_predict)的参数定义
#         self.add_loss(self.rating_weight * rating_loss + self.retrieval_weight * retrieval_loss)
#
#         # unused return value
#         return None


class TowerWrapper:
    def __init__(self,
                 feature_spec,
                 user_columns,
                 item_columns,
                 interaction_columns,
                 rating_weight: float,
                 retrieval_weight: float,
                 user_model: Layer = None,  # 输出一个用户向量
                 item_model: Layer = None,  # 输出一个item向量
                 match_model: Layer = None,
                 rating_model: Layer = None,
                 interaction_model: Layer = None,
                 activation=None,
                 ):
        ''''''
        self.feature_spec = feature_spec
        self.user_columns = user_columns
        self.item_columns = item_columns
        self.interaction_columns = interaction_columns
        self.rating_weight = rating_weight
        self.retrieval_weight = retrieval_weight
        self.user_model = user_model
        self.item_model = item_model
        self.match_model = match_model
        self.rating_model = rating_model
        self.interaction_model = interaction_model
        self.activation = activations.get(activation)

    '''
    前向传播过程处理，输出outputs
    '''

    def _forward(self, inputs, user_model, item_model, interaction_model, rating_model, training=None):
        if not isinstance(inputs, (tuple, list)) or len(inputs) != 3:
            raise ValueError('input must have 3 item.')

        user_inputs, item_inputs, interaction_input = inputs
        user_embeddings = user_model(user_inputs)
        item_embeddings = item_model(item_inputs)
        interaction_output = interaction_model(interaction_input)

        # todo:暂时简单concat，
        output = tf.concat([user_embeddings, item_embeddings, interaction_output], axis=1)
        output = rating_model(output)

        return user_embeddings, item_embeddings, output

    '''
    功能：拿到input、output，创建模型、编译、打印模型信息
    '''

    def build_model(self):
        inputs = input_layer(self.feature_spec)
        user_inputs = tf.keras.layers.DenseFeatures(self.user_columns)(inputs)
        item_inputs = tf.keras.layers.DenseFeatures(self.item_columns)(inputs)
        interaction_input = tf.keras.layers.DenseFeatures(self.interaction_columns)(inputs)

        if self.user_model:
            user_model = self.user_model
        else:
            user_model = DNN(hidden_units=[128, 64, 64], l2_reg=0.01)

        if self.item_model:
            item_model = self.item_model
        else:
            item_model = DNN(hidden_units=[128, 64, 64], l2_reg=0.01)
            # dnn_model = tf.keras.Sequential([tf.keras.layers.Dense(units=256),
            #                                  tf.keras.layers.Dense(units=128),
            #                                  tf.keras.layers.Dense(units=64),
            #                               tf.keras.layers.Dense(units=2)])

        if self.interaction_model:
            interaction_model = self.interaction_model
        else:
            interaction_model = FM()

        if self.rating_model:
            rating_model = self.rating_model
        else:
            rating_model = DNN(hidden_units=[128, 64, 1], l2_reg=0.01)

        if self.match_model:
            match_model = self.match_model
        else:
            # 每个batch内负采样计算交叉墒
            match_model = tfrs.tasks.Retrieval(
                metrics=None  # tfrs.metrics.FactorizedTopK(candidates=movies.batch(128).map(self.item_model))
            )

        outputs = self._forward(inputs=(user_inputs, item_inputs, interaction_input),
                                user_model=user_model,
                                item_model=item_model,
                                interaction_model=interaction_model,
                                rating_model=rating_model)

        user_embeddings, item_embeddings, rating_output = outputs

        retrieval_loss = match_model(user_embeddings, item_embeddings)
        model = tf.keras.Model(inputs=inputs, outputs=rating_output)
        # add_loss本用于收集layer的正则化损失，这里使用add_loss收集召回任务损失，不局限于(y_true,y_predict)的参数定义, 排序任务损失，正常计算
        # todo:后面可能需要针对不同的损失定义不同训练步骤，在此暂时简单处理
        model.add_loss(retrieval_loss)
        model.compile(optimizer='adam',
                      loss='binary_crossentropy',  # 使用loss层计算损失
                      metrics=['accuracy'])
        logging.info(model.summary())

        return model

    def train_model(self):
        # 暂时不实现

        pass
