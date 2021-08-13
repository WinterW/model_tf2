import tensorflow as tf
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Layer, Embedding, Dot, Flatten

#模型定义skipgram
class Word2Vec(Model):
  def __init__(self, vocab_size, embedding_dim, num_ns):
    super(Word2Vec, self).__init__()
    self.target_embedding = Embedding(vocab_size,
                                      embedding_dim,
                                      input_length=1,
                                      name="w2v_embedding", )
    self.context_embedding = Embedding(vocab_size,
                                       embedding_dim,
                                       input_length=num_ns+1)
    self.dots = Dot(axes=(3,2))
    self.flatten = Flatten()

  def call(self, pair):
    target, context = pair
    #(batch_size, 1, embed_size)
    we = self.target_embedding(target)
    # (batch_size, context_size, 1, embed_size)
    ce = self.context_embedding(context)
    dots = self.dots([ce, we])
    return self.flatten(dots)

def custom_loss(x_logit, y_true):
  return tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=y_true)

