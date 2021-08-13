# model_tf2

tensorflow模型实现方式
* function api
* 继承tf.keras.model

继承model方式限制太多，使用function api进行封装

可能依赖的第三方库
deepctr

一些基础layer可以自己定义，可以用deepctr中的layer，包括
* linear
* dnn
* fm
