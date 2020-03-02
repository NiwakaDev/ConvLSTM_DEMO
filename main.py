import tensorflow as tf
import numpy as np

#ConvLSTM(https://arxiv.org/pdf/1506.04214v1.pdf)
class ConvLSTM2D(tf.compat.v1.nn.rnn_cell.RNNCell):

    def __init__(self, units, filters, **kwargs):
        super(ConvLSTM2D, self).__init__(**kwargs)
        self.filters = filters
        #units->(height, width, channels)
        #strides=[1, 1]、paddingなしの場合のhiddenの形状計算
        units[0] = units[0] - 3 + 1
        units[1] = units[1] - 3 + 1
        units[2] = self.filters
        state_size  = tf.TensorShape(units)
        self._state_size = tf.compat.v1.nn.rnn_cell.LSTMStateTuple(state_size, state_size)
        self._output_size = tf.TensorShape(state_size)

    def build(self, input_shape):
        
        #重みの初期値:Heの初期値 he_normal(https://medium.com/@prateekvishnu/xavier-and-he-normal-he-et-al-initialization-8e3d7a087528)
        #tf.keras.layers.Conv2Dに対して、activationを指定しない場合、活性化関数を使用しないことになる。
        self.conv_xi = tf.keras.layers.Conv2D(self.filters, 3, kernel_initializer="he_normal")
        self.conv_hi = tf.keras.layers.Conv2D(self.filters, 3, kernel_initializer="he_normal", padding="same")
        self.conv_xf = tf.keras.layers.Conv2D(self.filters, 3, kernel_initializer="he_normal")
        self.conv_hf = tf.keras.layers.Conv2D(self.filters, 3, kernel_initializer="he_normal", padding="same")
        self.conv_xo = tf.keras.layers.Conv2D(self.filters, 3, kernel_initializer="he_normal")
        self.conv_ho = tf.keras.layers.Conv2D(self.filters, 3, kernel_initializer="he_normal", padding="same")
        self.conv_xg = tf.keras.layers.Conv2D(self.filters, 3, kernel_initializer="he_normal")
        self.conv_hg = tf.keras.layers.Conv2D(self.filters, 3, kernel_initializer="he_normal", padding="same")
        #RECURRENT BATCH NORMALIZATION(https://arxiv.org/pdf/1603.09025.pdf)
        self.batch_xi = tf.keras.layers.BatchNormalization()
        self.batch_hi = tf.keras.layers.BatchNormalization()
        self.batch_xf = tf.keras.layers.BatchNormalization()
        self.batch_hf = tf.keras.layers.BatchNormalization()
        self.batch_xo = tf.keras.layers.BatchNormalization()
        self.batch_ho = tf.keras.layers.BatchNormalization()
        self.batch_xg = tf.keras.layers.BatchNormalization()
        self.batch_hg = tf.keras.layers.BatchNormalization()
        self.batch_cell = tf.keras.layers.BatchNormalization()
        self.build = True
    
    @property
    def output_size(self):
        return self._output_size
    @property
    def state_size(self):
        return self._state_size   

    def compute_mask(self, inputs, mask=None):
        # Just pass the received mask from previous layer, to the next layer or 
        # manipulate it if this layer changes the shape of the input
        return mask
    
    def call(self, inputs, states, mask=None, training=True):
        cell, hidden = states
        f = tf.nn.sigmoid(self.batch_xf(self.conv_xf(inputs)) + self.batch_hf(self.conv_hf(hidden)))
        i = tf.nn.sigmoid(self.batch_xi(self.conv_xi(inputs)) + self.batch_hi(self.conv_hi(hidden)))
        o = tf.nn.sigmoid(self.batch_xo(self.conv_xo(inputs)) + self.batch_hi(self.conv_ho(hidden)))
        g = tf.nn.tanh(self.batch_xg(self.conv_xg(inputs)) + self.batch_hg(self.conv_hg(hidden)))
        #statesの更新!!       
        new_cell      = f * cell + (i * g)
        new_cell      = self.batch_cell(new_cell, training=training)
        
        new_hidden = o * tf.nn.tanh(new_cell)
        new_state   = tf.compat.v1.nn.rnn_cell.LSTMStateTuple(new_cell, new_hidden)
        
        return new_hidden, new_state

if __name__ == '__main__':
    data1 = np.random.randn(1, 5, 3, 3, 3)
    mask1 = tf.cast([[True, True, True, False, False]], tf.bool)

    #3フレームまでは同じデータを生成する
    data_l = data1[:,0:3]
    #4フレーム以降のデータを生成
    data_r = np.random.randn(1, 2, 3, 3, 3)
    data2 = np.concatenate([data_l, data_r],1)
    mask2 = tf.cast([[True, True, True, False, False]], tf.bool)
    #シーケンスの最後尾だけ出力(1フレームだけ)
    convLSTM = tf.keras.layers.RNN(ConvLSTM2D([3, 3, 3], 32), return_sequences=False)

    #4フレーム以降の値に依存していないかをチェックする。
    #4フレーム以降の値をRNNに適用していた場合、data1とdata2では値は異なるはず。。。
    print(tf.math.equal(convLSTM(data1, mask=mask1), convLSTM(data2, mask=mask2)))