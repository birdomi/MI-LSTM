import tensorflow as tf


class Attention_Layer():
    """
    어텐션 레이어.
    (None, TimeWindow, hidden_unit_size) shape의 LSTM 출력을 입력으로 받아 (None, 1, hidden_unit_size)의 텐서 출력.
    
    """
    def __init__(
        self,
        timewindow_size,
        input_hidden_unit_size):

        self.h_size=input_hidden_unit_size
        self.t_size=timewindow_size

        self.beta_weight=tf.ones([self.h_size,self.h_size])
        self.beta_bias=tf.zeros([self.h_size])

        self.v=tf.ones([self.h_size,1])

    def __call__(self,inputs):
        temp=tf.reshape(tf.matmul(tf.reshape(inputs,[-1,self.h_size]),self.beta_weight),[-1,self.t_size,self.h_size])
        temp=tf.tanh(temp+self.beta_bias)
            
        #j=tf.matmul(temp,self.v)
        j=tf.reshape(tf.matmul(tf.reshape(temp,[-1,self.h_size]),self.v),[-1,self.t_size,1])

        beta=tf.nn.softmax(j)

        output=beta*inputs
        return output

sess=tf.Session()

inputs=tf.ones([3,5,2])

b=tf.constant([1,2,1,2],dtype=tf.float32,shape=[2,2])
c=tf.constant([5,5],dtype=tf.float32,shape=[2])


d=tf.broadcast_to(b,(3,2,2))

result=tf.matmul(inputs,d)+c
print(sess.run([b,d]))
print(sess.run([tf.shape(result),result]))

a_layer=Attention_Layer(5,2)
print(sess.run([tf.shape(a_layer(inputs)),a_layer(inputs)]))



















"""
for testing shape of alpha
sess =tf.Session()

a=tf.ones([3,5])
b=a*2
c=a*3

d=[a,b,c]

alpha=tf.nn.softmax(d,axis=0)

#making L.
L=[]
for i,l in enumerate(d):
    L.append(alpha[i]*l)
L=tf.reduce_sum(L,axis=0)   

sess=tf.Session()
print(sess.run([tf.shape(d),d]))
print(sess.run([tf.shape(L),L]))
print(sess.run([tf.shape(alpha),alpha]))
"""

