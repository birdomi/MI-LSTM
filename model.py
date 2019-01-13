import tensorflow as tf
import numpy as np
import newLSTM

class Model():
    """
    모든 예측모델들의 기본 클래스
    """
    def __init__(self,sess,name,windowsize,Pos,Neg):
        self.sess=sess
        self.name=name
        self.T=windowsize
        self.P=Pos
        self.N=Neg

        
        self._build_net()

    def _build_net(self):
        pass

class LSTM_Model(Model):
    """
    Basic LSTM list for test.
    """
    def _build_net(self):
        self.Y=tf.placeholder(tf.float32,[None,self.T,1])
        self.Xp=tf.placeholder(tf.float32,[None,self.P,self.T,1])
        self.Xn=tf.placeholder(tf.float32,[None,self.N,self.T,1])
        self.Xi=tf.placeholder(tf.float32,[None,self.T,1])
        self.Target=tf.placeholder(tf.float32,[None,1])


        Xps=tf.split(self.Xp,self.P,1)
        Xns=tf.split(self.Xn,self.N,1)
        Xp_list=[]
        Xn_list=[]

    
        LSTM=tf.nn.rnn_cell.LSTMCell(64,name='lstm1')
        
        Y_1,_=tf.nn.dynamic_rnn(LSTM,self.Y,dtype=tf.float32)
        Xi_1,_=tf.nn.dynamic_rnn(LSTM,self.Xi,dtype=tf.float32)
        for i in range(len(Xps)):
            o,_=tf.nn.dynamic_rnn(LSTM,tf.squeeze(Xps[i],axis=1),dtype=tf.float32)
            Xp_list.append(o)
        for i in range(len(Xns)):
            o,_=tf.nn.dynamic_rnn(LSTM,tf.squeeze(Xns[i],axis=1),dtype=tf.float32)
            Xn_list.append(o)
        Xp_1=tf.reduce_mean(Xp_list,0)
        Xn_1=tf.reduce_mean(Xn_list,0)

        result=tf.concat([Y_1,Xp_1,Xn_1,Xi_1],axis=2)

        #MI-LSTM
        LSTM2=newLSTM.MI_LSTMCell(64,4,name='lstm2')
        Y_2,_ =tf.nn.dynamic_rnn(LSTM2,result,dtype=tf.float32)

        #Attention_Layer
        attention_layer=newLSTM.Attention_Layer(self.T,64)
        Y_3=attention_layer(Y_2)

        #Non-linear units for producing final prediction.
        R_1=tf.layers.dense(tf.layers.flatten(Y_3),64,tf.nn.relu)
        R_2=tf.layers.dense(R_1,64,tf.nn.relu)
        R_3=tf.layers.dense(R_2,64,tf.nn.relu)
        R_4=tf.layers.dense(R_3,64,tf.nn.relu)
        R_5=tf.layers.dense(R_4,64,tf.nn.relu)
        R_6=tf.layers.dense(R_5,1)

        self.out=R_6
        
        self.cost=tf.losses.mean_squared_error(labels=self.Target,predictions=self.out)
        self.optimizer=tf.train.AdamOptimizer(0.001).minimize(self.cost)

    def outputs(self,y,xp,xn,xi,target):
        fd={self.Y:y,self.Xp:xp,self.Xn:xn,self.Xi:xi,self.Target:target}
        return self.sess.run(self.out,feed_dict=fd)

    def training(self,y,xp,xn,xi,target):
        fd={self.Y:y,self.Xp:xp,self.Xn:xn,self.Xi:xi,self.Target:target}
        return self.sess.run([self.cost,self.optimizer],feed_dict=fd)

    def returnCost(self,y,xp,xn,xi,target):
        fd={self.Y:y,self.Xp:xp,self.Xn:xn,self.Xi:xi,self.Target:target}
        return self.sess.run(self.cost,feed_dict=fd)
