import numpy as np
import tensorflow as tf
import data as d
import model
import time

#parameter list
sess=tf.Session()
name='lstm'
timesize=20
timesize_for_calc_correlation=50
positive_correlation_stock_num=10
negative_correlation_sotck_num=10
train_test_rate=0.7
batch_size=512
#

kospi=d.StockData(
    'StockChart/SAMPLE',
    'StockChart/KOSPI.csv',
    timesize,timesize_for_calc_correlation,
    positive_correlation_stock_num,
    negative_correlation_sotck_num,
    train_test_rate,
    batch_size
    )

print('\n#training#')
lstmModel=model.LSTM_Model(
    sess,
    name,
    timesize,
    positive_correlation_stock_num,
    negative_correlation_sotck_num
    )



sess.run(tf.global_variables_initializer())
result_dic={}
for i in range(1000):
    #epoch start
    start_time = time.time()
    training_cost=0
    evalution_cost=0

    #training batch
    for batch in kospi.getBatch('training'):
        c,_=lstmModel.training(batch['y'],batch['xp'],batch['xn'],batch['xi'],batch['target'])
        training_cost+=c


    #evaluation batch
    for batch in kospi.getBatch('evaluation'):
        c=lstmModel.returnCost(batch['y'],batch['xp'],batch['xn'],batch['xi'],batch['target'])
        evalution_cost+=c
    
    #epoch end
    elapsed_time = time.time()-start_time
    training_cost=training_cost/kospi.batchNum
    evalution_cost=evalution_cost/kospi.batchNum
    result_dic[i]=[training_cost,evalution_cost]

    print('epoch : {}, t_cost : {:0.6f}, e_cost : {:0.6f}, elapsed time : {:0.2f}sec'.format(
        i,training_cost,evalution_cost,elapsed_time))
#
sorted_result=sorted(result_dic,key=lambda k:result_dic[k][1])
bestEpoch=sorted_result[0]
print('\n#Best result at epoch {}'.format(bestEpoch))
print('t_cost : {:0.6f}, e_cost : {:0.6f}'.format(result_dic[bestEpoch][0],result_dic[bestEpoch][1]))
