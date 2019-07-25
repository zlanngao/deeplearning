import tensorflow as tf
import numpy as np
input_data=[
              [[1,0,1,2,1],
               [0,2,1,0,1],
               [1,1,0,2,0],
               [2,2,1,1,0],
               [2,0,1,2,0]],

               [[2,0,2,1,1],
                [0,1,0,0,2],
                [1,0,0,2,1],
                [1,1,2,1,0],
                [1,0,1,1,1]],
            ]
weights_data=[ 
               [[ 1, 0, 1],
                [-1, 1, 0],
                [ 0,-1, 0]],
               [[-1, 0, 1],
                [ 0, 0, 1],
                [ 1, 1, 1]] 
           ]
def get_shape(tensor):
    [s1,s2,s3]= tensor.get_shape() 
    s1=int(s1)
    s2=int(s2)
    s3=int(s3)
    return s1,s2,s3

#将[c,h,w]转为[h,w,c]
def chw2hwc(chw_tensor): 
    [c,h,w]=get_shape(chw_tensor) 
    cols=[]

    for i in range(c):
        #每个通道里面的二维数组转为[w*h,1]即1列 
        line = tf.reshape(chw_tensor[i],[h*w,1])
        cols.append(line)

    #横向连接，即将所有竖直数组横向排列连接
    input = tf.concat(cols,1)#[w*h,c]
    #[w*h,c]-->[h,w,c]
    input = tf.reshape(input,[h,w,c])
    return input
#将[h,w,c]转为[c,h,w]
def hwc2chw(hwc_tensor):
    [h,w,c]=get_shape(hwc_tensor) 
    cs=[] 
    for i in range(c): 
        #[h,w]-->[1,h,w] 
        channel=tf.expand_dims(hwc_tensor[:,:,i],0)
        cs.append(channel)
    #[1,h,w]...[1,h,w]---->[c,h,w]
    input = tf.concat(cs,0)#[c,h,w]
    return input

def tf_conv2d(input,weights):
    conv = tf.nn.conv2d(input, weights, strides=[1, 1, 1, 1], padding='SAME')
    return conv

def main(): 
    const_input = tf.constant(input_data , tf.float32)
    const_weights = tf.constant(weights_data , tf.float32 )


    input = tf.Variable(const_input,name="input")
    #[2,5,5]------>[5,5,2]
    input=chw2hwc(input)
    #[5,5,2]------>[1,5,5,2]
    input=tf.expand_dims(input,0)


    weights = tf.Variable(const_weights,name="weights")
    #[2,3,3]-->[3,3,2]
    weights=chw2hwc(weights)
    #[3,3,2]-->[3,3,2,1]
    weights=tf.expand_dims(weights,3) 

    #[b,h,w,c]
    conv=tf_conv2d(input,weights)
    rs=hwc2chw(conv[0]) 

    init=tf.global_variables_initializer()
    sess=tf.Session()
    sess.run(init)
    conv_val = sess.run(rs)

    print(conv_val[0]) 

if __name__=='__main__':
    main()
