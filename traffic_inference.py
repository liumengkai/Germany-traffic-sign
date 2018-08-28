import readTrafficSigns
import tensorflow as tf
#配置神经网络参数
INPUT_NODE=784
OUTPUT_NODE=43           #使用位置生成y_的placeholder时候用到的

IMAGE_SIZE=28
NUM_CHANNELS=3
NUM_LABELS=64              #神经网络中第二层全连接层的输出节点

#第一套卷积层的深度与尺寸。
CONV1_DEEP=32
CONV1_SIZE=5
#第二套卷积层的尺寸与深度。
CONV2_DEEP=64
CONV2_SIZE=5
#全连接层的节点个数
FC_SIZE=512

#定义卷积神经网络的前向传播过程。这里添加一个新的参数train,用于区分训练过程和测试过程，在这个过程中用到dropout方法
def inference(input_tensor,train,regularizer):
    #声明第一层卷积层的变量并且实现传播过程。
    #通过使用不同的命名空间来隔离不同层的变量。
    with tf.variable_scope('layer1-conv1'):
        conv1_weights=tf.get_variable(
            "weight",[CONV1_SIZE,CONV1_SIZE,NUM_CHANNELS,CONV1_DEEP],
            initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv1_biases=tf.get_variable("bias",[CONV1_DEEP],initializer=tf.constant_initializer(0.01))
        #使用边长为5深度为32的过滤器，过滤器移动步长为1，使用全0填充
        conv1=tf.nn.conv2d(
            input_tensor,conv1_weights,strides=[1,1,1,1],padding='SAME'
        )
        relu1=tf.nn.relu(tf.nn.bias_add(conv1,conv1_biases))


    #实现第二层池化层的前向传播过程。这里选用最大池化层，池化层过滤器边长为2，使用全零填充且移动步长为2.
    # 这一层的输入是上一层的输出，也就是28*28*32,输出为14*14*32的矩阵
    with tf.name_scope('layer2-pool1'):
        pool1=tf.nn.max_pool(
            relu1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME'
        )

    #声明第三层卷积层的变量并实现前向传播过程，这一层的输入为14*14*32
    #输出为14*14*64的矩阵。
    with tf.variable_scope('layer3-conv2'):
        conv2_weights=tf.get_variable("conv2_weights",[CONV2_SIZE,CONV2_SIZE,CONV1_DEEP,CONV2_DEEP],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv2_biases=tf.get_variable("bias",[CONV2_DEEP],initializer=tf.constant_initializer(0.0))
        #使用边长为5，深度为64的过滤器，过滤器移动的步长为1,且使用全零填充。
        conv2=tf.nn.conv2d(pool1,conv2_weights,strides=[1,1,1,1],padding='SAME')
        relu2=tf.nn.relu(tf.nn.bias_add(conv2,conv2_biases))

    #实现第四城池化层的前向传播过程。这一层和第二层的结构是一样的。
    with tf.name_scope('layer4-pool2'):
        pool2=tf.nn.max_pool(relu2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

        #第四层输出转化为全连接层的输入格式拉直成为一个向量get_shape()函数
    pool_shape=pool2.get_shape().as_list()
    #计算矩阵拉伸成向量之后的长度，这个长度就是矩阵的长宽以及深度的乘积。注意这里
    #pool_shape[0]为一个batch中的数据的个数
    nodes=pool_shape[1]*pool_shape[2]*pool_shape[3]

    #节点个数为nodes，通过tf.reshape()函数将第四层的输出变为一个batch的向量。
    reshaped=tf.reshape(pool2,[pool_shape[0],nodes])
    #声明第五层全连接的变量并实现前向传播过程。这一层输入是拉直之后的向量3136，输出是一个512的向量
    # ，引入了droupout，一般只在全连接层而不是卷积层或者池化层使用
    with tf.variable_scope('layer5-fc1'):
        fc1_weights=tf.get_variable(
            "weight",[nodes,FC_SIZE],initializer=tf.truncated_normal_initializer(stddev=0.1)
        )
        #只有全连接层的权重需要正则化。
        if regularizer !=None :
            tf.add_to_collection('losses',regularizer(fc1_weights))
        fc1_biases=tf.get_variable(
            "bias",[FC_SIZE],initializer=tf.constant_initializer(0.1))

        fc1=tf.nn.relu(tf.matmul(reshaped,fc1_weights)+fc1_biases)
        if train: fc1=tf.nn.dropout(fc1,0.5)

    #声明第六层全链接变量并且实现前向传播过程。这一层的输入为一组长度为的向量，输出为一组长度为64的向量，
    # 这一层的输出经过softmax层之后就得到了最终的分类结果。
    with tf.variable_scope('layer6-fc2'):
        fc2_weights=tf.get_variable("weight",[FC_SIZE,NUM_LABELS],
                                    initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer !=None:
            tf.add_to_collection('losses',regularizer(fc2_weights))
        fc2_biases=tf.get_variable(
            "bias",[NUM_LABELS],initializer=tf.constant_initializer(0.1))
        logit=tf.matmul(fc1,fc2_weights)+fc2_biases
    return logit







