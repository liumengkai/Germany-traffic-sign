
import tensorflow as tf
import traffic_inference
import numpy as np
import readTrafficSigns
import  os
BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.01
LEARNING_RATE_DECAY = 0.99
REGULARZATION_RATE = 0.0001
TRAINING_STEPS = 30001
MOVING_AVERAGE_DECARY = 0.99  # 滑动平均的衰减率
MODEL_SAVE_PATH='D:/mymodel'
MODEL_NAME="model.ckpt"




def train(xs,ys):
    # 定义输入输出placeholder.
    x = tf.placeholder(tf.float32,
                       [BATCH_SIZE,  # 第一维表示一个batch中样例的个数
                        traffic_inference.IMAGE_SIZE,  # 第二维和第三维表示图片的尺寸。
                        traffic_inference.IMAGE_SIZE,
                        traffic_inference.NUM_CHANNELS],  # 第四维度表示图片的深度
                       name='x-input'
                       )
    y_ = tf.placeholder(tf.float32, [None,traffic_inference.OUTPUT_NODE], name='y-input')
    regularizer = tf.contrib.layers.l2_regularizer(REGULARZATION_RATE)
    # 直接使用mnist_inferience.py中的前向传播过程。
    y = traffic_inference.inference(x, True, regularizer)          #预测值在inference文件中进行计算前向传播值
    global_step = tf.Variable(0, trainable=False)

    # 定义损失函数，滑动平均率，学习率，以及训练过程。
    variable_average = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECARY, global_step)
    variable_average_op = variable_average.apply(
        tf.trainable_variables()
    )
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_,1))       #把神经网络和损失函数合在一起计算
    # y是正确的数字只有一个，y_是输出的数字有十个选出最大的一个
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        400,
        LEARNING_RATE_DECAY,
        staircase = True
    )
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)#最小化loss来进行反向传播
    with tf.control_dependencies([train_step, variable_average_op]):
        train_op = tf.no_op(name='train')

    # 初始化Tensorflow持久类
    saver=tf.train.Saver()
    with tf.Session() as sess:
        tf.initialize_all_variables().run()
        a=len(xs)
        # 训练过程中不再验证，测试与验证放在另一个程序中
        for i in range(TRAINING_STEPS):
            start=(i*BATCH_SIZE)%a
            end=min(start+BATCH_SIZE,a)
            xs_batch=xs[start:end]
            ys_batch=ys[start:end]
            reshaped_xs = np.reshape(xs_batch, (BATCH_SIZE,
                                          traffic_inference.IMAGE_SIZE,
                                          traffic_inference.IMAGE_SIZE,
                                          traffic_inference.NUM_CHANNELS))
            _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x:reshaped_xs,y_:ys_batch.eval()})
            #每1000轮保存一次模型。
            if i % 100== 0:
                # 输出当前训练情况，这里只输出了模型在当前训练batch上的损失函数，通过这个来近似了解当前训练情况。
                # 在验证数据上的正确信息会有一个单独的程序完成。
                print("After %d training step(s),loss on training " "batch is %g." % (step, loss_value))
                # 保存当前的模型。注意这里给出了global_step参数，这样可以让每个模型文件名最后都加上训练的轮数，比如
                saver.save(
                    sess,os.path.join(MODEL_SAVE_PATH,MODEL_NAME),global_step=global_step
                )
                # saver这个类里面带的函数最后由有参数可以自动加上步数，global_step


def main(self):
    images, ys = readTrafficSigns.readTrafficSigns('D:/files/比赛数据/交通标志识别数据/GTSRB/GTSRB/Final_Test/Images')
    images=np.array(images)
    ys_int = []
    for a in ys:
        a = int(a)
        ys_int.append(a)
    labels_expend = tf.expand_dims(ys_int, axis=1)  # 对矩阵或者向量增加一维，使其看得更清楚
    index_expends = tf.expand_dims(tf.range(len(ys_int)), axis=1)  # 和labels_expend上的元素一一对应，确定横坐标

    concat_result = tf.concat(values=[index_expends, labels_expend], axis=1)  # axis=1表示在列上跨越
    labels_one_hot = tf.sparse_to_dense(sparse_indices=concat_result,
                                        output_shape=tf.zeros([len(ys), traffic_inference.OUTPUT_NODE]).shape,
                                        sparse_values=1.0, default_value=0.0)
    train(images,labels_one_hot)
    print(len(ys))
if __name__ == '__main__':
    tf.app.run()
