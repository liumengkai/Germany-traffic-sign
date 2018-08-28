import time
import tensorflow as tf
#加载mnist_inference.py和mnist_train.py中常用的常量和函数
import traffic_inference
import traffic_train
import readTrafficSigns
import numpy as np
TEST_STEPS=127
BATCH_SIZE=100
#每十秒加载一次最新的模型，并在测试数据集上测试最新模型的正确率

def evaluate(images,labels):
    a=len(images)
    #定义输入输出的格式。
    x=tf.placeholder(
        tf.float32,
        [BATCH_SIZE,  # 第一维表示一个batch中样例的个数
            traffic_inference.IMAGE_SIZE,  # 第二维和第三维表示图片的尺寸。
            traffic_inference.IMAGE_SIZE,
            traffic_inference.NUM_CHANNELS],  # 第四维度表示图片的深度
            name='x-input'

    )
    y_= tf.placeholder(tf.float32, [None,traffic_inference.OUTPUT_NODE], name='y-input')

#直接用封装好的类来计算前向传播的结果，因为测试时候不关注正则化损失函数的值，所以这里用于计算正则化损失的函数被设置为None.
    y=traffic_inference.inference(x,False,None)

    #使用前向传播的结果计算正确率，如果需要对未知的样例进行分类，那么使用
    #tf.argmax(y,1)就可以得到输出样本的类别了。
    correct_prediction=tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
    accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))   #类型转换函数

    #通过变量重命名来加载模型，这样在前向传播过程中就不需要调用滑动平均的函数来获取平均值了。
    # 这样就可以完全公用mnist_inference.py中的前向传播过程了
    variable_averages=tf.train.ExponentialMovingAverage(
        traffic_train.MOVING_AVERAGE_DECARY)
    variable_to_restore=variable_averages.variables_to_restore()#加载模型时候可以将影子变量映射到变量本身
    saver=tf.train.Saver(variable_to_restore)

    #每隔EVAL_INTERVAL_SECS秒掉哦那个一次计算正确率的过程已检测训练过程中正确率的变化


    with tf.Session() as sess:
        #tf.train.get_checkpoint_state函数会通过checkpoint文件自动找到目录中最新的文件
        ckpt = tf.train.get_checkpoint_state(traffic_train.MODEL_SAVE_PATH)
        b=0
        if ckpt and ckpt.model_checkpoint_path:
            # 加载模型
            saver.restore(sess, ckpt.model_checkpoint_path)
            # 通过文件名得到迭代的轮数。
            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        else:
            print('No checkpoint file found')
            return
        for i in range(201):
            start = (i * BATCH_SIZE) % a
            end = min(start + BATCH_SIZE, a)
            xs_batch = images[start:end]
            ys_batch = labels[start:end]
            reshaped_xs = np.reshape(xs_batch, (BATCH_SIZE,
                                                traffic_inference.IMAGE_SIZE,
                                                traffic_inference.IMAGE_SIZE,
                                                traffic_inference.NUM_CHANNELS))
            accuracy_score=sess.run(accuracy,feed_dict={x:reshaped_xs,y_:ys_batch.eval()})
            b=b+accuracy_score
        print(b/201)


def main(argv=None):
    images,ys=readTrafficSigns.readTrafficSigns('D:/files/比赛数据/交通标志识别数据/GTSRB/GTSRB/Final_Test/Images')
    images = np.array(images)
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
    evaluate(images,labels_one_hot)
if __name__=='__main__':
    tf.app.run()