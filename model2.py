import sys
import cv2
import numpy as np
import tensorflow as tf

NUM_CLASSES = 3 #今回は3人なので3
IMAGE_SIZE = 28
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE * 3
batch_size = 10
learning_rate = 1e-5

#予測モデルの作成
def inference(images_placeholder, keep_prob):
    #重みを標準誤差0.1の正規分布で初期化
    def weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)
    
    #バイアスを標準誤差0.1の正規分布で初期化
    def bias_variable(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)
    
    #畳み込み層の作成
    def conv2d(x, W):
        #tf.nn.conv2d(input, filter, strides, padding) 
        return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')
    
    #プーリング層の作成
    def max_pool_2x2(x):
        #2*2のブロック、strides=[1,2,2,1]で2ピクセルずつずらす
        return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    
    #入力を28×28×3に変形
    x_image = tf.reshape(images_placeholder, [-1, 28, 28, 3])
    
    #畳み込み層1の作成
    with tf.name_scope('conv1') as scope:
        W_conv1 = weight_variable([3, 3, 3, 32])#5*5の重みを32個用意
        b_conv1 = bias_variable([32])#バイアスは出力チャンネルの分だけ準備
        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)#reLU関数
    
    #畳み込み層1_2の作成
    with tf.name_scope('conv1_2') as scope:
        W_conv1_2 = weight_variable([3, 3, 32, 32])#5*5の重みを32個用意
        b_conv1_2 = bias_variable([32])#バイアスは出力チャンネルの分だけ準備
        h_conv1_2 = tf.nn.relu(conv2d(h_conv1, W_conv1_2) + b_conv1_2)#reLU関数
    
    #プーリング層1の作成
    with tf.name_scope('pool1') as scope:
        h_pool1 = max_pool_2x2(h_conv1_2)#画像サイズを14×14に縮小
    
    #畳み込み層2の作成
    with tf.name_scope('conv2') as scope:
        W_conv2 = weight_variable([3, 3, 32, 64])#5×5パッチが32種類の64個
        b_conv2 = bias_variable([64])
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)#h_pool1とW_conv2の畳み込み
    
    #畳み込み層2_2の作成
    with tf.name_scope('conv2_2') as scope:
        W_conv2_2 = weight_variable([3, 3, 64, 64])#5×5パッチが32種類の64個
        b_conv2_2 = bias_variable([64])
        h_conv2_2 = tf.nn.relu(conv2d(h_conv2, W_conv2_2) + b_conv2_2)#h_pool1とW_conv2の畳み込み
    
    #プーリング層2の作成
    with tf.name_scope('pool2')as scope:
        h_pool2 = max_pool_2x2(h_conv2_2)
      
    #全結合層1の作成
    with tf.name_scope('fc1') as scope:
        W_fc1 = weight_variable([7*7*64, 1024])
        b_fc1 = bias_variable([1024])
        h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
        #dropoutの設定
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    
    #全結合層2の作成
    with tf.name_scope('fc2') as scope:
        W_fc2 = weight_variable([1024, NUM_CLASSES])
        b_fc2 = bias_variable([NUM_CLASSES])
    
    #ソフトマックス関数による正規化
    with tf.name_scope('softmax') as scope:
        y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
        
    #各ラベルの値を返す
    return y_conv

#損失関数
def loss(logits, labels):
    #交差エントロピー
    cross_entropy = -tf.reduce_sum(labels * tf.log(logits))
    #Tensorboardで表示
    tf.summary.scalar("cross_entropy", cross_entropy)
    return cross_entropy
    
def training(loss, learning_rate):
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    return train_step

#正解率の計算
def accuracy(logits, labels):
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    #Tensorboardで表示
    tf.summary.scalar("accuracy", accuracy)
    return accuracy
    
if __name__ == '__main__':
    #学習データのテキストファイル
    f = open("/Volumes/ハードディスク/data/train/data.txt", 'r')
    
    #データを入れる配列
    train_image = []
    train_label = []
    for line in f:
        line = line.rstrip() #改行なくす
        l = line.split() #スペース区切り
        img = cv2.imread(l[0]) #データ読み込み
        img = cv2.resize(img, (28, 28)) #28×28に縮小
        train_image.append(img.flatten().astype(np.float32)/255.0)#一列にし、0~1のfloatにする
        #ラベルを1-of-k方式で準備
        tmp = np.zeros(NUM_CLASSES)
        tmp[int(l[1])] = 1
        train_label.append(tmp)
        
    #numpy形式に変換
    train_image = np.asarray(train_image)
    train_label = np.asarray(train_label)
    f.close()
    #テストデータのテキストファイル
    f = open("パス", 'r')
    
    test_image = []
    test_label = []
    
    for line in f:
        line = line.rstrip()
        l = line.split()
        img = cv2.imread(l[0])
        img = cv2.resize(img, (28,28))
        test_image.append(img.flatten().astype(np.float32)/255.0)
        tmp = np.zeros(NUM_CLASSES)
        tmp[int(l[1])] = 1
        test_label.append(tmp)
    test_image = np.asarray(test_image)
    test_label = np.asarray(test_label)
    f.close()
    
    with tf.Graph().as_default():
        #画像を入れる仮のTensor
        images_placeholder = tf.placeholder("float", shape=(None, IMAGE_PIXELS))
        #ラベルを入れる仮のTensor
        labels_placeholder = tf.placeholder("float", shape=(None, NUM_CLASSES))
        #drop率を入れる仮のTensor
        keep_prob = tf.placeholder("float")
        
        #inference()を呼び出してモデルを作る
        logits = inference(images_placeholder, keep_prob)
        #損失を計算
        loss_value = loss(logits, labels_placeholder)
        #訓練
        train_op = training(loss_value, learning_rate)
        #精度
        acc = accuracy(logits, labels_placeholder)
        #保存の準備
        saver = tf.train.Saver()
        #Sessionの作成
        sess = tf.Session()
        #変数の初期化
        sess.run(tf.global_variables_initializer())
        #Tensorboardで表示する値の設定
        summary_op = tf.summary.merge_all()
        #データの保管場所
        summary_writer = tf.summary.FileWriter("パス", sess.graph_def)
        
        #訓練の実行
        for step in range(200):
            for i in range(int(len(train_image) / batch_size)):
                #batch_size分の画像に対して訓練の実行
                batch = batch_size * i
                #feed_dictでplaceholderに入れるデータを指定
                sess.run(train_op, feed_dict={
                        images_placeholder: train_image[batch:batch+batch_size],
                        labels_placeholder: train_label[batch:batch+batch_size],
                        keep_prob: 0.5})
            #1step毎に精度を計算
            train_accuracy = sess.run(acc, feed_dict={
                    images_placeholder: train_image,
                    labels_placeholder: train_label,
                    keep_prob: 1.0})
            print("step" + str(step) + " train_accuracy" + str(train_accuracy))
            
            #1step毎にTensorBoardに表示する値を追加
            summary_str = sess.run(summary_op, feed_dict={
                    images_placeholder: train_image,
                    labels_placeholder: train_label,
                    keep_prob: 1.0})
            summary_writer.add_summary(summary_str, step)

    #訓練が終了したらテストデータに対する精度を表示
    print(str(sess.run(acc, feed_dict={
                    images_placeholder: test_image,
                    labels_placeholder: test_label,
                    keep_prob: 1.0})))
    #最終的なモデルを保存
    save_path = saver.save(sess, "パス")
                    
                
