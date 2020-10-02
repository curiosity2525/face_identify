import sys
import numpy as np

import tensorflow as tf
import os
import random
import cv2
from PIL import Image, ImageDraw, ImageFont

NUM_CLASSES = 3
IMAGE_SIZE = 28
IMAGE_PIXELS = IMAGE_SIZE*IMAGE_SIZE*3
batch_size = 10
learning_rate = 1e-5

def inference(images_placeholder, keep_prob):
    # 重みを標準偏差0.1の正規分布で初期化
    def weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    # バイアスを標準偏差0.1の正規分布で初期化
    def bias_variable(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    # 畳み込み層の作成
    def conv2d(x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    # プーリング層の作成
    def max_pool_2x2(x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                            strides=[1, 2, 2, 1], padding='SAME')
    
    # 入力を28x28x3に変形
    x_image = tf.reshape(images_placeholder, [-1, 28, 28, 3])

    # 畳み込み層1の作成
    with tf.name_scope('conv1') as scope:
        W_conv1 = weight_variable([5, 5, 3, 32])
        b_conv1 = bias_variable([32])
        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

    # プーリング層1の作成
    with tf.name_scope('pool1') as scope:
        h_pool1 = max_pool_2x2(h_conv1)
    
    # 畳み込み層2の作成
    with tf.name_scope('conv2') as scope:
        W_conv2 = weight_variable([5, 5, 32, 64])
        b_conv2 = bias_variable([64])
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

    # プーリング層2の作成
    with tf.name_scope('pool2') as scope:
        h_pool2 = max_pool_2x2(h_conv2)

    # 全結合層1の作成
    with tf.name_scope('fc1') as scope:
        W_fc1 = weight_variable([7*7*64, 1024])
        b_fc1 = bias_variable([1024])
        h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
        # dropoutの設定
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # 全結合層2の作成
    with tf.name_scope('fc2') as scope:
        W_fc2 = weight_variable([1024, NUM_CLASSES])
        b_fc2 = bias_variable([NUM_CLASSES])

    # ソフトマックス関数による正規化
    with tf.name_scope('softmax') as scope:
        y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

    # 各ラベルの確率のようなものを返す
    return y_conv

def loss(logits, labels):
    # 交差エントロピーの計算
    cross_entropy = -tf.reduce_sum(labels*tf.log(logits))
    # TensorBoardで表示するよう指定
    tf.summary.scalar("cross_entropy", cross_entropy)
    return cross_entropy

def training(loss, learning_rate):
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    return train_step

def accuracy(logits, labels):
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    tf.summary.scalar("accuracy", accuracy)
    return accuracy
###############################################################################
cascade_path = "パス"
###############################################################################

faceCascade = cv2.CascadeClassifier(cascade_path)

#識別ラベルと各ラベル番号に対応する名前
HUMAN_NAMES = {
    0: u"桜井和寿",
    1: u"吉岡聖恵",
    2: u"福山雅治"
}

#指定した画像を学習結果を用いて判定する
def evaluation(img_path, ckpt_path):
    tf.reset_default_graph()
    #画像を開く
    f = open(img_path, 'r')
    #画像読み込み
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    #モノクロ画像に変換
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face = faceCascade.detectMultiScale(gray, 1.1, 3)
    if len(face) > 0:
        for rect in face:
            #加工した画像に適当な名前をつける
            random_str = str(random.random())
            #顔部分を赤線で書く
            cv2.rectangle(img, tuple(rect[0:2]), tuple(rect[0:2]+rect[2:4])
                          ,(0,0,255), thickness=2)
            #顔部分を赤線で囲んだ画像の保存先を記入
            ###############################################################################
            face_detect_img_path = '/顔検出した画像のパス'+random_str+'.jpg' #顔検出した画像のパスを記入
            ###############################################################################
            #顔部分を赤線で囲んだ画像の保存
            cv2.imwrite(face_detect_img_path, img)
            x = rect[0]
            y = rect[1]
            w = rect[2]
            h = rect[3]
            #検出した顔を切り抜いた画像を保存する場所を記入
            ###############################################################################
            cv2.imwrite('/保存先のパス' + random_str + '.jpg', img[y:y+h, x:x+x+w]) 
            #Tensorflowへ渡す切り抜いた顔画像
            target_image_path = '/保存先のパス' + random_str + '.jpg' 
           ###############################################################################
    else:
        #顔が見つからなければ処理終了
        print('image:NoFace')
        return
    f.close()
    
    f = open(target_image_path, 'r')
    #データを入れる配列
    image = []
    img = cv2.imread(target_image_path)
    #28px*28pxにリサイズ
    img = cv2.resize(img, (28, 28))
    #画像情報を1列にした後, 0-1のfloat値にする
    image.append(img.flatten().astype(np.float32)/255.0)
    #numpy形式に変換し、Tensorflowで処理できるようにする
    image = np.asarray(image)
    #入力画像に対して各ラベルの確立を出力して返す
    logits = inference(image, 1.0)
    #We can just use 'c.eval()' without passing 'sess'
    sess = tf.InteractiveSession()
    #restore(パラメータ読み込み)の準備
    saver = tf.train.Saver()
    #変数の初期化
    sess.run(tf.initialize_all_variables())
    if ckpt_path:
        #学習後のパラメータの読み込み
        saver.restore(sess, ckpt_path)
    #sess.run(logits)と同じ
    softmax = logits.eval()
    #判定結果
    result = softmax[0]
    #判定結果を%にして四捨五入
    rates = [round(n*100.0, 1) for n in result]
    humans = []
    #ラベル番号、名前、パーセンテージのHashを作成
    for index, rate in enumerate(rates):
        name = HUMAN_NAMES[index]
        humans.append({
                'label':index,
                'name':name,
                'rate':rate
        })
    #パーセンテージの高い順にソート
    rank = sorted(humans, key=lambda x:x['rate'], reverse=True)
    #判定結果と加工した画像のpathを返す
    return [rank, face_detect_img_path, target_image_path]
    
#コマンドラインからのテスト用
if __name__ == '__main__':
    #判定したい画像のパスを記入
    ###############################################################################
    img_path = 'パス' 
    #モデル1のパスを記入
    result = evaluation(img_path, 'パス')
    ###############################################################################
    print(result)
    result_str  = str(result[0][0]['rate']) + "%"
    result_name = result[0][0]['name']
    print(str(result_name))
    if(result_name == "桜井和寿"):
        result_str += "桜井和寿"
    
    elif(result_name == "吉岡聖恵"):
        result_str += "吉岡聖恵"
    
    elif(result_name == "福山雅治"):
        result_str += "福山雅治"
        
    
    #画像の読み込み
    img = Image.open(img_path)
    #drawインスタンスを生成
    draw = ImageDraw.Draw(img)
    #フォントの設定(フォントファイルのパスと文字の大きさ)
    ################################################################################################
    font = ImageFont.truetype("/System/Library/Fonts/Hiragino Sans GB.ttc", 50)
    ################################################################################################
    #文字を書く
    draw.text((10, 10), result_str, fill=(255, 0, 0), font=font)
    #画像の保存先
    ###############################################################################
    img.save("パス")
   ###############################################################################

