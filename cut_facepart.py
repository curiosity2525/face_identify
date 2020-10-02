import numpy
import cv2

# 画像データのあるディレクトリ
input_data_path = '画像データのあるパス'
# 切り抜いた画像の保存先ディレクトリ
save_path = '切り抜いた画像の保存先のパス'
# OpenCVのデフォルトの分類器のpath。(https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xmlのファイルを使う)
cascade_path = 'haarcascade_frontalface_default.xml'
faceCascade = cv2.CascadeClassifier(cascade_path)

# 収集した画像の枚数(任意で変更)
image_count = 87 
# 顔検知に成功した数(デフォルトで0を指定)
face_detect_count = 0

# 集めた画像データから顔が検知されたら、切り取り、保存する。
for i in range(image_count):
    
    img = cv2.imread(input_data_path + str(i) + '.jpg', cv2.IMREAD_COLOR)
    
    print(img.shape)
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face = faceCascade.detectMultiScale(gray, 1.1, 3)
    if len(face) > 0:
        for rect in face:
            x = rect[0]
            y = rect[1]
            w = rect[2]
            h = rect[3]
            cv2.imwrite(save_path + 'kiyoe' + str(face_detect_count) + '.jpg', img[y:y+h, x:x+w])
            face_detect_count = face_detect_count + 1
    else:
        print('image' + str(i) + ':NoFace')




