# test
本训练在谷歌colab服务器上进行
运行
!git clone https://github.com/AlexeyAB/darknetdarknet
下载Darknet框架进行训练
obj中为盲道和障碍物数据集，在colab上解压缩，yolov3_custom_final.weights文件为训练结束后权重文件，其余为配置文件。


训练过程：
# 改变makefile，启用GPU和OPENCV
%cd darknet
!sed -i 's/OPENCV=0/OPENCV=1/' Makefile
!sed -i 's/GPU=0/GPU=1/' Makefile
!sed -i 's/CUDNN=0/CUDNN=1/' Makefile

# 建立Darknet
!make

#链接到谷歌云盘
%cd ..
from google.colab import drive
drive.mount('/content/gdrive')

#cd到图片和配置文件所在文件夹
#将obj.zip解压到Darknet的data文件夹下
!cp /mydrive/yolov3/obj.zip ../
!unzip ../obj.zip -d data/

# 从谷歌云盘上传自定义的.cfg到Darknet的cfg文件夹
!cp /mydrive/yolov3/yolov3_custom.cfg ./cfg

# 从谷歌云盘上传obj.names和obj.names数据文件到Darknet的data文件夹
!cp /mydrive/yolov3/obj.names ./data
!cp /mydrive/yolov3/obj.data  ./data

# 从谷歌云盘上传generate_train.py脚本到云虚拟机并运行
!cp /mydrive/yolov3/generate_train.py .
!python generate_train.py

# 上传预先训练好的卷积层权重
!wget http://pjreddie.com/media/files/darknet53.conv.74

//CTRL + SHIFT + i在浏览器控制台输入以下js命令并回车，每十分钟点击一次屏幕
function ClickConnect(){
console.log("Working"); 
document.querySelector("colab-toolbar-button#connect").click() 
}
setInterval(ClickConnect,60000)

# 训练盲道和障碍物自定义检测器
!./darknet detector train data/obj.data cfg/yolov3_custom.cfg darknet53.conv.74 -dont_show

#由于不可抗力断开了可以使用以下命令接着训练，不用从头开始，不然会很惨T~T
!./darknet detector train data/obj.data cfg/yolov3_custom.cfg /mydrive/yolov3/backup/yolov3_custom.weights -dont_show


# 训练完成后查看训练结果，更改cfg为测试模式
%cd cfg
!sed -i 's/batch=64/batch=1/' yolov3_custom.cfg
!sed -i 's/subdivisions=16/subdivisions=1/' yolov3_custom.cfg
%cd ..

# 运行以下命令查看训练结果
!./darknet detector test data/obj.data cfg/yolov3_custom.cfg /mydrive/yolov3/backup/yolov3_custom_last.weights /mydrive/images/safari.jpg -thresh 0.3
imShow('predictions.jpg')
