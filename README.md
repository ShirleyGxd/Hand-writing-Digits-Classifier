# Hand-writing-Digits-Classifier
Hand-writing Digits Classifier based on "Binary Image + Naïve Bayes" and "LBP + SVM"

## 1、简介

　　这个项目主要是分别用"Binary Image + Naïve Bayes"的算法和"LBP + SVM"的算法实现了手写数字图片的分类。
  
## 2、Binary Image + Naïve Bayes

　　1)对应程序：main_naive_bayes.m，直接点击运行即可。
  
　　2)提取的特征为Binary Image：像素值非0的地方设置为1，其他地方设置为0。
  
　　3)分类器为Naïve Bayes：每一个像素位置作为一个属性+拉普拉斯修正
  
## 3、LBP + SVM

　　1)对应程序：main_svm.m，其中为了减少程序运行时间，我将计算好的特征和训练好的SVM模型保存到了.mat中，运行时会直接加载这些数据。

    需要libsvm的库函数，考虑到版权的问题，需要自行下载。程序中使用的版本是libsvm-3.21。

    配置好libsvm库函数后，直接运行main_svm.m即可。
  
　　2)提取的特征为LBP：像素值非0的地方设置为1，其他地方设置为0。
  
　　3)分类器为SVM：因为分类的结果是数字0-9，共10类。所以需要multi-class SVM，程序中采用的是One VS One的方案。
  
## 4、结果比较：
  
　　训练和测试数据：MNIST数据集。

　　LBP + SVM和Binary Image + Naïve Bayes都在0.84左右，性能差不多。
