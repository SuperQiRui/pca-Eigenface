#coding=UTF-8
from numpy import * 
import numpy as np
from numpy import linalg as la
import cv2
import os
 
def loadImageSet(add):
    FaceMat = mat(zeros((60,128*128)))
    j =0
    t=0
    nameList = ['001','002','003','004','005','006','007','008','009','010','011','012','013','014','015']
    num = ['01','02','03','04']
    for i in range(len(nameList)):  
         for j in range(len(num)):
              loadname = add+nameList[i]+ num[j] +'.bmp'
              img = cv2.imread(loadname,0)
              FaceMat[t,:] = mat(img).flatten()
              t=t+1
                         
    cv2.imwrite("FaceMat.jpg",FaceMat)
    return FaceMat
 
def ReconginitionVector(selecthr = 0.85):
    FaceMat = loadImageSet(r'C:\Users\RUI\Desktop\pca\mypca/train/').T #所有训练图片组成的矩阵
    avgImg = mean(FaceMat,1)#平均脸
    diffTrain = FaceMat-avgImg #差值矩阵
    eigvals,eigVects = linalg.eig(mat(diffTrain.T*diffTrain)) # numpy.linalg.eig() 计算矩阵特征向量
    # print("eigVects",eigVects.shape)
    #特征值和特征向量
    eigSortIndex = argsort(-eigvals) #排序从大到小 函数返回的是索引
    # 下面这个循环是找出对每一个图的特征值合最大的特征向量
    for i in range(shape(FaceMat)[1]):
        if (eigvals[eigSortIndex[:i]]/eigvals.sum()).sum() >= selecthr: #PCA
            eigSortIndex = eigSortIndex[:i]
            break
    covVects = diffTrain * eigVects[:,eigSortIndex] 
    #covVects是协方差矩阵的特征向量
    cv2.imwrite("avgImg.jpg",avgImg.reshape(128,128))
   
    # dif = mat(zeros((16384,1)))
    # for i in range(23) :
    #      for j in range(16384) :
    #         dif[j]= (covVects[j,i] * np.conj(covVects[j,i])).real.astype(np.float32)         
    
    #      cv2.imwrite("covVects%d.jpg"%i,dif.reshape(128,128)) 
    # print("covVects",covVects.shape)
    return avgImg,covVects,diffTrain
 
def judgeFace(judgeImg,FaceVector,avgImg,diffTrain):
    diff = judgeImg.T - avgImg
    weiVec = FaceVector.T* diff ##求测试图特征值
    res = 0
    resVal = inf
    
    for i in range(60):
        TrainVec = FaceVector.T*diffTrain[:,i] #求原训练图出特征值
        if  (array(weiVec-TrainVec)**2).sum() < resVal: #欧式距离
            res =  i
            resVal = (array(weiVec-TrainVec)**2).sum()
    # print("TrainVec",TrainVec.shape)
    return res+1
 

avgImg,FaceVector,diffTrain = ReconginitionVector(selecthr = 0.9)
nameList = ['001','002','003','004','005','006','007','008','009','010','011','012','013','014','015']
num = ['08','09','10']

count = 0
for i in range(len(nameList)):
     
     for j in range(len(num)):
         # 这里的loadname就是我们要测试的人脸图
        loadname = r'C:\Users\RUI\Desktop\pca\mypca/test/'+nameList[i]+ num[j] +'.bmp'
        # loadname = r"C:\Users\RUI\Desktop\pca\mypca/test/00108.bmp"
        judgeImg = cv2.imread(loadname,0)
        # print(loadname)
        t= judgeFace(mat(judgeImg).flatten(),FaceVector,avgImg,diffTrain)
        # print((int(nameList[i])-1)*4)
        # print("predict",t)
        # print(int(nameList[i])*4,"\n")   
        if (int(nameList[i])-1)*4 < t <= (int(nameList[i]))*4:
             count += 1
# print(count)
print ("Correct identification %s photos,Accuracy is %f"%(count, float(count)/(len(nameList)*len(num)) ))