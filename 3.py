
#基于FLANN的匹配器(FLANN based Matcher)定位图片
import numpy as np
import cv2
from matplotlib import pyplot as plt
 
MIN_MATCH_COUNT = 10 #设置最低特征点匹配数量为10
#template = cv2.imread('C:\\Users\\sys\\Desktop\\project_01\\pic\\1.jpg',0) # queryImage
#target = cv2.imread('C:\\Users\\sys\\Desktop\\project_01\\pic\\3.jpg',0) # trainImage
template = cv2.imread('C:\\Users\\86150\\Desktop\\project_01\\pic\\5.jpg',0) # queryImage
target = cv2.imread('C:\\Users\\86150\\Desktop\\project_01\\pic\\7.jpg',0) # trainImage

# Initiate SIFT detector创建sift检测器
sift = cv2.xfeatures2d.SIFT_create()
# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(template,None)
kp2, des2 = sift.detectAndCompute(target,None)
print(len(kp2))
#创建设置FLANN匹配
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 50)
flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1,des2,k=2)
# store all the good matches as per Lowe's ratio test.
good = []
#舍弃大于0.7的匹配
for m,n in matches:
    if m.distance < 0.7*n.distance:
        good.append(m)
print(len(good))
if len(good)>MIN_MATCH_COUNT:
    # 获取关键点的坐标
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
    #计算变换矩阵和MASK
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    matchesMask = mask.ravel().tolist()
    h,w = template.shape
    # 使用得到的变换矩阵对原图像的四个角进行变换，获得在目标图像上对应的坐标
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts,M)
    cv2.polylines(target,[np.int32(dst)],True,0,2, cv2.LINE_AA)
else:
    print( "Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT))
    matchesMask = None
draw_params = dict(matchColor=(0,255,0), 
                   singlePointColor=(0,0,255),
                   matchesMask=matchesMask, 
                   flags=2)
result = cv2.drawMatches(template,kp1,target,kp2,good,None,**draw_params)
plt.imshow(result, 'gray')
plt.show()


'''
import numpy as np

from matplotlib import pyplot as plt

import cv2


def sift(imgname1, imgname2):

    sift = cv2.xfeatures2d.SIFT_create()
    FLANN_INDEX_KDTREE = 0

    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    img1 = cv2.imread(imgname1)
#    img1 = cv2.resize(img1, (600, 400))
    kp1, des1 = sift.detectAndCompute(img1, None)    #des是描述子

    img2 = cv2.imread(imgname2)
#    img2 = cv2.resize(img2, (600, 400))
    kp2, des2 = sift.detectAndCompute(img2, None)

    matches = flann.knnMatch(des1,des2,k=2)

    good = []
    for m,n in matches:
        if m.distance < 0.70*n.distance:
            good.append([m])

    img5 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=2)
    cv2.imshow("FLANN", img5)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def surf(imgname1, imgname2):

    surf = cv2.xfeatures2d.SURF_create()
    FLANN_INDEX_KDTREE = 0

    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params,search_params)

    img1 = cv2.imread(imgname1)
#    img1 = cv2.resize(img1, (600, 400))
    kp1, des1 = surf.detectAndCompute(img1,None)    #des是描述子

    img2 = cv2.imread(imgname2)
#    img2 = cv2.resize(img2, (600, 400))
    kp2, des2 = surf.detectAndCompute(img2,None)

    matches = flann.knnMatch(des1,des2,k=2)

    good = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append([m])

    img5 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=2)
    cv2.imshow("SURF", img5)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def orb(imgname1, imgname2):

    orb = cv2.ORB_create()

    img1 = cv2.imread(imgname1)
#    img1 = cv2.resize(img1, (600, 400))
    kp1, des1 = orb.detectAndCompute(img1,None)#des是描述子

    img2 = cv2.imread(imgname2)
#    img2 = cv2.resize(img2, (600, 400))
    kp2, des2 = orb.detectAndCompute(img2,None)

    # BFMatcher解决匹配
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1,des2, k=2)
    # 调整ratio
    good = []
    for m,n in matches:
        if m.distance < 0.8*n.distance:
            good.append([m])

    img5 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=2)
    cv2.imshow("ORB", img5)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':

    imgname1 = 'C:\\Users\\sys\\Desktop\\project_01\\pic\\0.jpg'

    imgname2 = 'C:\\Users\\sys\\Desktop\\project_01\\pic\\16.jpg'


#    sift(imgname1, imgname2)

    surf(imgname1, imgname2)

    orb(imgname1, imgname2)
'''
'''
#基于FLANN的匹配器(FLANN based Matcher)定位图片
import numpy as np
import cv2
from matplotlib import pyplot as plt
 
MIN_MATCH_COUNT = 10 #设置最低特征点匹配数量为10
template = cv2.imread('C:\\Users\\sys\\Desktop\\project_01\\pic\\6.jpg',0) # queryImage
target = cv2.imread('C:\\Users\\sys\\Desktop\\project_01\\pic\\16.jpg',0) # trainImage
# Initiate SIFT detector创建sift检测器
surf = cv2.xfeatures2d.SURF_create()
# find the keypoints and descriptors with SIFT
kp1, des1 = surf.detectAndCompute(template,None)
kp2, des2 = surf.detectAndCompute(target,None)
#创建设置FLANN匹配
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 50)
flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1,des2,k=2)
# store all the good matches as per Lowe's ratio test.
good = []
#舍弃大于0.7的匹配
for m,n in matches:
    if m.distance < 0.7*n.distance:
        good.append(m)
print(len(good))
if len(good)>MIN_MATCH_COUNT:
    # 获取关键点的坐标
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
    #计算变换矩阵和MASK
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    matchesMask = mask.ravel().tolist()
    h,w = template.shape
    # 使用得到的变换矩阵对原图像的四个角进行变换，获得在目标图像上对应的坐标
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts,M)
    cv2.polylines(target,[np.int32(dst)],True,0,2, cv2.LINE_AA)
else:
    print( "Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT))
    matchesMask = None
draw_params = dict(matchColor=(0,255,0), 
                   singlePointColor=None,
                   matchesMask=matchesMask, 
                   flags=2)
result = cv2.drawMatches(template,kp1,target,kp2,good,None,**draw_params)
plt.imshow(result, 'gray')
plt.show()
'''