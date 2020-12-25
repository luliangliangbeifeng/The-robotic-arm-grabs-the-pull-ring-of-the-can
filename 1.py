'''
#opencv模板匹配----单目标匹配
import cv2

if __name__ == "__main__":
  #0是代表摄像头编号，只有一个的话默认为0
    capture=cv2.VideoCapture(0) 
    while(True):
        ref,frame=capture.read()
        #读取模板图片
        template = cv2.imread("C:\\Users\\sys\\Desktop\\project_01\\18.jpg")
        #获得模板图片的高宽尺寸
        theight, twidth = template.shape[:2]
        #执行模板匹配，采用的匹配方式cv2.TM_SQDIFF_NORMED
        result = cv2.matchTemplate(frame,template,cv2.TM_SQDIFF_NORMED)
        #归一化处理
        cv2.normalize( result, result, 0, 1, cv2.NORM_MINMAX, -1 )
        #寻找矩阵（一维数组当做向量，用Mat定义）中的最大值和最小值的匹配结果及其位置
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        #匹配值转换为字符串
        #对于cv2.TM_SQDIFF及cv2.TM_SQDIFF_NORMED方法min_val越趋近与0匹配度越好，匹配位置取min_loc
        #对于其他方法max_val越趋近于1匹配度越好，匹配位置取max_loc
        strmin_val = str(min_val)
        #绘制矩形边框，将匹配区域标注出来
        #min_loc：矩形定点
        #(min_loc[0]+twidth,min_loc[1]+theight)：矩形的宽高
        #(0,0,225)：矩形的边框颜色；2：矩形边框宽度
        cv2.rectangle(frame,min_loc,(min_loc[0]+twidth,min_loc[1]+theight),(0,0,225),2)
        print("min_loc.x",min_loc[0],"min_loc.y",min_loc[1])
        #显示结果,并将匹配值显示在标题栏上
        cv2.namedWindow('MatchResult----MatchingValue='+strmin_val, cv2.WINDOW_AUTOSIZE)
        cv2.imshow('MatchResult----MatchingValue='+strmin_val,frame)

#等待30ms显示图像，若过程中按“Esc”退出
        key = cv2.waitKey(1)
        # Press esc or 'q' to close the image window
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break
'''
import cv2
import math
import numpy as np
# 加载原始RGB图像
#img_rgb = cv2.imread('C:\\Users\\sys\\Desktop\\project_01\\16.jpg')
img_rgb = cv2.imread('C:\\Users\\sys\\Desktop\\project_01\\pic\\5.jpg')
# 创建一个原始图像的灰度版本，所有操作在灰度版本中处理，然后在RGB图像中使用相同坐标还原
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
# 加载将要搜索的图像模板
#模板1 筛选图案1
template1 = cv2.imread('C:\\Users\\sys\\Desktop\\project_01\\pic\\16.jpg', 0)
#template1 = cv2.imread('C:\\Users\\sys\\Desktop\\project_01\\18.jpg', 0)
#模板2 3 筛选图案2
#template2 = cv2.imread('template2.png', 0)
#template3 = cv2.imread('template3.png', 0)
# 记录图像模板的尺寸
w1, h1 = template1.shape[::-1]
print('w1:',w1,'h1',h1)
#w2, h2 = template2.shape[::-1]
#w3, h3 = template3.shape[::-1]

rotate_angle = []
sum1 = 0
Rotate_angle = 0

def rotate_bound(image, angle):#图片旋转但不改变大小，模板匹配中大小改变对匹配效果有影响
    (h, w) = image.shape[:2]                                          #取前两位
    (cX, cY) = (w // 2, h // 2)#//是向下取整                       #取中点x，y坐标
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])            #np.abs(x)、np.fabs(x)：计算数组各元素的绝对值    
    sin = np.abs(M[0, 1])
    
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
 
    return cv2.warpAffine(image, M, (nW, nH))

#选出所有匹配旋转模板且不重复的图案
def make_contour(template,w,h,angle,threshold):
    rects = []
    # 模板旋转匹配
    for i in range(0, 3600, angle):
        k = (float)(i / 10)
        new_rotate = rotate_bound(template, k)                        #注1
        # 把图片旋转后黑色的部分填充成白色
        new_rotate[new_rotate == 0] = 255
        # 使用matchTemplate对原始灰度图像和图像模板进行匹配
        res = cv2.matchTemplate(img_gray, new_rotate, cv2.TM_CCOEFF_NORMED)
        min_val,max_val,min_indx,max_indx=cv2.minMaxLoc(res)
        if max_val >= threshold:
            print(i)
            rotate_angle.append(i)
        # 设定阈值
        loc = np.where(res >= threshold)                              #注2
#        print('loc',loc)
        #x,y坐标对调打包
        for pt in zip(*loc[::-1]):
            point = np.array([[pt[0], pt[1]], [pt[0] + w, pt[1]],
                    [pt[0], pt[1] + h], [pt[0] + w, pt[1] + h]])
            rects.append(cv2.boundingRect(point))
    for j in rotate_angle:
        global sum1
        if j > 3500:
            j = j -3600
        sum1 += j
    global Rotate_angle
    Rotate_angle = sum1 / len(rotate_angle) / 10
    print('angle:',Rotate_angle)
    #模板匹配后符合要求的所有图案数量
    length = len(rects)
    #设定阈值
    threshold = 30
    i = 0
    #如果两个图案距离在阈值范围内，则等同，然后用集合去重
    while(i<length):
#        print(i)
        for j in range(length):
            if j != i:
                if np.abs(rects[j][0]-rects[i][0])<= threshold:
                    if np.abs(rects[j][1]-rects[i][1]) <= threshold:
                        rects[j] = rects[i]
        i = i+1
    return set(rects)

#在原图把匹配的模板框出来并输出坐标文档
def draw_contour(contours,color):
    count = 0
    global Rotate_angle
    Angle = Rotate_angle * math.pi / 180
    for contour in contours:
        cv2.rectangle(img_rgb, (contour[0], contour[1]), (contour[0] + contour[2], contour[1] + contour[3]),
                             color, 1)
        cx = contour[0] + (contour[2] // 2)
        cy = contour[1] + (contour[3] // 2)
        print('contour[0]',contour[0],'contour[1]',contour[1])
        count = count + 1
        cv2.putText(img_rgb, str(count), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 1,color, 1, 1)                                               #注1
        with open("data.txt", "a+") as f:                    #注2
            f.write("contour" + '\t' + str(count) + '\t' + 'cx: ' + str(cx) + '  \t' + 'cy: ' + str(cy) + '\n')
            print("contour" + '\t' + str(count) + '\t' + 'cx: ' + str(cx) + '  \t' + 'cy: ' + str(cy) + '\n')
            # 显示图像
    if Angle >= 0 and Angle <= 0.5 * math.pi:
        cx1 = int(contour[0] + h1 * math.sin(Angle) + w1 * 0.5 * math.cos(Angle))
        cy1 = int(contour[1] + w1 * 0.5 * math.sin(Angle))
        print('1:',h1 * math.sin(Angle),'2:',w1 * 0.5 * math.cos(Angle))
        print(Angle)
        print(math.sin(Angle * math.pi / 180))
    elif Angle <= math.pi:
        cx1 = int(contour[0] + h1 * math.sin(Angle) - w1 * 0.5 * math.cos(Angle))
        cy1 = int(contour[1] + w1 * 0.5 * math.sin(Angle) - h1 *  math.cos(Angle))
    elif Rotate_angle <= 1.5 * math.pi:
        cx1 = int(contour[0] - w1 * 0.5 * math.cos(Angle))
        cy1 = int(contour[1] - w1 * 0.5 * math.sin(Angle) - h1 * math.cos(Angle))
    else:
        cx1 = int(contour[0] + w1 * 0.5 * math.cos(Angle))
        cy1 = int(contour[1] - w1 * 0.5 * math.sin(Angle))
    cv2.circle(img_rgb, (contour[0],contour[1]), 1, (0,0,255), 4)    
    cv2.circle(img_rgb, (cx1,cy1), 1, (0,0,255), 4)
    print('cx1',cx1,'cy1',cy1)
    cv2.circle(img_rgb, (cx,cy), 1, (0,0,255), 4)
    cv2.imwrite("after.png", img_rgb)
    cv2.imshow("123", img_rgb)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__=='__main__':
    #a+可读可写覆盖
    with open("data.txt", "a+") as f:
        f.write('contour1'+'\n')
    threshold1 = 0.58
    contours1 = make_contour(template1,w1,h1,1,threshold1)
    color1 = (255, 0, 0)
    draw_contour(contours1,color1)

    img_rgb = cv2.imread("after.png")