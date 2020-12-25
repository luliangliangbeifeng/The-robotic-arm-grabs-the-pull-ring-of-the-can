import cv2

cap = cv2.VideoCapture(0)
i = 20
while(1):
    # 获得图片
    ret, frame = cap.read()
    # 展示图片
    cv2.imshow("capture", frame)
    if cv2.waitKey(1) & 0xFF == ord('p'):
        # 存储图片
        cv2.imwrite('C:\\Users\\86150\\Desktop\\project_01\\pic\\%d.jpg'%i, frame)
        i += 1
    if (cv2.waitKey(1) & 0xFF) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

