#include <iostream>
#include <fstream>

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/video/video.hpp>
#include <opencv2/opencv.hpp>


using namespace cv;
using namespace std;


int main(int argv, char** argc)
{
	VideoCapture cap(0); //打开相机

    if (!cap.isOpened())  // 无相机，退出工程
    {
        std::cout << "Cannot open the web cam" << std::endl;
        return -1;
    }
	cout << "1、请将鼠标指向图片窗口后，按键盘上的“p”键实现标定板照片拍摄。" << endl;
	cout << "（注：此时不可以点击键盘上其他按键，否则不能实现拍摄)" << endl;
	cout << " " << endl;
	//拍摄一张标定板图片
    while (true) {

        Mat imgOriginal;
        bool bSuccess = cap.read(imgOriginal); // read a new frame from video
        if (!bSuccess) {
            //拍摄不成功，退出
            cout << "Cannot read a frame from video stream" << endl;
            //return -1;
			continue;
        }
        //显示相机拍摄图片
        imshow("pic", imgOriginal);

		char key = (char)cv::waitKey(30);
        if (key == 'p') {
        	imwrite("sample.jpg", imgOriginal);
			break;
        }
    }

	Mat srcImage = imread("sample.jpg");
	
	//打开拍摄的图片，如若不能打开，退出工程
	if (srcImage.empty())
	{
		printf("could not load image..\n");
		return false;
	}
	Mat srcgray, dstImage, normImage, scaledImage;

	cvtColor(srcImage, srcgray, CV_BGR2GRAY);

	Mat srcbinary;
	threshold(srcgray, srcbinary, 0, 255, THRESH_OTSU | THRESH_BINARY);

	Mat kernel = getStructuringElement(MORPH_RECT, Size(15, 15), Point(-1, -1));
	morphologyEx(srcbinary, srcbinary, MORPH_OPEN, kernel, Point(-1, -1));

	vector<Point2f> corners;
	Size patternSize = Size(8, 11);
	findChessboardCorners(srcgray, patternSize,  corners);

	//寻找亚像素角点
	Size winSize = Size(5, 5);  //搜素窗口的一半尺寸
	Size zeroZone = Size(-1, -1);//表示死区的一半尺寸
	//求角点的迭代过程的终止条件，即角点位置的确定
	TermCriteria criteria = TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 40, 0.001);

	cornerSubPix(srcgray, corners, winSize, zeroZone, criteria);


	//绘制角点
	for (int i = 0; i < corners.size(); i++)
	{
		char temp[16];
		sprintf_s(temp, "%d", i+1);
		putText(srcImage, temp, corners[i], FONT_HERSHEY_SCRIPT_SIMPLEX, 0.5, Scalar(0, 255, 0), 2, LINE_AA, false);
		circle(srcImage, corners[i], 2, Scalar(255, 0, 0), -1, 8, 0);
	}

	cout << "第1个角点的像素坐标，：" << corners[0] << endl;
	cout << "第4个角点的像素坐标，：" << corners[3] << endl;
	cout << "第8个角点的像素坐标，：" << corners[7] << endl;
	cout << "第49个角点的像素坐标，：" << corners[48] << endl;
	cout << "第52个角点的像素坐标，：" << corners[51] << endl;
	cout << "第56个角点的像素坐标，：" << corners[55] << endl;
	cout << "第81个角点的像素坐标，：" << corners[80] << endl;
	cout << "第84个角点的像素坐标，：" << corners[83] << endl;
	cout << "第88个角点的像素坐标，：" << corners[87] << endl;

	
	cout << " " << endl;
	cout << "2、请将鼠标指向图片窗口后，按键盘上的“q”键后，依次移动标定针在点（1、4、8、49、52、56、81、84、88）正上方时机械臂的坐标。" << endl;
	cout << "(注：机械臂坐标只需要输入X和Y两个坐标即可，X和Y之间输入通过空格区分，如输入点（1，1），就输入'1','空格','1')" << endl;
	cout << " " << endl;
	//显示标定板角点信息
	while (true) {
		char key = (char)cv::waitKey(30);

		imshow("角点检测", srcImage);
		if (key == 'q') {
//			destroyWindow("pic");
			break;
		}
	}

	/* 九点标定*/
    double A, B, C, D, E, F;
	Mat warpMat;
	vector<Point2f>points_camera;
	vector<Point2f>points_robot;

	points_camera.push_back(corners[0]);
	points_camera.push_back(corners[3]);
	points_camera.push_back(corners[7]);
	points_camera.push_back(corners[48]);
	points_camera.push_back(corners[51]);
	points_camera.push_back(corners[55]);
	points_camera.push_back(corners[80]);
	points_camera.push_back(corners[83]);
	points_camera.push_back(corners[87]);

	for (int i = 0; i < 9; i++) {
		int index;
		if (i == 0) index = 1;
		if (i == 1) index = 4;
		if (i == 2) index = 8;
		if (i == 3) index = 49;
		if (i == 4) index = 52;
		if (i == 5) index = 56;
		if (i == 6) index = 81;
		if (i == 7) index = 84;
		if (i == 8) index = 88;
		cout << "标定针在第" << index << "个角点正上方的机械臂坐标：";
		double x, y;
		cin >> x >> y;
		points_robot.push_back(Point2f(x, y));

	}

	warpMat = estimateRigidTransform(points_camera, points_robot, true);
	Mat estimateRigidTransform(InputArray src, InputArray dst, bool fullAffine);
	cout << " " << endl;
    cout <<"3、机械臂坐标与像素坐标的变换关系：" << warpMat << endl;

	A = warpMat.ptr<double>(0)[0];
	B = warpMat.ptr<double>(0)[1];
	C = warpMat.ptr<double>(0)[2];
	D = warpMat.ptr<double>(1)[0];
	E = warpMat.ptr<double>(1)[1];
	F = warpMat.ptr<double>(1)[2];
	cout << "A = " << A << endl;
	cout << "B = " << B << endl;
	cout << "C = " << C << endl;
	cout << "D = " << D << endl;
	cout << "E = " << E << endl;
	cout << "F = " << F << endl;
	cout << " " << endl;
	cout << "九点标定工作完成，后续只要相机和机械臂位置没有发生改变，就不再需要重新标定了。" << endl;

	ofstream dataFile;
	dataFile.open("caliberation_camera.txt", ofstream::app);
	// 朝TXT文档中写入数据
	dataFile <<"机械臂坐标与像素坐标转换矩阵关系：" << "\n" << "A:" << A << "\n" << "B:" << B << "\n" 
		<< "C:" << C << "\n" << "D:" << D << "\n" << "E:" << E << "\n" << "F:" << F << endl;
	// 关闭文档
	dataFile.close();

	cout << "以第2、10、11号点为例进行验证" << endl;
	cout << "2号点的坐标，x:" << A * corners[1].x + B * corners[1].y + C << "  y:" << D * corners[1].x + E * corners[1].y + F << endl;
	cout << "10号点的坐标，x:" << A * corners[9].x + B * corners[9].y + C << "  y:" << D * corners[9].x + E * corners[9].y + F << endl;
	cout << "11号点的坐标，x:" << A * corners[10].x + B * corners[10].y + C << "  y:" << D * corners[10].x + E * corners[10].y + F << endl;

	waitKey(0);
	return(0);

}
