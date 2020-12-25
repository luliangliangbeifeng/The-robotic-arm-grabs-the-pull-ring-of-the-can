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
	VideoCapture cap(0); //�����

    if (!cap.isOpened())  // ��������˳�����
    {
        std::cout << "Cannot open the web cam" << std::endl;
        return -1;
    }
	cout << "1���뽫���ָ��ͼƬ���ں󣬰������ϵġ�p����ʵ�ֱ궨����Ƭ���㡣" << endl;
	cout << "��ע����ʱ�����Ե������������������������ʵ������)" << endl;
	cout << " " << endl;
	//����һ�ű궨��ͼƬ
    while (true) {

        Mat imgOriginal;
        bool bSuccess = cap.read(imgOriginal); // read a new frame from video
        if (!bSuccess) {
            //���㲻�ɹ����˳�
            cout << "Cannot read a frame from video stream" << endl;
            //return -1;
			continue;
        }
        //��ʾ�������ͼƬ
        imshow("pic", imgOriginal);

		char key = (char)cv::waitKey(30);
        if (key == 'p') {
        	imwrite("sample.jpg", imgOriginal);
			break;
        }
    }

	Mat srcImage = imread("sample.jpg");
	
	//�������ͼƬ���������ܴ򿪣��˳�����
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

	//Ѱ�������ؽǵ�
	Size winSize = Size(5, 5);  //���ش��ڵ�һ��ߴ�
	Size zeroZone = Size(-1, -1);//��ʾ������һ��ߴ�
	//��ǵ�ĵ������̵���ֹ���������ǵ�λ�õ�ȷ��
	TermCriteria criteria = TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 40, 0.001);

	cornerSubPix(srcgray, corners, winSize, zeroZone, criteria);


	//���ƽǵ�
	for (int i = 0; i < corners.size(); i++)
	{
		char temp[16];
		sprintf_s(temp, "%d", i+1);
		putText(srcImage, temp, corners[i], FONT_HERSHEY_SCRIPT_SIMPLEX, 0.5, Scalar(0, 255, 0), 2, LINE_AA, false);
		circle(srcImage, corners[i], 2, Scalar(255, 0, 0), -1, 8, 0);
	}

	cout << "��1���ǵ���������꣬��" << corners[0] << endl;
	cout << "��4���ǵ���������꣬��" << corners[3] << endl;
	cout << "��8���ǵ���������꣬��" << corners[7] << endl;
	cout << "��49���ǵ���������꣬��" << corners[48] << endl;
	cout << "��52���ǵ���������꣬��" << corners[51] << endl;
	cout << "��56���ǵ���������꣬��" << corners[55] << endl;
	cout << "��81���ǵ���������꣬��" << corners[80] << endl;
	cout << "��84���ǵ���������꣬��" << corners[83] << endl;
	cout << "��88���ǵ���������꣬��" << corners[87] << endl;

	
	cout << " " << endl;
	cout << "2���뽫���ָ��ͼƬ���ں󣬰������ϵġ�q�����������ƶ��궨���ڵ㣨1��4��8��49��52��56��81��84��88�����Ϸ�ʱ��е�۵����ꡣ" << endl;
	cout << "(ע����е������ֻ��Ҫ����X��Y�������꼴�ɣ�X��Y֮������ͨ���ո����֣�������㣨1��1����������'1','�ո�','1')" << endl;
	cout << " " << endl;
	//��ʾ�궨��ǵ���Ϣ
	while (true) {
		char key = (char)cv::waitKey(30);

		imshow("�ǵ���", srcImage);
		if (key == 'q') {
//			destroyWindow("pic");
			break;
		}
	}

	/* �ŵ�궨*/
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
		cout << "�궨���ڵ�" << index << "���ǵ����Ϸ��Ļ�е�����꣺";
		double x, y;
		cin >> x >> y;
		points_robot.push_back(Point2f(x, y));

	}

	warpMat = estimateRigidTransform(points_camera, points_robot, true);
	Mat estimateRigidTransform(InputArray src, InputArray dst, bool fullAffine);
	cout << " " << endl;
    cout <<"3����е����������������ı任��ϵ��" << warpMat << endl;

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
	cout << "�ŵ�궨������ɣ�����ֻҪ����ͻ�е��λ��û�з����ı䣬�Ͳ�����Ҫ���±궨�ˡ�" << endl;

	ofstream dataFile;
	dataFile.open("caliberation_camera.txt", ofstream::app);
	// ��TXT�ĵ���д������
	dataFile <<"��е����������������ת�������ϵ��" << "\n" << "A:" << A << "\n" << "B:" << B << "\n" 
		<< "C:" << C << "\n" << "D:" << D << "\n" << "E:" << E << "\n" << "F:" << F << endl;
	// �ر��ĵ�
	dataFile.close();

	cout << "�Ե�2��10��11�ŵ�Ϊ��������֤" << endl;
	cout << "2�ŵ�����꣬x:" << A * corners[1].x + B * corners[1].y + C << "  y:" << D * corners[1].x + E * corners[1].y + F << endl;
	cout << "10�ŵ�����꣬x:" << A * corners[9].x + B * corners[9].y + C << "  y:" << D * corners[9].x + E * corners[9].y + F << endl;
	cout << "11�ŵ�����꣬x:" << A * corners[10].x + B * corners[10].y + C << "  y:" << D * corners[10].x + E * corners[10].y + F << endl;

	waitKey(0);
	return(0);

}
