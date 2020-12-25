#include <iostream>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/opencv.hpp>
#include "libmodbus/modbus.h"
#pragma comment(lib,"modbus.lib")  //��һ��Ҳ����ͨ��Project->properties->linker-
//>input->additional additional dependencies����õ���lib

using namespace std;
using namespace cv;

void homography(cv::Mat image, cv::Mat Opened);

int main()
{
	modbus_t* mb;
	uint16_t tab_reg[10] = { 0 };

	int nPort = 600;
	const char* chIp = "192.168.2.198";
	mb = modbus_new_tcp(chIp, nPort);

//	mb = modbus_new_rtu("COM2", 9600, 'N', 8, 1);   //��ͬ�Ķ˿�ֻ��ͬʱ��һ�� 
	modbus_set_slave(mb, 1);  //����modbus�ӻ���ַ 

	modbus_connect(mb);

	struct timeval t;
	t.tv_sec = 0;
	t.tv_usec = 1000000;   //����modbus��ʱʱ��Ϊ1000���� 
	modbus_set_response_timeout(mb, (int)&t.tv_sec, (int)&t.tv_usec);

	Point2f target;  //�����޹ؼ���

	float angle = 0; //һ�����ݵĽǶ�
	Vec3f c; //����Բ������Ͱ뾶,�������޵�����

	VideoCapture cap(0); 
	//��������粻�ܴ򿪣��˳�����
	if (!cap.isOpened()) {
		std::cout << "Cannot open the web cam" << std::endl;
		return -1;
	}

	cout << "1���뽫���ָ��ͼƬ���ں󣬰������ϵġ�p����ʵ�ֱ궨����Ƭ���㡣" << endl;
	cout << "��ע����ʱ�����Ե������������������������ʵ������)" << endl;
	cout << " " << endl;
	//����һ����Ƭ
	while (true) {

		char key = (char)cv::waitKey(30);

		Mat imgOriginal;
		bool bSuccess = cap.read(imgOriginal); // read a new frame from video
		if (!bSuccess) {
			//���㲻�ɹ����˳�
			cout << "Cannot read a frame from video stream" << endl;
			return -1;
		}
		//��ʾ�������ͼƬ
		imshow("pic", imgOriginal);

		Mat img;
		img = imgOriginal.clone();

		Mat image2hsv, bgr;
		img.convertTo(bgr, CV_32FC3, 1.0 / 255, 0);
		cvtColor(bgr, image2hsv, cv::COLOR_BGR2HSV);
		//��ɫͼ��ҶȻ�
		Mat image2gray;
		cvtColor(img, image2gray, cv::COLOR_RGB2GRAY);
		//ͼ����ֵ�˲�
		medianBlur(image2gray, image2gray, 3);

		vector<cv::Vec3f> circles;
		//���Բ��
		HoughCircles(image2gray, circles, CV_HOUGH_GRADIENT, 1, 10, 200, 100, 6, 220);
		int max = 0;
		//cv::Vec3f c;
		for (size_t i = 0; i < circles.size(); i++) {
			if (circles[i][2] > max) {
				max = circles[i][2];
				c = circles[i];
			}
		}
		Mat img1;
		img1 = img.clone();
		circle(img1, Point2f(c[0], c[1]), c[2], cv::Scalar(0, 255, 255), 3, CV_AA);
		circle(img1, Point2f(c[0], c[1]), 3, cv::Scalar(0, 255, 255), 3, CV_AA);
		imshow("img1", img1);
		if (key == 'p') {
			imwrite("1.jpg", imgOriginal);
			break;
		}
	}
	cout << "���������ģ�" << c[0] << " , " << c[1] << endl;

	//����һ�����ڣ����ڵ���HSV����
	cv::namedWindow("Control", CV_WINDOW_AUTOSIZE); //create a window called "Control"

	int iLowH = 32, iHighH = 55, iLowS = 0, iHighS = 255, iLowV = 0, iHighV = 18;
	cvCreateTrackbar("LowH", "Control", &iLowH, 255); //Hue (0 - 255)
	cvCreateTrackbar("HighH", "Control", &iHighH, 255);
	cvCreateTrackbar("LowS", "Control", &iLowS, 255); //Saturation (0 - 255)
	cvCreateTrackbar("HighS", "Control", &iHighS, 255);
	cvCreateTrackbar("LowV", "Control", &iLowV, 255); //Value (0 - 255)
	cvCreateTrackbar("HighV", "Control", &iHighV, 255);

	Mat srcImage = imread("1.jpg");
	while (true) {
		char key = (char)cv::waitKey(30);
		if (key == 'q') break;

		Mat image2hsv, bgr;
		srcImage.convertTo(bgr, CV_32FC3, 1.0 / 255, 0);
		cvtColor(bgr, image2hsv, cv::COLOR_BGR2HSV);

		//��ֵ��ͼ��,����HSV����
		Mat mask;
		inRange(image2hsv, cv::Scalar(iLowH, iLowS, iLowV), cv::Scalar(iHighH, iHighS, iHighV), mask);

		//��mask���������̬ѧ-�����㣬����������
		Mat mask_Opened = cv::Mat::zeros(mask.size(), CV_8U);
		homography(mask, mask_Opened);

		Mat img1;
		img1 = srcImage.clone();

		imshow("mask", mask); //show the thresholded image
		imshow("mask_Opened", mask_Opened);

		//Ѱ��mask_opened��ͨ���򣬱�������δ�ͨ����״����ȷ��Ŀ������
		//contours���ڱ�������������Ϣ,hierarchy���ڼ�¼����֮��ļ̳й�ϵ
		vector<std::vector<cv::Point>> contours;
		vector<cv::Vec4i> hierarchy;
		findContours(mask_Opened, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_NONE);

		int idx = 0;
		int idx_num = contours.size();  //����������
		int idx_left = idx_num;         //ɸѡ��ʣ����������
		//ɸѡʣ�µ�������Ҫ�������޽������ӵ�����
		RotatedRect rect1, rect2;
		Point2f P[4], P1[4];

		for (int i = 0; i < idx_num; i++) {
			//��С��Ӿ���
			rect2 = cv::minAreaRect(contours[i]);
			rect2.points(P1);

			int length1 = pow(pow(P1[0].x - P1[1].x, 2) + pow(P1[0].y - P1[1].y, 2), 0.5); //����һ���ߵĳ��� P[0]-- P[1]
			int length2 = pow(pow(P1[2].x - P1[1].x, 2) + pow(P1[2].y - P1[1].y, 2), 0.5); // ������һ���ߵĳ��� P[1]--P[2]

			//������С��Ӿ��ε�Բ�ľ��룬��ҪС��Բ�İ뾶
			int distanceA = pow(P1[0].x - c[0], 2) + pow(P1[0].y - c[1], 2);
			int distanceB = pow(P1[1].x - c[0], 2) + pow(P1[1].y - c[1], 2);
			int distanceC = pow(P1[2].x - c[0], 2) + pow(P1[2].y - c[1], 2);
			int distanceD = pow(P1[3].x - c[0], 2) + pow(P1[3].y - c[1], 2);

			//������С(��������)�������Լ��
			if (contours[i].size() < 100 || length1 < 0.2 * length2 || length1 > 5 * length2 || length1 * length2 < 500 || length1 * length2 > 6000) {
				idx_left--;
				continue;
			}
			//������С��Ӿ��ε�Բ�ľ��룬��ҪС��Բ�İ뾶
			if (distanceA > c[2] * c[2] || distanceB > c[2] * c[2] || distanceC > c[2] * c[2] || distanceD > c[2] * c[2]) {
				idx_left--;
				continue;
			}
			//�����������и�������˵���������Ҳ�������ǵ�Ŀ�꣬ɾ��֮
			if (hierarchy[i][3] != -1) {
				idx_left--;
				continue;
			}
			rect1 = cv::minAreaRect(contours[i]);
			rect1.points(P);
		}
		if (idx_left == 1) {
			//������Ӿ���;
			for (int j = 0; j <= 3; j++) {
				line(img1, P[j], P[(j + 1) % 4], cv::Scalar(0, 0, 255), 1);
			}

			//�ҳ��̱��Լ�Ҫ�����Ǹ�������꣬ȡ����Բ�Ľ�Զ��Ķ̱��е�ΪĿ��㣬������Ƕ�
			int length1_P = pow(P[0].x - P[1].x, 2) + pow(P[0].y - P[1].y, 2);
			int length2_P = pow(P[1].x - P[2].x, 2) + pow(P[1].y - P[2].y, 2);

			if (length1_P > length2_P) {
				if ((pow(P[1].x + P[2].x - 2 * c[0], 2) + pow(P[1].y + P[2].y - 2 * c[1], 2)) > (pow(P[0].x + P[3].x - 2 * c[0], 2) + pow(P[0].y + P[3].y - 2 * c[1], 2))) {
					target.x = (P[1].x + P[2].x) / 2;
					target.y = (P[1].y + P[2].y) / 2;
				}
				else {
					target.x = (P[0].x + P[3].x) / 2;
					target.y = (P[0].y + P[3].y) / 2;
				}
			}
			else {
				if ((pow(P[0].x + P[1].x - 2 * c[0], 2) + pow(P[0].y + P[1].y - 2 * c[1], 2)) > (pow(P[2].x + P[3].x - 2 * c[0], 2) + pow(P[2].y + P[3].y - 2 * c[1], 2))) {
					target.x = (P[0].x + P[1].x) / 2;
					target.y = (P[0].y + P[1].y) / 2;
				}
				else {
					target.x = (P[2].x + P[3].x) / 2;
					target.y = (P[2].y + P[3].y) / 2;
				}
			}
			//�����������
			float P_x = (float)(P[0].x + P[1].x + P[2].x + P[3].x) / 4;
			float P_y = (float)(P[0].y + P[1].y + P[2].y + P[3].y) / 4;
			//����б��
			float k = 0;
			circle(img1, cv::Point(target.x, target.y), 3, cv::Scalar(0, 255, 255), 3, CV_AA);
			if (target.x > P_x) {
				float k = (float)(target.y - P_y) / (float)(target.x - P_x);
				angle = atan(k) * 180 / CV_PI;
				if (angle < 0)
					angle = angle + 360;
			} else if (target.x == P_x) {
				if (target.y > P_y)
					angle = 90;
				else
					angle = 270;
			} else {
				float k = (float)(target.y - P_y) / (float)(target.x - P_x);
				angle = 180 + atan(k) * 180 / CV_PI;
			}		
		}
		imshow("result", img1);
		break;
	}
	cout << "Target: " << " x:" << target.x << " y:" << target.y << endl;
	cout << "angle: " << angle << endl;

	tab_reg[0] = c[0];
	tab_reg[1] = c[1];
	tab_reg[2] = angle;
	cout << "�������������꣺" << tab_reg[0] << " " << tab_reg[1] << endl;
	cout << "��Ҫ��ת�ĽǶȣ�" << tab_reg[2] << endl;
	waitKey(0);
	while (1) {
		int nRet = modbus_write_registers(mb, 0, 3, tab_reg);
	}
	return 0;
}

//��̬ѧ�˲�
void homography(cv::Mat image, cv::Mat Opened)
{
	cv::Mat element_9(9, 9, CV_8U, cv::Scalar(1));
	cv::Mat element_3(3, 3, CV_8U, cv::Scalar(1));
	cv::Mat element_2(2, 2, CV_8U, cv::Scalar(1));
	cv::Mat element_4(4, 4, CV_8U, cv::Scalar(1));
	cv::Mat element_5(5, 5, CV_8U, cv::Scalar(1));
	cv::Mat element_6(6, 6, CV_8U, cv::Scalar(1));
	cv::Mat element_7(7, 7, CV_8U, cv::Scalar(1));
	cv::Mat element_8(8, 8, CV_8U, cv::Scalar(1));
	cv::Mat element_1(1, 1, CV_8U, cv::Scalar(1));
	cv::morphologyEx(image, Opened, cv::MORPH_OPEN, element_3);
	cv::morphologyEx(Opened, Opened, cv::MORPH_CLOSE, element_3);
	//	cv::dilate(Opened, Opened, element_3);
}



//#include <iostream>
//#include "opencv2/highgui/highgui.hpp"
//#include "opencv2/imgproc/imgproc.hpp"
//#include <opencv2/highgui/highgui_c.h>
//
//void homography(cv::Mat image, cv::Mat Opened);
//
//int main()
//{
//	int num_read = 0;
//	cv::Point2f key;  //�����޹ؼ���
//	cv::Point2i main_point;//���������ĵ�
//	float Angle = 0; //��������ת�Ƕ�
//	float angle = 0; //һ�����ݵĽǶ�
//	cv::Vec3f c; //����Բ������Ͱ뾶
//	std::vector<float> angle_measure;//���루����angle����
//
//	cv::VideoCapture cap(0); //capture the video from web cam
//
//	if (!cap.isOpened())  // if not success, exit program
//	{
//		std::cout << "Cannot open the web cam" << std::endl;
//		return -1;
//	}
//	cv::namedWindow("Control", CV_WINDOW_AUTOSIZE); //create a window called "Control"
//
//	int iLowH = 32, iHighH = 55, iLowS = 0, iHighS = 255, iLowV = 0, iHighV = 18;
//	Create trackbars in "Control" window
//	cvCreateTrackbar("LowH", "Control", &iLowH, 255); //Hue (0 - 255)
//	cvCreateTrackbar("HighH", "Control", &iHighH, 255);
//	cvCreateTrackbar("LowS", "Control", &iLowS, 255); //Saturation (0 - 255)
//	cvCreateTrackbar("HighS", "Control", &iHighS, 255);
//	cvCreateTrackbar("LowV", "Control", &iLowV, 255); //Value (0 - 255)
//	cvCreateTrackbar("HighV", "Control", &iHighV, 255);
//
//	while(true) {
//		cv::Mat imgOriginal;
//		��ֵ��ͼ��
//		cv::Mat mask;
//		bool bSuccess = cap.read(imgOriginal); // read a new frame from video
//		if (!bSuccess) {
//			if not success, break loop
//			std::cout << "Cannot read a frame from video stream" << std::endl;
//			break;
//		}
//		cv::Mat image2hsv, bgr;
//		imgOriginal.convertTo(bgr, CV_32FC3, 1.0 / 255, 0);
//		cv::cvtColor(bgr, image2hsv, cv::COLOR_BGR2HSV);
//		��ɫͼ��ҶȻ�
//		cv::Mat image2gray;
//		cv::cvtColor(imgOriginal, image2gray, cv::COLOR_RGB2GRAY);
//		ͼ����ֵ�˲�
//		cv::medianBlur(image2gray, image2gray, 3);
//
//		��ֵ��ͼ��,����HSV����
//		cv::inRange(image2hsv, cv::Scalar(iLowH, iLowS, iLowV), cv::Scalar(iHighH, iHighS, iHighV), mask);
//
//		step4:��mask���������̬ѧ-�����㣬����������
//		cv::Mat mask_Opened = cv::Mat::zeros(mask.size(), CV_8U);
//		homography(mask, mask_Opened);
//
//		cv::namedWindow("mask_Opened", 1);
//		cv::imshow("mask_Opened", mask_Opened);
//
//		std::vector<cv::Vec3f> circles;
//		���Բ��
//		cv::HoughCircles(image2gray, circles, CV_HOUGH_GRADIENT, 1, 10, 200, 100, 6, 220);
//		int max = 0;
//		cv::Vec3f c;
//		for (size_t i = 0; i < circles.size(); i++) {
//			if (circles[i][2] > max) {
//				max = circles[i][2];
//				c = circles[i];
//				main_point.x = c[0];
//				main_point.y = c[1];
//			}			
//		}
//		std::cout << "Բ" << circles.size() << std::endl;
//		�ҳ������ޣ���������λ��
//		circle(imgOriginal, main_point, c[2], cv::Scalar(0, 255, 255), 3, CV_AA);
//		circle(imgOriginal, main_point, 3, cv::Scalar(0, 255, 255), 3, CV_AA);
//																										   
//		step5��Ѱ��mask_opened��ͨ���򣬱�������δ�ͨ����״����ȷ��Ŀ������
//		contours���ڱ�������������Ϣ,hierarchy���ڼ�¼����֮��ļ̳й�ϵ
//		std::vector<std::vector<cv::Point>> contours;
//		std::vector<cv::Vec4i> hierarchy;
//		cv::findContours(mask_Opened, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_NONE);
//
//		int idx = 0;
//		int idx_num = contours.size();  //����������
//		int idx_left = idx_num;         //ɸѡ��ʣ����������
//		ɸѡʣ�µ�������Ҫ�������޽������ӵ�����
//		cv::RotatedRect rect1,rect2;
//		cv::Point2f P[4],P1[4];
//
//		for(int i = 0; i < idx_num; i++) {
//			��С��Ӿ���
//			rect2 = cv::minAreaRect(contours[i]);
//			rect2.points(P1);
//
//			int length1 = pow(pow(P1[0].x - P1[1].x, 2) + pow(P1[0].y - P1[1].y, 2), 0.5); //����һ���ߵĳ��� P[0]-- P[1]
//			int length2 = pow(pow(P1[2].x - P1[1].x, 2) + pow(P1[2].y - P1[1].y, 2), 0.5); // ������һ���ߵĳ��� P[1]--P[2]
//
//			������С��Ӿ��ε�Բ�ľ��룬��ҪС��Բ�İ뾶
//			int distanceA = pow(P1[0].x - c[0], 2) + pow(P1[0].y - c[1], 2);
//			int distanceB = pow(P1[1].x - c[0], 2) + pow(P1[1].y - c[1], 2);
//			int distanceC = pow(P1[2].x - c[0], 2) + pow(P1[2].y - c[1], 2);
//			int distanceD = pow(P1[3].x - c[0], 2) + pow(P1[3].y - c[1], 2);
//
//			������С(��������)�������Լ��
//			if (contours[i].size() < 100 || length1 < 0.2 * length2 || length1 > 5 * length2 || length1 * length2 < 500 || length1 * length2 > 6000) {
//				idx_left--;
//				continue;
//			}
//			������С��Ӿ��ε�Բ�ľ��룬��ҪС��Բ�İ뾶
//			if (distanceA > c[2] * c[2] || distanceB > c[2] * c[2] || distanceC > c[2] * c[2] || distanceD > c[2] * c[2]) {
//				idx_left--;
//				continue;
//			}
//
//			�����������и�������˵���������Ҳ�������ǵ�Ŀ�꣬ɾ��֮
//			if(hierarchy[i][3] != -1) {
//				idx_left--;
//				continue;
//			}
//			rect1 = cv::minAreaRect(contours[i]);
//			rect1.points(P);
//		}
//		if (idx_left == 1) {
//			������Ӿ���;
//			for (int j = 0; j <= 3; j++) {
//				cv::line(imgOriginal, P[j], P[(j + 1) % 4], cv::Scalar(0, 0, 255), 1);
//			}
//
//			�ҳ��̱��Լ�Ҫ�����Ǹ�������꣬ȡ����Բ�Ľ�Զ��Ķ̱��е�ΪĿ��㣬������Ƕ�
//			int length1_P = pow(P[0].x - P[1].x, 2) + pow(P[0].y - P[1].y, 2);
//			int length2_P = pow(P[1].x - P[2].x, 2) + pow(P[1].y - P[2].y, 2);
//				
//			if (length1_P > length2_P) {
//				if ((pow(P[1].x + P[2].x - 2 * c[0], 2) + pow(P[1].y + P[2].y - 2 * c[1], 2)) > (pow(P[0].x + P[3].x - 2 * c[0], 2) + pow(P[0].y + P[3].y - 2 * c[1], 2))) {
//					key.x = (P[1].x + P[2].x) / 2;
//					key.y = (P[1].y + P[2].y) / 2;
//				} else {
//					key.x = (P[0].x + P[3].x) / 2;
//					key.y = (P[0].y + P[3].y) / 2;
//				}
//			} else {
//				if ((pow(P[0].x + P[1].x - 2 * c[0], 2) + pow(P[0].y + P[1].y - 2 * c[1], 2)) > (pow(P[2].x + P[3].x - 2 * c[0], 2) + pow(P[2].y + P[3].y - 2 * c[1], 2))) {
//					key.x = (P[0].x + P[1].x) / 2;
//					key.y = (P[0].y + P[1].y) / 2;
//				} else {
//					key.x = (P[2].x + P[3].x) / 2;
//					key.y = (P[2].y + P[3].y) / 2;
//				}
//			}
//			�����������
//			int P_x = (P[0].x + P[1].x + P[2].x + P[3].x) / 4;
//			int P_y = (P[0].y + P[1].y + P[2].y + P[3].y) / 4;
//			����б��
//			float k = 0;
//			circle(imgOriginal, cv::Point(key.x, key.y), 3, cv::Scalar(0, 255, 255), 3, CV_AA);
//			if (key.x > P_x) {
//				float k = (float)(key.y - P_y) / (float)(key.x - P_x);
//				angle = atan(k) * 180 / CV_PI;
//				if (angle < 0)
//					angle = angle + 360;
//			} else if (key.x == P_x) {
//				if (key.y > P_y)
//					angle = 90;
//				else
//					angle = -90;
//			} else {
//				float k = (float)(key.y - P_y) / (float)(key.x - P_x);
//				angle = 180 + atan(k) * 180 / CV_PI;
//			}
//			std::cout <<"k:"<< k << std::endl;
//			std::cout << key.x <<" " << key.y << "    " << P_x <<"  "<< P_y << std::endl;
//
//			����õ���̬���ǶȷŽ������У�����������С��һ�����ݵ�ɾ��
//			angle_measure.push_back(angle);
//
//			�޸���ֵʱ���޸������num_read������if(num_read < 0),
//			ȡ50�����ݼ���
//			if (num_read < 0) {
//			if (num_read == 25) {
//				����
//				sort(angle_measure.begin(), angle_measure.end());
//				float angle_sum = 0;
//				for (int i = 10; i < angle_measure.size()- 10; i++) {
//					angle_sum += angle_measure[i];
//				}
//				Angle = angle_sum / (angle_measure.size() - 20);
//				std::cout << "Angle:" << Angle <<std::endl;
//				break;
//			}
//			num_read++;
//		}
//			std::cout << "the total contours:" << idx_num << std::endl;
//		
//		std::cout << "the left contours:" << idx_left << std::endl;
//		std::cout << "angle:" << angle << std::endl;
//		cv::namedWindow("origin", 1);
//		cv::imshow("origin", imgOriginal);
//		cv::namedWindow("Thresholded Image", 1);
//		cv::imshow("Thresholded Image", mask); //show the thresholded image
//
//
//		char key = (char)cv::waitKey(300);
//		if (key == 27) break;
//
//	}
//
//	return 0;
//}
