#include<opencv2/opencv.hpp>
#include<iostream>
#include<string>
#include <fstream>
#include <filesystem>
#include"math.h"
#include "CC.h"

#include <iostream>
#include <stdio.h>
#include <io.h>

#define _CRT_SECURE_NO_DEPRECATE
#define M_PI 3.14159265358979323846
#define  MAX_DOUBLE 99999999999999999

using namespace std;
using namespace cv;
namespace fs = std::filesystem;
using namespace cv::ml;

Mat capture_frame, filter_frame, gaussian_frame, threshold_frame, centredImage;

vector<vector<CC>> matriceCompClassifier;

// vecteur des symoboles
vector<String> items;

//matrice avec des labels des composants connexes
Mat Matlabled;

// vecteur des composants connexes
vector<CC> composants;

// vectors des gfd (vector's des characteristic)
vector< vector<vector<float>>> vecteursCarPrim;
vector<vector<float>> vecteursCar;

// variable tampon
vector<float> vectC;

bool balance_flag = false;
int kernel_size = 3;
int block_size = 3;
int c = 0;
double segma = 0;

//
int M = 4;
int N = 9;
// en moin 1 composant clasifier comme symbole qlq
int C = 1;

void capture(Mat& capture_frame, string path);
inline bool exists(const std::string& name);

void readOrLoad(int m, int n, String Extension);
void symbolTocomposantGfd(Mat& mat, int m, int n);
void filterNonInv(Mat& capture_frame, Mat& threshold_frame);
void filter(Mat& capture_frame, Mat& threshold_frame);
double distance(Point& p1, Point& p2);
double ManhattanDistance(vector<float>& a, vector<float>& b);
void circshift(Mat& out, const Point& delta);
Point GetCentroid(Mat& img);
void centreObject(Mat& img, Mat& centredImage);
void CcToMat(CC cc, Mat& img);
void connectedComponentsVector(Mat& threshold_frame, vector<CC>& composants);
double GetExtrema(Mat& img, Point& center);
vector<double> linspace(double min, double max, int n);

static void meshgrid(const cv::Mat& xgv, const cv::Mat& ygv, cv::Mat1d& X, cv::Mat1d& Y);
void GFD(CC& composant, Mat& centredImage, int m, int n);
void CalculateGfdAndPushAllVectsCar(int m, int n);
void classification();
void clean_SomeShit();

void drawComposant(CC& composant, Mat& sub);

void drawComposantsClassifier(vector<CC>& composantsDejaclassifier, Mat& sub);

Mat src_gray;
int thresh = 100;
RNG rng(12345);
Mat src;

int NmbrSymbole = 1;
int numImagesMaxParSymbole = 1;

double Seuil = 100;
int Seuil_distance = 100;
int connexité = 4;

string path_name_test = "esc1-Standard.jpg";

void hough() {
	// Declare the output variables
	Mat dst, cdst, cdstP;

	// Check if image is loaded fine
	if (src.empty()) {
		printf(" Error opening image\n");
		return;
	}
	// Edge detection
	Canny(src_gray, dst, 40, 140, 3);
	// Copy edges to the images that will display the results in BGR

	cvtColor(dst, cdst, COLOR_GRAY2BGR);
	cdstP = cdst.clone();
	// Standard Hough Line Transform
	vector<Vec2f> lines; // will hold the results of the detection
	HoughLines(dst, lines, 1, CV_PI / 180, 150, 0, 0); // runs the actual detection
	// Draw the lines
	for (size_t i = 0; i < lines.size(); i++)
	{
		float rho = lines[i][0], theta = lines[i][1];
		Point pt1, pt2;
		double a = cos(theta), b = sin(theta);
		double x0 = a * rho, y0 = b * rho;
		pt1.x = cvRound(x0 + 1000 * (-b));
		pt1.y = cvRound(y0 + 1000 * (a));
		pt2.x = cvRound(x0 - 1000 * (-b));
		pt2.y = cvRound(y0 - 1000 * (a));
		line(cdst, pt1, pt2, Scalar(0, 0, 255), 2, LINE_AA);
	}

	// Show results
	imshow("Source", src);
	imshow("Detected Lines" + path_name_test, cdst);
	imwrite("./Detected Lines" + path_name_test + ".jpg", cdst);
}

void thresh_callback(int, void*)
{
	Mat canny_output;
	Canny(src_gray, canny_output, thresh, thresh * 2);
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	findContours(canny_output, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);
	Mat drawing = Mat::zeros(canny_output.size(), CV_8UC3);
	for (size_t i = 0; i < contours.size(); i++)
	{
		Scalar color = Scalar(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256));
		drawContours(drawing, contours, (int)i, color, 2, LINE_8, hierarchy, 0);

		cout << "color here" << endl;
	}
	imshow("Contours", drawing);
}

void canny()
{
	if (src.empty())
	{
		cout << "Could not open or find the image!\n" << endl;

		return;
	}
	cvtColor(src, src_gray, COLOR_BGR2GRAY);
	blur(src_gray, src_gray, Size(3, 3));
	const char* source_window = "Source";
	namedWindow(source_window);
	imshow(source_window, src);
	const int max_thresh = 255;
	createTrackbar("Canny thresh:", source_window, &thresh, max_thresh, thresh_callback);
	thresh_callback(0, 0);
	waitKey();
}

int main()
{
	/*src = imread("./bdd/" + path_name_test + ".jpg");
	cvtColor(src, src_gray, COLOR_BGR2GRAY);
	threshold(src_gray, threshold_frame, 100, 255, cv::THRESH_BINARY);
	//canny();
	hough();*/

	capture_frame = imread("./bdd/HoughResults/" + path_name_test);

	cvtColor(capture_frame, capture_frame, COLOR_BGR2GRAY);

	for (int x = 0; x < capture_frame.rows; ++x)
	{
		uchar* row_ptr = capture_frame.ptr<uchar>(x);
		for (int y = 0; y < capture_frame.cols; ++y)
		{
			if (capture_frame.at<uchar>(x, y) >= 150) row_ptr[y] = 0;
		}
	}

	threshold(capture_frame, capture_frame, 40, 255, cv::THRESH_BINARY);

	imshow("capture_frame", capture_frame);
	imwrite("./clean_" + path_name_test, capture_frame);

	readOrLoad(M, N, ".png");
	filter(capture_frame, threshold_frame);

	connectedComponentsVector(threshold_frame, composants);

	CalculateGfdAndPushAllVectsCar(4, 9);

	classification();

	cout << "fin";

	waitKey(0);

	return 0;
}

void capture(Mat& capture_frame, string path) {
	capture_frame = imread(path);
}

inline bool exists(const std::string& name) {
	struct stat buffer;
	return (stat(name.c_str(), &buffer) == 0);
}

void filterNonInv(Mat& capture_frame, Mat& threshold_frame) {
	// capture frame, convert to grayscale, apply Gaussian blur, apply balance (if applicable), and apply adaptive threshold method
	cvtColor(capture_frame, filter_frame, COLOR_BGR2GRAY);
	GaussianBlur(filter_frame, gaussian_frame, cv::Size(kernel_size, kernel_size), segma, segma);
	threshold(gaussian_frame, threshold_frame, Seuil, 255, cv::THRESH_BINARY);
}

void filter(Mat& capture_frame, Mat& threshold_frame) {
	// capture frame, convert to grayscale, apply Gaussian blur, apply balance (if applicable), and apply adaptive threshold method
	//cvtColor(capture_frame, filter_frame, COLOR_BGR2GRAY);
	//GaussianBlur(filter_frame, gaussian_frame, cv::Size(kernel_size, kernel_size), segma, segma);
	threshold(capture_frame, threshold_frame, Seuil, 255, cv::THRESH_BINARY_INV);
}

double distance(Point& p1, Point& p2)
{
	double res;
	res = sqrt((p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p1.y));
	return res;
}

double ManhattanDistance(vector<float>& a, vector<float>& b) {
	double dist = 0;
	int i;
	for (i = 0; i < min(a.size(), b.size()); i++)
	{
		dist += abs(a.at(i) - b.at(i));
	}
	int j = i;
	while (i < a.size())
	{
		dist += abs(a.at(i));
		i++;
	}

	while (j < b.size())
	{
		dist += abs(b.at(j));
		j++;
	}

	return dist;
}

void circshift(Mat& out, const Point& delta)
{
	Size sz = out.size();

	assert(sz.height > 0 && sz.width > 0);

	if ((sz.height == 1 && sz.width == 1) || (delta.x == 0 && delta.y == 0))
		return;

	int x = delta.x;
	int y = delta.y;
	if (x > 0) x = x % sz.width;
	if (y > 0) y = y % sz.height;
	if (x < 0) x = x % sz.width + sz.width;
	if (y < 0) y = y % sz.height + sz.height;

	vector<Mat> planes;
	split(out, planes);

	for (size_t i = 0; i < planes.size(); i++)
	{
		Mat tmp0, tmp1, tmp2, tmp3;
		Mat q0(planes[i], Rect(0, 0, sz.width, sz.height - y));
		Mat q1(planes[i], Rect(0, sz.height - y, sz.width, y));
		q0.copyTo(tmp0);
		q1.copyTo(tmp1);
		if (tmp0.size() != Size(0, 0))
			tmp0.copyTo(planes[i](Rect(0, y, sz.width, sz.height - y)));
		if (tmp1.size() != Size(0, 0))
			tmp1.copyTo(planes[i](Rect(0, 0, sz.width, y)));

		Mat q2(planes[i], Rect(0, 0, sz.width - x, sz.height));
		Mat q3(planes[i], Rect(sz.width - x, 0, x, sz.height));
		q2.copyTo(tmp2);
		q3.copyTo(tmp3);
		if (tmp2.size() != Size(0, 0))
			tmp2.copyTo(planes[i](Rect(x, 0, sz.width - x, sz.height)));
		if (tmp3.size() != Size(0, 0))
			tmp3.copyTo(planes[i](Rect(0, 0, x, sz.height)));
	}

	merge(planes, out);
}

Point GetCentroid(Mat& img) {
	Moments m = moments(img, true);

	return  Point((int)m.m10 / m.m00, (int)m.m01 / m.m00);
}

void centreObject(Mat& img, Mat& centredImage) {
	int width = img.size().width;
	int height = img.size().height;

	auto type = img.type();

	Mat ligne = cv::Mat::zeros(1, width, type);
	Mat colonne = cv::Mat::zeros(height, 1, type);

	centredImage = img;

	int temp = (int)(std::max(width, height) - std::min(width, height)) * 0.5;

	if (height < width) {
		if ((temp % 1) > 0) {
			hconcat(centredImage, colonne, centredImage);
			ligne = cv::Mat::zeros(1, centredImage.size().width, type);
		}

		for (int i = 0; i < round(temp); i++)
		{
			vconcat(centredImage, ligne, centredImage);
		}
	}
	else if (width < height) {
		if ((temp % 1) > 0) {
			vconcat(centredImage, ligne, centredImage);
			colonne = cv::Mat::zeros(centredImage.size().height, 1, type);
		}
		for (int i = 0; i < round(temp); i++)
		{
			hconcat(centredImage, colonne, centredImage);
		}
	}

	Point state = GetCentroid(centredImage);

	width = centredImage.size().width;
	height = centredImage.size().height;

	int delta_y = round(height / 2 - state.y);
	int delta_x = round(width / 2 - state.x);
	int delta_max = max(abs(delta_y), abs(delta_x));

	colonne = cv::Mat::zeros(height, 1, type);

	for (int i = 0; i < delta_max + 10; i++) {
		hconcat(centredImage, colonne, centredImage);
		hconcat(colonne, centredImage, centredImage);
	}
	width = centredImage.size().width;
	ligne = cv::Mat::zeros(1, width, type);

	for (int i = 0; i < delta_max + 10; i++) {
		vconcat(ligne, centredImage, centredImage);
		vconcat(centredImage, ligne, centredImage);
	}

	circshift(centredImage, Point(delta_y, delta_x));
}

void CcToMat(CC cc, Mat& img) {
	int width = cc.getdX();
	int height = cc.getdY();

	img = cv::Mat(width, height, CV_8UC1);

	int longeur = Matlabled.size().width;

	Point deb = cc.getPtr_debut();

	for (int x = deb.x; x < deb.x + width; ++x)
	{
		uchar* row_ptr = img.ptr<uchar>(x - deb.x);

		for (int y = deb.y; y < deb.y + height; ++y)
		{
			if (Matlabled.at<uint16_t>(Point(x, y)) == cc.getId_label()) row_ptr[y - deb.y] = 255;
			else row_ptr[y - deb.y] = 0;
		}
	}

	img = img.t();
}

void connectedComponentsVector(Mat& threshold_frame, vector<CC>& composants) {
	Mat centroids;

	Mat stats;
	connectedComponentsWithStats(threshold_frame, Matlabled, stats, centroids, connexité, CV_16U);

	Mat m;
	composants.clear();

	for (int i = 0; i < stats.rows; i++)
	{
		CC composant;
		composant.setId_label(i);
		composant.setdX(stats.at<int>(i, 2));
		composant.setdY(stats.at<int>(i, 3));
		Point centroid((int)centroids.at<double>(i, 0), (int)centroids.at<double>(i, 1));
		composant.setCentroid(centroid);
		composant.setPtr_debut(Point(stats.at<int>(i, 0), stats.at<int>(i, 1)));

		CcToMat(composant, m);

		composant.setMat(m);

		composants.push_back(composant);

		imwrite("./CCs/composant_" + to_string(i) + ".jpg", composant.getMat());
	}
}

//still background
double GetExtrema(Mat& img, Point& center) {
	// Find contours
	vector<vector<Point>> cnts;

	cv::findContours(img, cnts, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

	//cnts = imutils.grab_contours(cnts);
	//auto c = std::max(cnts, contourArea);
	if (cnts.size() > 1) cout << "plusieurs contours";

	/*int min_x = cnts.at(0).at(0).x;
	int max_x = min_x;

	int min_y = cnts.at(0).at(0).y;
	int max_y = min_y;
	Point lefttop;
	Point leftbottom;
	Point rightbottom;
	Point righttop;

	for (Point p : cnts.at(0))
	{
		if (min_x >= p.x && min_y >= p.y) { lefttop = p;  min_x = p.x;  min_y = p.y; }
		if (min_x >= p.x && min_y < p.y) { leftbottom = p;  min_y = p.y; min_x = p.x;}
		if (max_x <= p.x && max_y <= p.y) { rightbottom = p;  max_x = p.x;  max_y = p.y;}
		if (max_x <= p.x && max_y > p.y) { righttop = p;  max_y = p.y; max_x = p.x;}
	}
	cout << lefttop;
	cout << leftbottom;
	cout << rightbottom;
	cout << righttop;
	vector<Point> extrema;
	extrema.push_back(lefttop);
	extrema.push_back(leftbottom);
	extrema.push_back(rightbottom);
	extrema.push_back(righttop);
	*/

	double maxRad = 0;
	for (Point p : cnts.at(0)) {
		if (maxRad < distance(center, p))  maxRad = distance(center, p);
	}

	return maxRad;
}

vector<double> linspace(double min, double max, int n)
{
	vector<double> result;

	int iterator = 0;

	for (int i = 0; i <= n - 2; i++)
	{
		double temp = min + i * (max - min) / (floor((double)n) - 1);
		result.insert(result.begin() + iterator, temp);
		iterator += 1;
	}

	result.insert(result.begin() + iterator, max);
	return result;
}

static void meshgrid(const cv::Mat& xgv, const cv::Mat& ygv,
	cv::Mat1d& X, cv::Mat1d& Y)
{
	cv::repeat(xgv.reshape(1, 1), ygv.total(), 1, X);
	cv::repeat(ygv.reshape(1, 1).t(), 1, xgv.total(), Y);
}

void GFD(CC& composant, Mat& centredImage, int m, int n)
{
	centreObject(composant.getMat(), centredImage);

	cout << "width : " << centredImage.size().width << endl;
	cout << "height : " << centredImage.size().height << endl;

	Point Centroid = GetCentroid(centredImage);

	cout << Centroid << endl;

	double maxRad = GetExtrema(centredImage, Centroid);

	double radius, tempR, tempI;
	double theta;

	int N = centredImage.size().height;

	vector<vector<double>> FR;
	FR.resize(m, vector<double>(n));
	vector<vector<double>> FI;
	FI.resize(m, vector<double>(n));

	for (int rad = 0; rad < m; rad++)
	{
		for (int ang = 0; ang < n; ang++)
		{
			for (int x = 0; x < centredImage.size().width; x++)
			{
				for (int y = 0; y < centredImage.size().height; y++)
				{
					radius = sqrt(std::pow(x - Centroid.x, 2) + std::pow(y - Centroid.y, 2)) / maxRad;
					theta = atan2((y - Centroid.y), (x - Centroid.x));
					if (theta < 0) theta += 2 * M_PI;

					tempR = centredImage.at<uchar>(Point(x, y)) * std::cos(2 * M_PI * rad * (radius)+ang * theta);
					tempI = centredImage.at<uchar>(Point(x, y)) * std::sin(2 * M_PI * rad * (radius)+ang * theta);

					FR.at(rad).at(ang) += tempR;
					FI.at(rad).at(ang) -= tempI;
				}
			}
		}
	}

	int taille = (m) * (n);
	vectC.clear();
	vectC.resize(taille);
	float DC;

	for (int rad = 0; rad < m; rad++)
	{
		for (int ang = 0; ang < n; ang++)
		{
			if (rad == 0 && ang == 0) {
				DC = sqrt(std::pow(FR.at(0).at(0), 2) + std::pow(FR.at(0).at(0), 2));

				vectC.at(0) = DC / (M_PI * std::pow(maxRad, 2));
			}
			else {
				int pos = rad * n + ang;
				vectC.at(pos) = sqrt(std::pow(FR.at(rad).at(ang), 2) + std::pow(FI.at(rad).at(ang), 2)) / DC;
			}
		}
	}
}

void CalculateGfdAndPushAllVectsCar(int m, int n) {
	vector<float>& ptr_vect = vectC;

	// i=0 => background

	for (int i = 1; i < composants.size(); i++)
	{
		cout << "calcul GFD Composant_" + to_string(i) << endl;
		GFD(composants.at(i), centredImage, m, n);

		//namedWindow("composantCentred : " + i, WINDOW_NORMAL);
		//imshow("composantCentred : " + i, centredImage);

		vecteursCar.push_back(ptr_vect);
	}
}

void clean_SomeShit() {
	std::error_code errorCode;
	std::filesystem::path dir = fs::current_path();

	std::filesystem::remove_all(dir / "CCs/", errorCode);
}

void drawComposant(CC& composant, Mat& sub) {
	//Mat sub = cv::Mat::zeros(100, 82, CV_8UC1);

	for (int x = composant.getPtr_debut().x; x < composant.getPtr_debut().x + composant.getdX(); ++x)
	{
		uchar* row_ptr = sub.ptr<uchar>(x);
		for (int y = composant.getPtr_debut().y; y < composant.getPtr_debut().y + composant.getdY(); ++y)
		{
			if (composant.getMat().at<uchar>(Point(x - composant.getPtr_debut().x, y - composant.getPtr_debut().y)) == 255) row_ptr[y] = 255;
		}
	}
}

void drawComposantsClassifier(vector<CC>& composantsDejaclassifier, Mat& sub) {
	for (int i = 0; i < composantsDejaclassifier.size(); i++)
	{
		drawComposant(composantsDejaclassifier.at(i), sub);
	}

	sub = sub.t();
}

void classification() {
	std::ofstream out("./resultat.txt");

	double min = MAX_DOUBLE;
	int symbole;
	double max = -MAX_DOUBLE;
	double dist = MAX_DOUBLE;

	matriceCompClassifier.clear();
	int N = vecteursCarPrim.size();
	matriceCompClassifier.resize(N, std::vector<CC>(C));

	for (int i = 0; i < vecteursCar.size(); i++)
	{
		cout << "composant_" << i + 1 << " =>";
		out << "composant_" << i + 1 << " =>";

		symbole = 0;

		for (int it = 0; it < vecteursCarPrim.at(0).size(); it++)
		{
			dist = std::min(dist, ManhattanDistance(vecteursCar.at(i), vecteursCarPrim.at(0).at(it)));
		}

		cout << items.at(0) << " " << dist << " | ";
		out << items.at(0) << " " << dist << " | ";

		max = dist;
		min = dist;

		for (int j = 1; j < vecteursCarPrim.size(); j++) {
			dist = MAX_DOUBLE;

			for (int it = 0; it < vecteursCarPrim.at(j).size(); it++)
			{
				dist = std::min(dist, ManhattanDistance(vecteursCar.at(i), vecteursCarPrim.at(j).at(it)));
			}

			if (dist <= min) {
				min = dist;
				symbole = j;
			}
			if (max <= dist) {
				max = dist;
			}

			cout << items.at(j) << " " << dist << " | ";
			out << items.at(j) << " " << dist << " | ";
		}

		cout << " => " << min;
		out << " => " << min;
		string decision;
		if (min < Seuil_distance) {
			decision = items.at(0);
		}
		else decision = "classe rejet";
		cout << " => " << decision;
		out << " => " << decision;

		cout << endl;
		out << endl;

		matriceCompClassifier.at(symbole).emplace_back(composants.at(i + 1));
	}

	Mat image = cv::Mat::zeros(capture_frame.size().width, capture_frame.size().height, CV_8UC1);
	for (int i = 0; i < matriceCompClassifier.size(); i++)
	{
		image = cv::Mat::zeros(capture_frame.size().width, capture_frame.size().height, CV_8UC1);
		drawComposantsClassifier(matriceCompClassifier.at(i), image);
		imwrite("./CCs Classifier/" + items.at(i) + ".jpg", image);
	}
}

void readOrLoad(int m, int n, String Extension) {
	items.clear();
	String numS;
	String path;
	vecteursCarPrim.clear();
	vecteursCarPrim.resize(NmbrSymbole, vector<std::vector<float>>(C));

	for (auto& p : fs::directory_iterator("Symboles")) {
		items.push_back(p.path().filename().string());
		for (int num = 0; num < numImagesMaxParSymbole; num++)
		{
			numS = "_" + to_string(num);
			path = "Symboles/" + p.path().filename().string() + "\/" + p.path().filename().string() + numS;

			if (!exists(path + Extension)) continue;

			if (exists(path + ".txt")) {
				std::ifstream file(path + ".txt");

				std::string str;

				std::getline(file, str);
				std::istringstream iss(str);
				int a, b;

				if (!(iss >> a >> b)) { break; }

				if (a == m & b == n) {
					std::getline(file, str);

					int pos = 0;
					std::string token;
					string delimiter = " ";
					vector<float> vect;

					while ((pos = str.find(delimiter)) != std::string::npos) {
						token = str.substr(0, pos);
						vect.push_back(stod(token));
						str.erase(0, pos + delimiter.length());
					}
					vecteursCarPrim.at(items.size() - 1).emplace_back(vect);
				}
				else
				{
					std::ofstream outfile(path + ".txt");
					int a = 0;
					int b = 0;
					outfile << m << " " << n << std::endl;

					Mat symbole;
					capture(symbole, path + Extension);

					cout << "calcul GFD du Symbole " + items.at(items.size() - 1) << endl;
					symbolTocomposantGfd(symbole, m, n);

					String str;
					vector<float>& ptr_vect = vectC;
					for (int i = 0; i < vectC.size(); i++)
					{
						str += to_string(ptr_vect.at(i)) + " ";
					}
					outfile << str;

					outfile.close();
					vecteursCarPrim.at(items.size() - 1).emplace_back(ptr_vect);
				}
			}
			else
			{
				std::ofstream outfile(path + ".txt");
				int a = 0;
				int b = 0;
				outfile << m << " " << n << std::endl;

				Mat symbole;
				capture(symbole, path + Extension);

				cout << "calcul GFD du Symbole " + items.at(items.size() - 1) << endl;
				symbolTocomposantGfd(symbole, m, n);

				String str;
				vector<float>& ptr_vect = vectC;
				for (int i = 0; i < vectC.size(); i++)
				{
					str += to_string(ptr_vect.at(i)) + " ";
				}
				outfile << str;

				outfile.close();
				vecteursCarPrim.at(items.size() - 1).emplace_back(ptr_vect);
			}
		}
	}
}

void symbolTocomposantGfd(Mat& mat, int m, int n) {
	Mat symbolecentred;

	filterNonInv(mat, mat);

	CC composant;
	composant.setId_label(-1);
	composant.setdX(mat.size().width);
	composant.setdY(mat.size().height);
	Point centroid((int)mat.size().width / 2, (int)mat.size().height / 2);
	composant.setCentroid(centroid);
	composant.setPtr_debut(Point(0, 0));
	composant.setMat(mat);

	GFD(composant, symbolecentred, m, n); //vectC
}