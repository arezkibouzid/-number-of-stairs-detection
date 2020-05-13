#include <opencv/cv.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
using namespace cv;
using namespace std;

int main(int argc, char const *argv[]) {
  vector<Mat> dst;
  vector<Mat> images;
  for (int a = 0; a < 6; a++) {

    String ImgBDD = format("/home/lbenboudiaf/Bureau/ProjetImagerie/bdd/esc%d.jpg", a);
    Mat src = imread(ImgBDD,IMREAD_GRAYSCALE);
    if (src.empty()){
      printf(" Error loading image %d\n", a);
      continue;
    }
    images.push_back(src);
  }

  for (int i = 0; i < images.size(); i++) {
    Canny(images[i], dst[i], 50, 200, 3);
    printf("Hello\n");
    imshow("Images", images[i]);
    waitKey();
  }
  return 0;
}
