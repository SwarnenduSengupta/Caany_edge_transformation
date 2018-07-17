#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
using namespace std;
using namespace cv;
Mat src, src_gray;
int thresh=100;
RNG rng(12345);
Mat canny_output;
vector<vector<Point>> contours;
vector<Vec4i> hierarchy;
vector<Vec4i> hierarchy_update;
int max_trackbar = 500;
const char* standard_name = "Standard Hough Lines Demo";
const char* probabilistic_name = "Probabilistic Hough Lines Demo";
int min_threshold = 50;
int s_trackbar = max_trackbar;
Mat edges;
Mat standard_hough;


bool compareContourAreas ( std::vector<cv::Point> contour1, std::vector<cv::Point> contour2 ) {
    double i = fabs( contourArea(cv::Mat(contour1)) );
    double j = fabs( contourArea(cv::Mat(contour2)) );
    return ( i < j );
}
cv::Point2f center(0,0);

cv::Point2f computeIntersect(cv::Vec4i a,
                             cv::Vec4i b)
{
    int x1 = a[0], y1 = a[1], x2 = a[2], y2 = a[3], x3 = b[0], y3 = b[1], x4 = b[2], y4 = b[3];
    float denom;

    if (float d = ((float)(x1 - x2) * (y3 - y4)) - ((y1 - y2) * (x3 - x4)))
    {
        cv::Point2f pt;
        pt.x = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / d;
        pt.y = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / d;
        return pt;
    }
    else
        return cv::Point2f(-1, -1);
}

void sortCorners(std::vector<cv::Point2f>& corners,
                 cv::Point2f center)
{
    std::vector<cv::Point2f> top, bot;

    for (int i = 0; i < corners.size(); i++)
    {
        if (corners[i].y < center.y)
            top.push_back(corners[i]);
        else
            bot.push_back(corners[i]);
    }
    corners.clear();

    if (top.size() == 2 && bot.size() == 2){
        cv::Point2f tl = top[0].x > top[1].x ? top[1] : top[0];
        cv::Point2f tr = top[0].x > top[1].x ? top[0] : top[1];
        cv::Point2f bl = bot[0].x > bot[1].x ? bot[1] : bot[0];
        cv::Point2f br = bot[0].x > bot[1].x ? bot[0] : bot[1];


        corners.push_back(tl);
        corners.push_back(tr);
        corners.push_back(br);
        corners.push_back(bl);
    }
}
void Standard_Hough( int, void* )
{
    vector<Vec2f> s_lines;
    cvtColor( edges, standard_hough, COLOR_GRAY2BGR );

    /// 1. Use Standard Hough Transform
    HoughLines( edges, s_lines, 1, CV_PI/180, min_threshold + s_trackbar, 0, 0 );

    /// Show the result
    for( size_t i = 0; i < s_lines.size(); i++ )
    {
        float r = s_lines[i][0], t = s_lines[i][1];
        double cos_t = cos(t), sin_t = sin(t);
        double x0 = r*cos_t, y0 = r*sin_t;
        double alpha = 1000;

        Point pt1( cvRound(x0 + alpha*(-sin_t)), cvRound(y0 + alpha*cos_t) );
        Point pt2( cvRound(x0 - alpha*(-sin_t)), cvRound(y0 - alpha*cos_t) );
        line( standard_hough, pt1, pt2, Scalar(255,255,255), 3, LINE_AA);
    }
    int dilation_size = 1 ;
    Mat element = getStructuringElement( MORPH_RECT,
                                         Size( 2*dilation_size + 1, 2*dilation_size+1 ),
                                         Point( dilation_size, dilation_size ) );
    /// Apply the dilation operation
    dilate(standard_hough,standard_hough,element);
    // applying dilation in an
    int erosion_size = 1 ;
    Mat element_e = getStructuringElement( MORPH_RECT,
                                           Size( 2*erosion_size+ 1, 2*erosion_size+1 ),
                                           Point( erosion_size,erosion_size ) );
    /// Apply the erosion operation
    erode(standard_hough,standard_hough,element);
    /// Applying erosion and dilation for getting continuous lines

    imshow( standard_name, standard_hough );
}

int main(int argc,char** argv) {
    String imagename("/home/swarnendu/Downloads/pics/1.jpeg");
    // reading the image path
    if(argc>1)
    {
        imagename=argv[1];
    }
    src=imread(imagename,IMREAD_COLOR);
    // reading the image
    if(src.empty())
    {
        cerr << "No image supplied ..." << endl;
        return -1;
    }
    //Ptr<CLAHE> hist = createCLAHE(400);
    //hist->apply(src,src);
    cvtColor(src,src_gray,COLOR_BGR2GRAY);
    // converting the colored image to grayscale
    blur(src_gray,src_gray,Size(3,3));
    //blurring the image which makes it better for detection of edges
    const char* source_window="Source";
    // Naming the window for printing the original image
    namedWindow(source_window,WINDOW_AUTOSIZE);
    // opening the window for display
    imshow(source_window,src);
    // displaying the image

    Canny(src_gray,canny_output,thresh,thresh*2,3);
    // Canny edge detection with threshold at 100 aand 200
    // Kernel size is 3 X 3
    namedWindow("Canny");
    int dilation_size = 1 ;
    Mat element = getStructuringElement( MORPH_RECT,
                                         Size( 2*dilation_size + 1, 2*dilation_size+1 ),
                                         Point( dilation_size, dilation_size ) );
    /// Apply the dilation operation
    dilate(canny_output,canny_output,element);
    // applying dilation in an
    int erosion_size = 1 ;
    Mat element_e = getStructuringElement( MORPH_RECT,
                                           Size( 2*erosion_size+ 1, 2*erosion_size+1 ),
                                           Point( erosion_size,erosion_size ) );
    /// Apply the erosion operation
    erode(canny_output,canny_output,element);
    /// Applying erosion and dilation for getting continuous lines

    imshow("Canny",canny_output);
    // Showing output of canny after erosion and dilation
    findContours(canny_output,contours,hierarchy,RETR_EXTERNAL,CHAIN_APPROX_SIMPLE,Point(0,0));
    // here the contour are being found using a tree hierarchy
    Mat drawing=Mat::zeros(canny_output.size(),CV_8UC3);
    std::sort(contours.begin(), contours.end(), compareContourAreas);
    // sorting the contours according to their area
    int print_till = contours.size()-2;
    for(size_t i=contours.size()-1;i>print_till;i--)
    {
        if(hierarchy.at(i).val[3]==-1) {
            Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
            drawContours(drawing, contours, (int) i, color, 1, 8, hierarchy, 0, Point());
            cout << contourArea(contours.at(i))<<endl;
            // drawing the contour and printing the area

            //for(auto vec:hierarchy_update)
            //  cout << vec << endl;
        }
    }
    namedWindow("Contours",WINDOW_AUTOSIZE);
    imshow("Contours",drawing);
    cvtColor(drawing,drawing,COLOR_RGB2GRAY);
    imshow("SECOND CANNY",drawing);
    Canny(drawing,edges,thresh,2*thresh);
    cout << "going in" << endl;
    //cvtColor(canny_output,canny_output,COLOR_RGB2GRAY);
    //cvtColor(drawing,drawing,COLOR_GRAY2BGR);
    // Standard Hough Line Transform
    /// Create Trackbars for Thresholds
//    char thresh_label[50];
//    sprintf( thresh_label, "Thres: %d + input", min_threshold );

    namedWindow( standard_name, WINDOW_AUTOSIZE );
    cout <<"start "<< endl;
    createTrackbar( "Threshold", standard_name, &s_trackbar, max_trackbar, Standard_Hough);// Draw the lines
    Standard_Hough(0, 0);

    waitKey(0);
    return (0);
}
