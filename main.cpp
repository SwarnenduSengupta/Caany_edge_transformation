#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
using namespace std;
using namespace cv;
Mat src, src_gray;
int thresh=50;
RNG rng(12345);
Mat canny_output;
vector<vector<Point>> contours;
vector<Vec4i> hierarchy;
vector<Vec4i> hierarchy_update;
#define length  1280
#define width 720
#define w 400
#define dist_thresh 50
int max_trackbar = 150;
const char* standard_name = "Standard Hough Lines Demo";
const char* probabilistic_name = "Probabilistic Hough Lines Demo";
int min_threshold = 1;
int s_trackbar = max_trackbar;
Mat edges;
Mat standard_hough;
cv::Point2f center(0,0);


bool compareContourAreas ( std::vector<cv::Point> contour1, std::vector<cv::Point> contour2 ) {
    double i = fabs( contourArea(cv::Mat(contour1)) );
    double j = fabs( contourArea(cv::Mat(contour2)) );
    return ( i < j );
}
void MyFilledCircle( Mat img, Point center )
{
    int thickness = -1;
    int lineType = 8;

    circle( img,
            center,
            w/32.0,
            Scalar( 0, 0, 255 ),
            thickness,
            lineType );
}
bool check_dist(Point p, vector<Point> &final_intersection)
{
    cout << final_intersection.size()<< endl;
    int flag=1;
    for( int j= 0; j<final_intersection.size();j++)
    {
        double res = norm(p-final_intersection.at(j));
        cout << "dist"<<res<< endl;
        if(res<dist_thresh)
        {
            flag++;
        }
    }
    if(flag==1)
        return true;
    else
        return false;
}

void Standard_Hough( int, void* )
{
    vector<Vec2f> s_lines;
    vector<vector<Point>> lines;
    vector<vector<float>> pt_loc;
    cvtColor( edges, standard_hough, COLOR_GRAY2BGR );

    /// 1. Use Standard Hough Transform
    HoughLines( edges, s_lines, 0.5, CV_PI/180, min_threshold + s_trackbar, 0, 0 );
    vector<Point> intersection;
    /// Show the result
    //cout << s_lines.size()<< endl;
    for( size_t i = 0; i < s_lines.size(); i++ )
    {
        int no_pt=0;
        float r1 = s_lines[i][0], t1 = s_lines[i][1];
        double c2 = cos(t1), c1 = sin(t1);
        for( size_t j = 0; j< s_lines.size(); j++ )
        {
            float r2 = s_lines[j][0], t2 = s_lines[j][1];
            double c4 = cos(t2), c3 = sin(t2);
            if(abs(r1-r2)>20 || abs(t1-t2)>0.1 )
            {
                int x = cvRound((r2*c1-r1*c3)/(c1*c4-c2*c3));
                int y = cvRound((r1-x*c2)/c1);
                if(x<length && x>0 && y< width && y>0)
                {
                    Point pt(x, y);
                    int dilation_size = 1;
                    Mat element = getStructuringElement(MORPH_RECT,
                                                        Size(2 * dilation_size + 1, 2 * dilation_size + 1),
                                                        Point(dilation_size, dilation_size));
                    /// Apply the dilation operation
                    dilate(standard_hough, standard_hough, element);
                    // applying dilation in an
                    int erosion_size = 1;
                    Mat element_e = getStructuringElement(MORPH_RECT,
                                                          Size(2 * erosion_size + 1, 2 * erosion_size + 1),
                                                          Point(erosion_size, erosion_size));
                    /// Apply the erosion operation
                    erode(standard_hough, standard_hough, element);
                    double x01 = r1 * c2, y01 = r1 * c1;
                    no_pt++;
                    vector<float> temp;
                    temp.push_back(r1);
                    temp.push_back(r2);
                    temp.push_back(x);
                    temp.push_back(y);
                    temp.push_back(t1);
                    temp.push_back(t2);
                    pt_loc.push_back(temp);
                    double alpha = 1000;
                    Point pt1(cvRound(x01 + alpha * (-c1)), cvRound(y01 + alpha * c2));
                    Point pt2(cvRound(x01 - alpha * (-c1)), cvRound(y01 - alpha * c2));
                    line(standard_hough, pt1, pt2, Scalar(255, 255, 255), 3, LINE_AA);
                    double x02 = r2 * c4, y02 = r2 * c3;
                    Point pt3(cvRound(x02 + alpha * (-c3)), cvRound(y02 + alpha * c4));
                    Point pt4(cvRound(x02 - alpha * (-c3)), cvRound(y02 - alpha * c4));
                    line(standard_hough, pt3, pt4, Scalar(255, 255, 255), 3, LINE_AA);
                    intersection.push_back(pt);
                }
            }
        }
    }
    vector<Point> corners;
    if(intersection.size()>0)
    {
        cout << intersection.size() << endl;
        for(int i=0;i<intersection.size();i++)
        {
            Point p = intersection.at(i);
            if(check_dist(p,corners) )//|| i==(intersection.size()-1))
            {
                cout << "Final before" << corners.size() << endl;
                corners.push_back(p);
                cout << "Final" <<corners.size() << endl;
            }

        }
    }
    for(int i=0;i < corners.size();i++)
        MyFilledCircle(standard_hough,corners.at(i));
    imshow( standard_name, standard_hough );
}

int main(int argc,char** argv) {
    String imagename("/home/swarnendu/Downloads/pics/5.jpeg");
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
    imshow("Source Original",src);

    // test
    cvtColor(src,src,COLOR_BGR2YCrCb);
    vector<Mat> channels;
    split(src,channels);
    //Ptr<CLAHE> ptr = createCLAHE(2);
    //ptr->apply(channels[0],channels[0]);
    equalizeHist(channels[0],channels[0]);
    merge(channels,src);
    cvtColor(src,src, COLOR_YCrCb2BGR);
    imshow("Source2",src);

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

    imshow(source_window,src_gray);
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
    char thresh_label[50];
    sprintf( thresh_label, "Thres: %d + input", min_threshold );

    namedWindow( standard_name, WINDOW_AUTOSIZE );
    cout <<"start "<< endl;
    createTrackbar( "Threshold", standard_name, &s_trackbar, max_trackbar, Standard_Hough);// Draw the lines
    Standard_Hough(0, 0);
    //int start_threshold = 500;

    waitKey();
    return (0);
}
