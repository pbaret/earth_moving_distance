#include <iostream>
#include <opencv2/opencv.hpp>

// Association of a color in CIE-Lab space and a cluster label
struct colorLAB
{
    int k; // cluster
    cv::Vec3b color; // Lab value

    colorLAB()
    {
        this->k = -1;
        this->color = cv::Vec3b(0,0,0);
    };

    colorLAB(int cluster, cv::Vec3b col )
    {
        this->k = cluster;
        this->color = col;
    };
};

// Euclidian distance between 2 colors in CIE-Lab space
float dist (cv::Vec3b a, cv::Vec3b b)
{
    return std::sqrt( (a[0]-b[0])*(a[0]-b[0]) + (a[1]-b[1])*(a[1]-b[1]) + (a[2]-b[2])*(a[2]-b[2]) );
}


int main(int argc, char *argv[])
{
    // 1. Load the image
    cv::Mat img = cv::imread(argv[1], 1);
    cv::Mat blurred;

    // 2. Blur the image to make color more homogeneous
    cv::GaussianBlur(img,blurred,cv::Size(),5,5);

    // 3. Convert the image from BGR to CIE-Lab space
    cv::Mat lab_img;
    cv::cvtColor(blurred, lab_img, cv::COLOR_BGR2Lab);
    cv::Mat channels[3];
    cv::split(lab_img, channels);

    // 4. Init the points for K-Mean
    int nb_pixel = img.rows * img.cols;
    colorLAB pixels[nb_pixel];

    for (int l = 0; l < img.rows; l++)
    {
        for (int c = 0; c < img.cols; c++)
        {
            int idx = l * img.cols + c;
            unsigned char L = channels[0].at<unsigned char>(l, c);
            unsigned char a = channels[1].at<unsigned char>(l, c);
            unsigned char b = channels[2].at<unsigned char>(l, c);
            pixels[idx] = colorLAB(-1,cv::Vec3b(L,a,b));
       }
    }

    // 5. Init the clusters for K-Mean
    int nb_cluster = 8;
    cv::Vec3b centroids[nb_cluster];    // Centroids of the clusters
    int tmp[nb_cluster][3];             // Tmp array to compute centroids when updated
    int count[nb_cluster];              // Nb of points in each cluster (used to compute centroids)

    // initial values of the centroids
    centroids[0] = cv::Vec3b(64,64,64);
    centroids[1] = cv::Vec3b(64,64,191);
    centroids[2] = cv::Vec3b(64,191,191);
    centroids[3] = cv::Vec3b(64,191,64);
    centroids[4] = cv::Vec3b(191,64,64);
    centroids[5] = cv::Vec3b(191,64,191);
    centroids[6] = cv::Vec3b(191,191,191);
    centroids[7] = cv::Vec3b(191,191,64);

    // Try random initialization
//    centroids[0] = cv::Vec3b(std::rand()%255, std::rand()%255, std::rand()%255);
//    centroids[1] = cv::Vec3b(std::rand()%255, std::rand()%255, std::rand()%255);
//    centroids[2] = cv::Vec3b(std::rand()%255, std::rand()%255, std::rand()%255);
//    centroids[3] = cv::Vec3b(std::rand()%255, std::rand()%255, std::rand()%255);
//    centroids[4] = cv::Vec3b(std::rand()%255, std::rand()%255, std::rand()%255);
//    centroids[5] = cv::Vec3b(std::rand()%255, std::rand()%255, std::rand()%255);
//    centroids[6] = cv::Vec3b(std::rand()%255, std::rand()%255, std::rand()%255);
//    centroids[7] = cv::Vec3b(std::rand()%255, std::rand()%255, std::rand()%255);


    // 6. K-Mean to sum up colors in the image
    bool stop = false;
    int max_iter = 50;
    int iter = 0;

    while (!stop)
    {
        // reset tmp values
        bool change = false;
        for (int i = 0; i < nb_cluster; i++)
        {
            tmp[i][0] = 0;
            tmp[i][1] = 0;
            tmp[i][2] = 0;
            count[i] = 0;
        }

        // For each point, search for the closest cluster
        for (int i = 0; i < nb_pixel; i++)
        {
            // Search for closest cluster
            float min_dist = dist(pixels[i].color, centroids[0]);
            int cluster = 0;
            for (int k = 1; k < nb_cluster; k++)
            {
                float d = dist(pixels[i].color, centroids[k]);
                if (d < min_dist)
                {
                    min_dist = d;
                    cluster = k;
                }
            }

            change = change || (cluster == pixels[i].k);    // Check if the closest cluster is the same as before
            pixels[i].k = cluster;                          // Move the pixel in the closest cluster
            tmp[cluster][0] += static_cast<int>(pixels[i].color[0]);
            tmp[cluster][1] += static_cast<int>(pixels[i].color[1]);
            tmp[cluster][2] += static_cast<int>(pixels[i].color[2]);
            count[cluster]++;
        }

        stop = (!change);

        // Update the clusters (and check that no cluster is empty)
        for (int i = 0; i < nb_cluster; i++)
        {
            if (count[i] == 0) // if the current cluster is now empty, we randomly move the centroid
            {
                centroids[i][0] = 128;
                centroids[i][1] = std::rand() % 255;
                centroids[i][2] = std::rand() % 255;
                stop = false;
            }
            else // otherwise we update the centroid
            {
                centroids[i][0] = static_cast<unsigned char>(tmp[i][0] * 1.0 / count[i]);
                centroids[i][1] = static_cast<unsigned char>(tmp[i][1] * 1.0 / count[i]);
                centroids[i][2] = static_cast<unsigned char>(tmp[i][2] * 1.0 / count[i]);
            }
        }

        iter++;
        stop = stop || (iter > max_iter);
    }

    // 7. Build for signature image
    cv::Mat test(80,800,CV_8UC3);
    for (int l = 0; l < test.rows; l++)
    {
        for (int c = 0; c < test.cols; c++)
        {
            test.at<cv::Vec3b>(l,c) = centroids[c/100];
        }
    }

    // Ratio of pixels in each cluster
    for (int i = 0; i < nb_cluster; i++)
    {
        std::cout << count[i] * 100.0 / nb_pixel << "%    " ;
    }
    std::cout << std::endl;


    cv::cvtColor(test,test,cv::COLOR_Lab2BGR);

    cv::imshow("img",img);
    cv::imshow("test",test);
    cv::waitKey(0);

    
    return 0;
}
