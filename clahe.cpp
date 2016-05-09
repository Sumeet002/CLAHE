/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2013, NVIDIA Corporation, all rights reserved.
// Copyright (C) 2014, Itseez Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the copyright holders or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#include "opencv2/highgui/highgui.hpp"
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <sys/time.h>

using namespace cv;
using namespace std;







int main(int argc, char* argv[])
{

	//Src matrix for loading image
        static Mat src;
	
	src =imread("sample2.jpg",CV_LOAD_IMAGE_GRAYSCALE);
	int scale = 2 ;
	Size size(src.cols/scale,src.rows/scale);
	int imgSize= (src.cols/scale)*(src.rows/scale);
        resize(src,src,size);
	imshow("Gray Image" , src);
	waitKey(30);

        //clipLimit
        double clipLimit = 4.0;

        //Tile size
        Size tileGridSize(8,8);
        int tilesX = tileGridSize.width;
        int tilesY = tileGridSize.height;
        Size tileSize;
       
        Mat src_Ext,lut,dst;
        int histSize = 256;
        cv::InputArray srcForLut;

	//Pass image directly if divisible by tileSize
        if(src.size().width % tilesX == 0 && src.size().height % tilesY == 0){
        	srcForLut = src ;
        }
	//else create a padding on all sides of image to make it divisible by tileSize
        else{
		cv::copyMakeBorder(src,src_Ext,0,tilesY - (src.size().height % tilesY),0,tilesX - (src.size().width % tilesX),cv::BORDER_REFLECT_101);
                tileSize = Size(src_Ext.size().width/tilesX,src_Ext.size().height/tilesY);
                srcForLut = src_Ext; 
        }

        const int tileSizeTotal = tileSize.area();
        const float lutScale = static_cast<float>(histSize - 1) / tileSizeTotal;
        
       
        if(clipLimit > 0.0){
        	clipLimit = static_cast<int>(clipLimit * tileSizeTotal / histSize);
		clipLimit = std::max(clipLimit, 1);
        }
   
        
        dst.create(src.size(),src.type()); //Create dst image with same type and size as src image	
        lut.create(tilesX*tilesY,histSize,src.type());  //Create the look-up-table matrix

        calcLutBody(srcForLut, lut, tileSize, tilesX, clipLimit, lutScale); 
        Interpolate(src, dst, lut, tileSize, tilesX, tilesY);


	while(true){
		imshow("CLAHE Image",dst);
		waitKey(30);
	}

	return 0;

}
