#include <iostream>
#include <cstdio>
#include <fstream>
#include <numeric>
#include <algorithm>
#include <functional>
#include <vector>
#include <limits>
#include <stdexcept>
#include <dlr.h>
#include <libgen.h>
#include <utility>
#include "dmlc/logging.h"

#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/videoio.hpp"
#include <memory.h>

using namespace cv;
#define RES_X 224
#define RES_Y 224
#define NUM_CHANNELS 3

Mat image;

// Set to video port, if camera input needed
int live_input = -1;

char imagenet_win[160];
bool is_big_endian();
bool RunInference(DLRModelHandle model, 
                  const char *data_path,
                  const std::string& input_name,
                  const int batch_size,
                  const int frame_cnt);

bool SetupInput(VideoCapture& cap, const char *video_clip)
{
   if(live_input >= 0)
   {
      cap.open(live_input);

      const double fps = cap.get(CAP_PROP_FPS);
      const int width  = cap.get(CAP_PROP_FRAME_WIDTH);
      const int height = cap.get(CAP_PROP_FRAME_HEIGHT);
      std::cout << "Capture camera with " << fps << " fps, " << width << "x"
                << height << " px" << std::endl;
   } else {
     std::cout << "Video input clip: " << video_clip << std::endl;
     cap.open(std::string(video_clip));
      const double fps = cap.get(CAP_PROP_FPS);
      const int width  = cap.get(CAP_PROP_FRAME_WIDTH);
      const int height = cap.get(CAP_PROP_FRAME_HEIGHT);
      std::cout << "Clip with " << fps << " fps, " << width << "x"
                << height << " px" << std::endl;
   }

   if (!cap.isOpened()) {
      std::cout << "Video input not opened!" << std::endl;
      return false;
   }

   return true;
}

void ImagePreprocessing(Mat &image, float *output_data) {
  Mat spl[NUM_CHANNELS];
  split(image,spl);

  for(int c = 0; c < NUM_CHANNELS; c++)
  {
     const unsigned char* data = spl[c].ptr();
     for(int y = 0; y < RES_Y; y++)
     {
       for(int x = 0; x < RES_X; x++)
       {
         int32_t in =  data[y*RES_X + x];
         in -= 128;
         if(in > 127)  in  = 127;
         if(in < -128) in = -128;
         output_data[NUM_CHANNELS * (y*RES_X + x) + NUM_CHANNELS-1 - c] = (float)in / 128.0;
       }
     }
  }
}

void imagenetCallBackFunc(int event, int x, int y, int flags, void* userdata) {
    if  ( event == EVENT_RBUTTONDOWN )
    {
        std::cout << "Right button of the mouse is clicked - position (" << x << ", " << y << ")" << " ... prepare to exit!" << std::endl;
        exit(0);
    }
}

void SetupLiveDisplay(){
    sprintf(imagenet_win, "Neo-AI-DLR inference");
    namedWindow(imagenet_win, WINDOW_AUTOSIZE | CV_GUI_NORMAL);
    //set the callback function for any mouse event
    setMouseCallback(imagenet_win, imagenetCallBackFunc, NULL);
}

void DisplayFrames(Mat &show_image, std::string &output_labels) {
    // overlay the display window, if ball seen during last two times
    cv::putText(show_image, output_labels.c_str(),
                cv::Point(32, 32), // Coordinates
                cv::FONT_HERSHEY_COMPLEX_SMALL, // Font
                1.5, // Scale. 2.0 = 2x bigger
                cv::Scalar(0,0,0), // Color
                1, // Thickness
                8); // Line type
    cv::imshow(imagenet_win, show_image); 
}

void CollectFrames(VideoCapture &cap, const int numFrames, float *output_data, Mat &in_image)
{
  int frame_cnt = 0;
  while(frame_cnt < numFrames)
  {
    if (cap.grab())
    {
        if (cap.retrieve(in_image))
        {
            if(live_input >= 0)
            { //Crop central square portion
              int loc_xmin = (in_image.size().width - in_image.size().height) / 2; //Central position
              int loc_ymin = 0;
              int loc_w = in_image.size().height;
              int loc_h = in_image.size().height;

              cv::resize(in_image(Rect(loc_xmin, loc_ymin, loc_w, loc_h)), image, Size(RES_X, RES_Y));
            } else {
              cv::resize(in_image, image, Size(RES_X,RES_Y));
            }
            ImagePreprocessing(image, &output_data[NUM_CHANNELS * frame_cnt * RES_X * RES_Y]);
            frame_cnt ++;
        }
    } else {
      if(live_input == -1) {
        //Rewind!
        cap.set(CAP_PROP_POS_FRAMES, 0);
      }
    }
  }
}

bool is_big_endian() {
  int32_t n = 1;
  // big endian if true
  return (*(char *)&n == 0);
}

/*! \brief A generic inference function using C-API.
 */
bool RunInference(DLRModelHandle model, const char* data_path, const std::string& input_name, const int batch_size, const char *videoClip, 
                  int frame_cnt, std::vector<std::string> &labels)
{
  Mat single_image_from_batch;
  int num_outputs;
  GetDLRNumOutputs(&model, &num_outputs);
  std::vector<int64_t> output_sizes;
  float *input_data = (float *)malloc(batch_size * RES_X * RES_Y * NUM_CHANNELS * sizeof(float));

  // Setup Input
  VideoCapture cap;
  if (! SetupInput(cap, videoClip))  return false;

  for (int i = 0; i < num_outputs; i++) {
    int64_t cur_size = 0;
    int cur_dim = 0;
    GetDLROutputSizeDim(&model, i, &cur_size, &cur_dim);
    output_sizes.push_back(cur_size);
  }

  std::vector<std::vector<float>> outputs;
  for (auto i : output_sizes) {
    outputs.push_back(std::vector<float>(i, 0));
  }

  int frame_batch_cnt = frame_cnt / batch_size; // How many batches to process?
  int frame_index = 0; 
  while(frame_batch_cnt >= 0) {
    int64_t image_shape[4] = { batch_size, RES_Y, RES_X, NUM_CHANNELS };
    int argmax = -1;
    std::string last_label = "None";

    CollectFrames(cap, batch_size, input_data, single_image_from_batch);

    // Place to put OpenCV camera capture - for video clip, just take 'batch_size' images, pre-process and store into input.data
    // In case of camera capture, ping-pong of batch size buffers should happen here - use semaphore to wait for batch size frames
    // to be ready ... to proceed with the processing.
    if (SetDLRInput(&model, input_name.c_str(), image_shape, input_data, 4) != 0) {
      throw std::runtime_error("Could not set input '" + input_name + "'");
    }
    if (RunDLRModel(&model) != 0) {
      LOG(INFO) << DLRGetLastError() << std::endl;  
      throw std::runtime_error("Could not run");
    }
    for (int i = 0; i < num_outputs; i++){
      if (GetDLROutput(&model, i, outputs[i].data()) != 0) {
        throw std::runtime_error("Could not get output" + std::to_string(i));
      }
    }
    const int single_inference_size =  outputs[0].size() / batch_size;
    for(int j = 0; j < batch_size; j++)
    {
      argmax = -1;
      float max_pred = 0.0f;
      for (int i = 0; i < single_inference_size; i++) {
        if (outputs[0][j * single_inference_size + i] > max_pred) {
          max_pred = outputs[0][j * single_inference_size + i];
          argmax = i;
        }
      }
      std::cout << "[" << (frame_index + j) << "] Max probability at " << argmax << " with probability " << max_pred;
      last_label = "None";
      if(argmax < labels.size()) {
        std::cout << " label:" << labels[argmax];
        std::cout << std::endl;
        last_label = labels[argmax];
      }
    }
    // Display every BATCH_SIZEth frame
    DisplayFrames(single_image_from_batch, last_label);

    frame_batch_cnt --;
    frame_index += batch_size;
  }
  free(input_data);
  return true;
}
//-----------------------------------------------------------------------------------------------------------------------------------------
/*
* It will iterate through all the lines in file and
* put them in given vector
*/
bool getFileContent(std::string fileName, std::vector<std::string> & vecOfStrs)
{
  // Open the File
  std::ifstream in(fileName.c_str());
  // Check if object is valid
  if(!in)
  {
    std::cerr << "Cannot open the File : "<<fileName<<std::endl;
    return false;
  }
  std::string str;
  // Read the next line from File untill it reaches the end.
  while (std::getline(in, str))
  {
    // Line contains string of length > 0 then save it in vector
    if(str.size() > 0)
    vecOfStrs.push_back(str);
  }
  //Close The File
  in.close();
  return true;
}

//------------------------------------------------------------------------------------------------------------------------------------------
int main(int argc, char** argv) {
int batch_size = 1;
bool clip_f = false;
const char *videoClip = "/usr/share/ti/tidl/examples/classification/clips/test10.mp4";
std::vector<std::string> vecOfLabels;
bool labels_ok = false;
int frameCount = 1200;

  if (is_big_endian()) {
    std::cerr << "Big endian not supported" << std::endl;
    return 1;
  }
  int device_type = 1;
  std::string input_name = "data";
  if (argc < 3) {
    std::cerr << "Usage: " << argv[0] << " <model dir> <ndarray file> [device] [input name]" << std::endl;
    return 1;
  }
  if (argc >= 4) {
    std::string argv3(argv[3]);
    if (argv3 == "cpu"){
      device_type = 1;
    } else if (argv3 == "gpu") {
      device_type = 2;
    } else if (argv3 == "opencl") {
      device_type = 4;
    } else {
      std::cerr << "Unsupported device type!" << std::endl;
      return 1; 
    }
  }
  if (argc >= 5) {
    input_name = argv[4];
  }
  if (argc >= 6) {
    batch_size = atoi(argv[5]);
  }
  if (argc >= 7) {
    clip_f = true;
  }
  if (argc >= 8) {
    frameCount = atoi(argv[7]);
  }
  if (argc >= 9) {
    // Get the contents of file in a vector
    labels_ok = getFileContent(argv[8], vecOfLabels);
  }

  std::cout << "Loading model... " << std::endl;
  DLRModelHandle model;

  if (CreateDLRModel(&model, argv[1], device_type, 0) != 0) {
    LOG(INFO) << DLRGetLastError() << std::endl;
    throw std::runtime_error("Could not load DLR Model");
  }
  
  std::cout << "Running inference... " << std::endl;  

  if(RunInference(model, argv[2], input_name, batch_size, clip_f ? argv[6] : videoClip, frameCount, vecOfLabels))
  {
    std::cout << "TEST PASSED" << std::endl;
  } else {
    std::cout << "TEST FAILED" << std::endl;
  }
  return 0;
}
