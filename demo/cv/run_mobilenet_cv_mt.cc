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
#include <thread>
#include <chrono>
#include <mutex>
#include "dmlc/logging.h"

#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/videoio.hpp"
#include <memory.h>
#include <unistd.h>
#include <getopt.h>

using namespace cv;
#define RES_X 224
#define RES_Y 224
#define BATCH_SIZE 4
VideoCapture cap;
DLRModelHandle model;
char dlr_model[320];
std::string input_name("input"); //Mobilenet is default
std::vector<std::string> vecOfLabels;
// Set to video port, if camera input needed
int live_input = -1;
int do_inference = 1;
char imagenet_win[160];
bool is_big_endian();
bool set_verbose = false;
int  max_frameCount = 1200;
//===============================================================================
std::chrono::time_point<std::chrono::system_clock> start_ts;
const int MAX_GRAB_SIZE = 12;	// Frame buffer around motion ...
const int MAX_PROC_SIZE = 24;	// Frame buffer around motion ...
Mat frameStack[MAX_GRAB_SIZE];
Mat processedStack[MAX_PROC_SIZE];
volatile int frameStack_rd = 0;
volatile int frameStack_wr = 0;
volatile int processedStack_rd = 0;
volatile int processedStack_wr = 0;
std::mutex m_grab;
std::mutex m_proc;
std::mutex m_ts;
int stopSig = 0;				// Global stop signal...

//===============================================================================
void show_ts(const char *checkpoint)
{
  if(set_verbose)
  {
    std::chrono::time_point<std::chrono::system_clock> now_ts = std::chrono::system_clock::now();
    m_ts.lock();
    std::cout << checkpoint << ":" <<  std::chrono::duration_cast<std::chrono::milliseconds>(now_ts - start_ts).count() << "ms\n";
    m_ts.unlock();
  }
}
//===============================================================================
bool SetupInput(const char *video_clip)
{
   if(live_input >= 0)
   {
      cap.open(live_input);
      cap.set(CV_CAP_PROP_FRAME_WIDTH, 640);
      cap.set(CV_CAP_PROP_FRAME_HEIGHT, 480);
      cap.set(CV_CAP_PROP_FPS, 30);
      std::cout << "Setting up camera HxWxFPS\n";
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
/*! \brief Image preprocessing for TF model
 */
void ImagePreprocessing(Mat &image, float *output_data) {
Mat spl[3];
  split(image,spl);

  for(int c = 0; c < 3; c++)
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
         output_data[3 * (y*RES_X + x) + 3 - c] = (float)in / 128.0;
       }
     }
  }
}

/*! \brief Do inference on acquired images
 */
void ProcessFrames(void) {
Mat frame[BATCH_SIZE], bigFrame[BATCH_SIZE];
std::vector<int64_t> output_sizes;
int frame_cnt = 0;
char tmp_string[160];
int num_outputs;
const int batch_size = BATCH_SIZE;
float *input_data = (float *)malloc(batch_size * RES_X * RES_Y * 3 * sizeof(float));
int64_t image_shape[4] = { batch_size, RES_Y, RES_X, 3 };
double fp_ms_avg = 0.0; //Initial inference time

    GetDLRNumOutputs(&model, &num_outputs);

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
    //First inference is dummy, initialization call!!
    std::cout << "DUMMY inference call (initialization) started...\n";
    memset(input_data, 0, batch_size * RES_X * RES_Y * 3 * sizeof(float));
    if (SetDLRInput(&model, input_name.c_str(), image_shape, input_data, 4) != 0) {
        throw std::runtime_error("Could not set input '" + input_name + "'");
    }
    if (RunDLRModel(&model) != 0) {
        LOG(INFO) << DLRGetLastError() << std::endl;  
        throw std::runtime_error("Could not run");
    }
    for (int i = 0; i < num_outputs; i++) {
      if (GetDLROutput(&model, i, outputs[i].data()) != 0) {
          throw std::runtime_error("Could not get output" + std::to_string(i));
      }
    }
    std::cout << "...DUMMY inference call ended\n";
    // Endless processing loop...
    while(!::stopSig) {
      int argmax = -1;
      float max_pred = 0.0;
      std::string last_label = "None";
		if((frameStack_wr - frameStack_rd) >= BATCH_SIZE) 
                {   // If the original video stack is not empty...
                    show_ts("\tPROC_ENTER");
                    m_grab.lock();
                    for(int bcnt = 0; bcnt < BATCH_SIZE; bcnt ++)
                    {
                        Mat in_image = frameStack[frameStack_rd % MAX_GRAB_SIZE];
                        in_image.copyTo(bigFrame[bcnt]);
                        //Crop central square portion
                        int loc_xmin = (in_image.size().width - in_image.size().height) / 2; //Central position
                        int loc_ymin = 0;
                        int loc_w = in_image.size().height;
                        int loc_h = in_image.size().height;
                        cv::resize(in_image(Rect(loc_xmin, loc_ymin, loc_w, loc_h)), frame[bcnt], Size(RES_X, RES_Y));
                        frameStack_rd ++;
                    }
                    m_grab.unlock();
                    auto inference_begin_ts = std::chrono::high_resolution_clock::now();
		    //----------------------------------OpenCV image manipulations---------------------------------------
                    for(int bcnt = 0; bcnt < BATCH_SIZE; bcnt ++)
                    {
                       ImagePreprocessing(frame[bcnt], &input_data[3 * bcnt * RES_X * RES_Y]);
                    }
                    if(do_inference)
                    {
                      //----------------------------------------------------------------------------
                      // Single batch, runs BATCH_SIZE inferences
                      //----------------------------------------------------------------------------
                      if (SetDLRInput(&model, input_name.c_str(), image_shape, input_data, 4) != 0) {
                        throw std::runtime_error("Could not set input '" + input_name + "'");
                      }
                      if (RunDLRModel(&model) != 0) {
                        LOG(INFO) << DLRGetLastError() << std::endl;  
                        throw std::runtime_error("Could not run");
                      }
                      for (int i = 0; i < num_outputs; i++) {
                        if (GetDLROutput(&model, i, outputs[i].data()) != 0) {
                          throw std::runtime_error("Could not get output" + std::to_string(i));
                        }
                      }
                    }
                    auto inference_end_ts = std::chrono::high_resolution_clock::now();
                    std::chrono::duration<double, std::milli> fp_ms = inference_end_ts - inference_begin_ts;
                    fp_ms_avg = 0.9 * fp_ms_avg + 0.1 * ((double)fp_ms.count() / (double)BATCH_SIZE);

                    const int single_inference_size =  outputs[0].size() / BATCH_SIZE;
                    for(int bcnt = 0; bcnt < BATCH_SIZE; bcnt++)
                    {
                      if(do_inference)
                      {
                        argmax = -1;
                        max_pred = 0.0f;
                        for (int i = 0; i < single_inference_size; i++) {
                          if (outputs[0][bcnt * single_inference_size + i] > max_pred) {
                            max_pred = outputs[0][bcnt * single_inference_size + i];
                            argmax = i;
                          }
                        }
                        std::cout << "[" << (frame_cnt + bcnt) << "] Max probability at " << argmax << " with probability " << max_pred;
                        last_label = "None";
                        if(argmax < vecOfLabels.size()) {
                          std::cout << " label:" << vecOfLabels[argmax];
                          std::cout << std::endl;
                          last_label = vecOfLabels[argmax];
                        }
                      } else {
                        last_label = "No inference";
                        max_pred   = 0.0;
                      }
                      sprintf(tmp_string, "%04dfrm %5.2lfms (%s)", frame_cnt + bcnt, fp_ms_avg, last_label.c_str());
                      putText(bigFrame[bcnt], tmp_string,
                                Point(3, 20),
                                FONT_HERSHEY_COMPLEX_SMALL,
                                0.75,
                                Scalar(0,0,0), 1, 8);
                    }
		    //---------------------------------------------------------------------------------------------------
                    for(int bcnt = 0; bcnt < BATCH_SIZE; bcnt ++)
                    {
		        //  If a new processed frame is available and the stack is not yet full..:
		        if((processedStack_wr - processedStack_rd) < MAX_PROC_SIZE) {
		          // Put the new processed frame at the front location of the stack...:
                          m_proc.lock();
		          bigFrame[bcnt].copyTo(processedStack[processedStack_wr % MAX_PROC_SIZE]);
                          processedStack_wr ++;
                          m_proc.unlock();
                          show_ts("\tPROC_INSERT");
		        } else {
                          show_ts("\tPROC_SKIP");
                          std::this_thread::sleep_for(std::chrono::milliseconds(20));
                        }
                    }
                    frame_cnt += BATCH_SIZE;
		}
		
	}
	std::cout << "processFrame: esc key is pressed by user" << std::endl;
	return;
}
/*! \brief Thread to capture images from camera, or do video clip decoding
 */
void CollectFrames(void) {
	Mat frame;
        char tmp_string[160];

	while(!::stopSig)
        {
          if (cap.grab())
          {
            if (cap.retrieve(frame))
            {
#if 0
                sprintf(tmp_string, "Input%03d", input_cnt);
                putText(frame, tmp_string,
                        Point(5, 20),
                        FONT_HERSHEY_COMPLEX_SMALL,
                        0.75,
                        Scalar(255,255,255), 1, 8);
#endif
		if ((frameStack_wr - frameStack_rd) < MAX_GRAB_SIZE) { 
		  // Insert new frame
                  m_grab.lock();
		  frame.copyTo(frameStack[frameStack_wr % MAX_GRAB_SIZE]); 
                  frameStack_wr ++;
                  m_grab.unlock();
                  show_ts("CAM_INSERT");
		} else {
		  // This line clears the stack when it is full...
                  show_ts("CAM_SKIP");
                  std::this_thread::sleep_for(std::chrono::milliseconds(10));
		}
            }
          } else {
            if(live_input == -1) {
              //Rewind!
              cap.set(CAP_PROP_POS_FRAMES, 0);
            }
          }
	}
	std::cout << "Esc key pressed by user" << std::endl;
	return;
}
//-----------------------------------------------------------------------------------------------------------------------------------------
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
//-----------------------------------------------------------------------------------------------------------------------------------------
bool is_big_endian() {
  int32_t n = 1;
  // big endian if true
  return (*(char *)&n == 0);
}
/*! \brief Get all class labels from the provided file (one per line)
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
void print_usage(int argc, char **argv)
{
  std::cout << "Usage: " << argv[0] << std::endl;
  std::cout << "  --model (-m)  ... path to compiled TVM model\n";
  std::cout << "  --clip  (-c)  ... path to video clip to be used as input\n";
  std::cout << "  --labels (-l) ... file with labels (mapping of class ID to human readable classes)\n";
  std::cout << "  --max_frame_count (-f) ... max count of frames to run\n";
  std::cout << "  --video_port (-p) ... set camera video port if using camera as video input\n";
  std::cout << "  --device (-d) ... set device to use, e.g. 'cpu', 'gpu', 'opencl'\n";
  std::cout << "  --batch_size (-b) ... set batch size for DLR inference (currently hardcoded to 4)\n";
  std::cout << "  --input_node (-i) ... name of TVM input node\n";
  std::cout << "  --write_display (-w) ... copy display output to .png file every Nth frame (number set by this option)\n";
  std::cout << "  --no_inference (-n)  ... (flag) turn off DLR inference, just to live cam input, display output. Useful for profiling\n";
  std::cout << "  --verbose (-v) ... (flag) show in stdout frame processing events (camera in, process, display)\n";
  std::cout << "  --help (-h) ... (flag) this help\n"; 
}
//------------------------------------------------------------------------------------------------------------------------------------------
int main(int argc, char** argv) {
bool clip_f = false;
const char *default_videoClip = "/usr/share/ti/tidl/examples/classification/clips/test10.mp4";
char video_clip[160];
bool labels_ok = false;
int  write_display_file_throttling = 0;
int device_type = 1; //'cpu' type is default
int c;

  if (is_big_endian()) {
    std::cerr << "Big endian not supported" << std::endl;
    return 1;
  }

  while (1) {
    int option_index = 0;
    static struct option long_options[] = {
      {"model",            required_argument, 0, 'm' },
      {"clip",             required_argument, 0, 'c' },
      {"labels",           required_argument, 0, 'l' },
      {"max_frame_count",  required_argument, 0, 'f' },
      {"video_port",       required_argument, 0, 'p' },
      {"device",           required_argument, 0, 'd' },
      {"batch_size",       required_argument, 0, 'b' },
      {"input_node",       required_argument, 0, 'i' },
      {"write_display",    required_argument, 0, 'w' },
      {"no_inference",     no_argument,       0, 'n' },
      {"verbose",          no_argument,       0, 'v' },
      {"help",             no_argument,       0, 'h' },
      {0,                  0,                 0,  0  }
    };

    c = getopt_long(argc, argv, "m:c:l:f:p:d:b:i:i:w:nvh", long_options, &option_index);
    if (c == -1)
      break;

    switch (c) {
      case 'm':
        strcpy(dlr_model, optarg);
        break;
      case 'c':
        clip_f = true;
        strcpy(video_clip, optarg);
        break;
      case 'l':
        labels_ok = getFileContent(optarg, vecOfLabels);
        break;
      case 'f':
        max_frameCount = atoi(optarg);
        break;
      case 'p':
        live_input = atoi(optarg);
        break;
      case 'd':
        if (strcmp(optarg, "cpu") == 0)
        {
          device_type = 1;
        } else if (strcmp(optarg, "gpu") == 0) {
          device_type = 2;
        } else if (strcmp(optarg, "opencl") == 0) {
          device_type = 4;
        } else {
          std::cerr << "Unsupported device type! " << optarg << std::endl;
          return 1;
        }
        break;
      case 'b':
        std::cout << "Batch size hardcoded to:" << BATCH_SIZE << std::endl;
        break;
      case 'i':
        input_name = std::string(optarg); 
        break;
      case 'n':
        do_inference = false;
        break;
      case 'w':
        write_display_file_throttling = atoi(optarg);
      case 'v':
        set_verbose = true;
        break;
      case 'h':
      case '?':
        print_usage(argc, argv);
        exit(EXIT_FAILURE);

      default:
        printf("!? getopt returned character code 0%o ??\n", c);
     }
  }
  std::cout << "DUMP CONFIGURATION:\n";
  std::cout << "Model:" << dlr_model << std::endl;
  std::cout << "Clip:"  << video_clip << std::endl;
  std::cout << "Labels? :" << labels_ok << std::endl;
  std::cout << "Max frame count:" << max_frameCount << std::endl;
  std::cout << "Live input port:" << live_input << std::endl;
  std::cout << "Device type:" << device_type << std::endl;
  std::cout << "Input node name:" << input_name << std::endl;
  std::cout << "Do inference?:" << do_inference << std::endl;

  if (CreateDLRModel(&model, dlr_model, device_type, 0) != 0) {
    LOG(INFO) << DLRGetLastError() << std::endl;
    throw std::runtime_error("Could not load DLR Model");
  }
  
  std::cout << "Running inference... " << std::endl;  

  SetupLiveDisplay();
  // Setup Input
  if (!SetupInput(clip_f ? video_clip : default_videoClip)) {
    std::cout << "..cannot open video input!\n";
    return -1;
  }
  //=========================================================================================
  Mat show_image;	
  int display_count = 0;
  int display_idle  = 0;

  start_ts = std::chrono::system_clock::now();
  // This thread does image preprocessing and DLR inference
  std::thread t_process (ProcessFrames);
  // In the beginning we run one dummy inference call which does model initialization
  std::this_thread::sleep_for(std::chrono::milliseconds(3000));
  // This thread captures the incoming frames (decoding from clip, or from camera)
  std::thread t_input (CollectFrames);
  while(1) {
	if((processedStack_wr - processedStack_rd) > 0)  {
            char tmp_string[160];
            char displayMsg[80];
            m_proc.lock();
            processedStack[processedStack_rd % MAX_PROC_SIZE].copyTo(show_image);
            processedStack_rd ++;
            m_proc.unlock();
            display_idle = 0;
            cv::imshow(imagenet_win, show_image); 
            if(write_display_file_throttling > 0)
            {
              if((display_count % write_display_file_throttling) == 0) {
                sprintf(tmp_string, "cam%03d.png", display_count);
                imwrite(tmp_string, show_image);       
              }
            }
            display_count ++;
            if(display_count >= max_frameCount) {
              std::cout << "DISPLAY STOPPING\n";
	      ::stopSig = 1;		// Signal to threads to end their run...
            }
            sprintf(displayMsg, "\t\tDISPLAY(%d):", display_idle);
            show_ts(displayMsg);
	} else display_idle++;
		
	if (waitKey(1) == 27)	//Is  'esc' key pressed?
	{
	  std::cout << "MAIN: esc key is pressed by user" << std::endl;
  	  ::stopSig = 1;  // Signal to threads to end their run...
	}

        if(::stopSig == 1) break; 
  }
  std::cout << "...stopping threads, first processFrame\n";
  t_process.join();
  std::cout << "...stopping threads, stop grabFrame\n";
  t_input.join();
  std::cout << "exit program!\n";
  return 0;
}
