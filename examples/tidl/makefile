PHONY: all
CXXFLAGS =-funroll-loops -O3 -DNDEBUG 
all: neo-ai-dlr-demo
LIBS = -ldlr -lpthread -lopencv_highgui -lopencv_imgcodecs -lopencv_videoio -lopencv_imgproc -lopencv_core -lrt -ldl 

neo-ai-dlr-demo: run_mobilenet_cv_mt.cc
	$(CXX) $(CXXFLAGS) run_mobilenet_cv_mt.cc -o run_mobilenet_cv_mt $(LIBS)

clean:
	rm -rf run_mobilenet_cv_mt

