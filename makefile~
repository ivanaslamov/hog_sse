
all: test leonid

leonid: leonid.cpp hog_sse.o
	g++ hog_sse.o leonid.cpp -O3 -o leonid -lopencv_core -lopencv_highgui -lopencv_imgproc -msse4.1

test: test.cpp hog_sse.o
	g++ hog_sse.o test.cpp -O3 -o test -lopencv_core -lopencv_highgui -lopencv_imgproc -msse4.1

hog_sse.o: hog_sse.cpp
	g++ -O3 -msse4.1 -c hog_sse.cpp -o hog_sse.o

clean:
	rm *.o test leonid

