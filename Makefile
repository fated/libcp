CXX ?= g++
CFLAGS = -Wall -Wconversion -O3 -fPIC
SHVER = 2
OS = $(shell uname)

all: cp-offline cp-online cp-cv

cp-offline: cp-offline.cpp utilities.o knn.o svm.o cp.o
	$(CXX) $(CFLAGS) cp-offline.cpp utilities.o knn.o svm.o cp.o -o cp-offline -lm

cp-online: cp-online.cpp utilities.o knn.o svm.o cp.o
	$(CXX) $(CFLAGS) cp-online.cpp utilities.o knn.o svm.o cp.o -o cp-online -lm

cp-cv: cp-cv.cpp utilities.o knn.o svm.o cp.o
	$(CXX) $(CFLAGS) cp-cv.cpp utilities.o knn.o svm.o cp.o -o cp-cv -lm

utilities.o: utilities.cpp utilities.h
	$(CXX) $(CFLAGS) -c utilities.cpp

svm.o: knn.cpp knn.h
	$(CXX) $(CFLAGS) -c knn.cpp

svm.o: svm.cpp svm.h
	$(CXX) $(CFLAGS) -c svm.cpp

cp.o: cp.cpp cp.h
	$(CXX) $(CFLAGS) -c cp.cpp

clean:
	rm -f utilities.o knn.o svm.o cp.o cp-offline cp-online cp-cv