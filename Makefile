CXX ?= g++
CFLAGS = -Wall -Wconversion -O3 -fPIC
SHVER = 2
OS = $(shell uname)

all: cp-offline

cp-offline: cp-offline.cpp utilities.o knn.o cp.o
	$(CXX) $(CFLAGS) cp-offline.cpp utilities.o knn.o cp.o -o cp-offline -lm

utilities.o: utilities.cpp utilities.h
	$(CXX) $(CFLAGS) -c utilities.cpp

svm.o: knn.cpp knn.h
	$(CXX) $(CFLAGS) -c knn.cpp

cp.o: cp.cpp cp.h
	$(CXX) $(CFLAGS) -c cp.cpp

clean:
	rm -f utilities.o knn.o cp.o cp-offline
