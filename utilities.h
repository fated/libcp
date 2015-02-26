#ifndef LIBCP_UTILITIES_H_
#define LIBCP_UTILITIES_H_

#include <iostream>
#include <vector>
#include <map>
#include <cmath>
#include <cstdarg>

const double kInf = HUGE_VAL;
const double kTau = 1e-12;
const double kEpsilon = 1e-15;

struct Node {
  int index;
  double value;
};

struct Problem {
  int num_ex;  // number of examples
  int max_index;
  double *y;
  struct Node **x;
};

void PrintCout(const char *s);
void PrintNull(const char *s);
void Info(const char *format, ...);
void SetPrintNull();
void SetPrintCout();

template <typename T>
T FindMostFrequent(T *array, int size) {
  std::vector<T> v(array, array+size);
  std::map<T, int> frequency_map;
  int max_frequency = 0;
  T most_frequent_element;

  for (std::size_t i = 0; i < v.size(); ++i) {
    if (v[i] != -1) {
      ++frequency_map[v[i]];
    }
    int cur_frequency = frequency_map[v[i]];
    if (cur_frequency > max_frequency) {
      max_frequency = cur_frequency;
      most_frequent_element = v[i];
    }
  }

  return most_frequent_element;
}

template <typename T, typename S>
static inline void clone(T *&dest, S *src, int size) {
  dest = new T[size];
  if (sizeof(T) < sizeof(S))
    std::cerr << "WARNING: destination type is smaller than source type, data will be truncated." << std::endl;
  std::copy(src, src+size, dest);

  return;
}

template <typename T>
void QuickSortIndex(T array[], size_t index[], size_t left, size_t right) {
  size_t i = left, j = right;
  size_t p = left + (right-left)/2;
  size_t ind = index[p];
  T pivot = array[p];
  for ( ; i < j; ) {
    while ((i < p) && (pivot >= array[i]))
      ++i;
    if (i < p) {
      array[p] = array[i];
      index[p] = index[i];
      p = i;
    }
    while ((j > p) && (array[j] >= pivot))
      --j;
    if (j > p) {
      array[p] = array[j];
      index[p] = index[j];
      p = j;
    }
  }
  array[p] = pivot;
  index[p] = ind;
  if (p - left > 1)
    QuickSortIndex(array, index, left, p - 1);
  if (right - p > 1)
    QuickSortIndex(array, index, p + 1, right);

  return;
}

Problem *ReadProblem(const char *file_name);
void FreeProblem(struct Problem *problem);
void GroupClasses(const Problem *prob, int *num_classes_ret, int **labels_ret, int **start_ret, int **count_ret, int *perm);
int *GetLabels(const Problem *prob, int *num_classes_ret);

#endif  // LIBCP_UTILITIES_H_