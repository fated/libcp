#ifndef LIBCP_KNN_H_
#define LIBCP_KNN_H_

#include "utilities.h"

struct KNNParameter {
  int num_neighbors;
};

struct KNNModel {
  struct KNNParameter param;
  int num_ex;
  int num_classes;
  int *labels;  // label of each class (label[k])
  double **dist_neighbors;
  int **label_neighbors;
};

template <typename T>
static inline void InsertLabel(T *labels, T label, int num_neighbors, int index) {
  for (int i = num_neighbors-1; i > index; --i)
    labels[i] = labels[i-1];
  labels[index] = label;

  return;
}

KNNModel *TrainKNN(const struct Problem *prob, const struct KNNParameter *param);
double PredictKNN(struct Problem *train, struct Node *x, const int num_neighbors);
double CalcDist(const struct Node *x1, const struct Node *x2);
int CompareDist(double *neighbors, double dist, int num_neighbors);

int SaveKNNModel(std::ofstream &model_file, const struct KNNModel *model);
KNNModel *LoadKNNModel(std::ifstream &model_file);
void FreeKNNModel(struct KNNModel *model);

void FreeKNNParam(struct KNNParameter *param);
void InitKNNParam(struct KNNParameter *param);
const char *CheckKNNParameter(const struct KNNParameter *param);

#endif  // LIBCP_KNN_H_
