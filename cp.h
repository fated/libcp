#ifndef LIBCP_CP_H_
#define LIBCP_CP_H_

#include "utilities.h"
#include "knn.h"
#include "svm.h"

enum { KNN, SVM_EL, SVM_ES, SVM_KM };

struct Parameter {
  struct KNNParameter *knn_param;
  struct SVMParameter *svm_param;
  int save_model;
  int load_model;
  int measure_type;
  int num_folds;
  int probability;
};

struct Model {
  struct Parameter param;
  struct SVMModel *svm_model;
  struct KNNModel *knn_model;
  int num_ex;
  int num_classes;
  int *labels;
};

Model *TrainCP(const struct Problem *train, const struct Parameter *param);
double PredictCP(const struct Problem *train, const struct Model *model, const struct Node *x, double &lower, double &upper, double **avg_prob);
void CrossValidation(const struct Problem *prob, const struct Parameter *param, double *predict_labels, double *lower_bounds, double *upper_bounds, double *brier, double *logloss);
void OnlinePredict(const struct Problem *prob, const struct Parameter *param, double *predict_labels, int *indices, double *lower_bounds, double *upper_bounds, double *brier, double *logloss);

int SaveModel(const char *model_file_name, const struct Model *model);
Model *LoadModel(const char *model_file_name);
void FreeModel(struct Model *model);

void FreeParam(struct Parameter *param);
const char *CheckParameter(const struct Parameter *param);

#endif  // LIBCP_CP_H_