#ifndef LIBCP_CP_H_
#define LIBCP_CP_H_

#include "utilities.h"
#include "knn.h"
#include <vector>

enum { KNN };

struct Parameter {
  struct KNNParameter *knn_param;
  int cp_type;
  int measure_type;
  int save_model;
  int load_model;
  int num_folds;
  double epsilon;
};

struct Model {
  struct Parameter param;
  struct KNNModel *knn_model;
  int num_ex;
  int num_classes;
  int *labels;
  double *alpha;
};

Model *TrainCP(const struct Problem *train, const struct Parameter *param);
std::vector<int> PredictCP(const struct Problem *train, const struct Model *model, const struct Node *x, double &conf, double &cred);
void CrossValidation(const struct Problem *prob, const struct Parameter *param, std::vector<int> *predict_labels, double *conf, double *cred);
void OnlinePredict(const struct Problem *prob, const struct Parameter *param, std::vector<int> *predict_labels, int *indices, double *conf, double *cred);

int SaveModel(const char *model_file_name, const struct Model *model);
Model *LoadModel(const char *model_file_name);
void FreeModel(struct Model *model);

void FreeParam(struct Parameter *param);
const char *CheckParameter(const struct Parameter *param);

#endif  // LIBCP_CP_H_