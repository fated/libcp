#ifndef LIBCP_SVM_H_
#define LIBCP_SVM_H_

#include "utilities.h"

enum { C_SVC, NU_SVC };  // svm_type
enum { LINEAR, POLY, RBF, SIGMOID, PRECOMPUTED };  // kernel_type

struct SVMParameter {
  int svm_type;
  int kernel_type;
  int degree;  // for poly
  double gamma;  // for poly/rbf/sigmoid
  double coef0;  // for poly/sigmoid
  double cache_size; // in MB
  double eps;  // stopping criteria
  double C;  // for C_SVC
  int num_weights;  // for C_SVC
  int *weight_labels;  // for C_SVC
  double *weights;  // for C_SVC
  double nu;  // for NU_SVC
  int shrinking;  // use the shrinking heuristics
};

struct SVMModel {
  struct SVMParameter param;
  int num_ex;
  int num_classes;  // number of classes (k)
  int total_sv;  // total #SV
  struct Node **svs;  // SVs (SV[total_sv])
  double **sv_coef;  // coefficients for SVs in decision functions (sv_coef[k-1][total_sv])
  double *rho;  // constants in decision functions (rho[k*(k-1)/2])
  int *sv_indices;  // sv_indices[0,...,nSV-1] are values in [1,...,num_traning_data] to indicate SVs in the training set
  int *labels;  // label of each class (label[k])
  int *num_svs;  // number of SVs for each class (nSV[k])
                 // nSV[0] + nSV[1] + ... + nSV[k-1] = total_sv
  int free_sv;  // 1 if SVMModel is created by LoadSVMModel
                // 0 if SVMModel is created by TrainSVM
};

SVMModel *TrainSVM(const struct Problem *prob, const struct SVMParameter *param);
double PredictSVMValues(const struct SVMModel *model, const struct Node *x, double *decision_values);
double PredictSVM(const struct SVMModel *model, const struct Node *x);

int SaveSVMModel(std::ofstream &model_file, const struct SVMModel *model);
SVMModel *LoadSVMModel(std::ifstream &model_file);
void FreeSVMModel(struct SVMModel **model);

void FreeSVMParam(struct SVMParameter *param);
void InitSVMParam(struct SVMParameter *param);
const char *CheckSVMParameter(const struct SVMParameter *param);

void SetPrintNull();
void SetPrintCout();

#endif  // LIBCP_SVM_H_