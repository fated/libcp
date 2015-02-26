#include "cp.h"
#include <iostream>
#include <fstream>
#include <cmath>
#include <random>

static double CalcAlpha(double *min_same, double *min_diff, int num_neighbors) {
  double alpha;
  double sum_same = 0, sum_diff = 0;

  for (int i = 0; i < num_neighbors; ++i) {
    sum_same += min_same[i];
    sum_diff += min_diff[i];
  }
  if (sum_diff == 0.0) {
    if (sum_same == 0.0) {
      alpha = 0.0;
    } else {
      alpha = kInf;
    }
  } else {
    alpha = sum_same / sum_diff;
  }

  return alpha;
}

Model *TrainCP(const struct Problem *train, const struct Parameter *param) {
  Model *model = new Model;
  model->param = *param;
  int num_ex = train->num_ex;

  if (param->measure_type == KNN) {
    model->knn_model = TrainKNN(train, param->knn_param);

    int num_neighbors = param->knn_param->num_neighbors;
    int num_classes = model->knn_model->num_classes;
    double *alpha = new double[num_ex];

    for (int i = 0; i < num_ex; ++i) {
      alpha[i] = CalcAlpha(model->knn_model->min_same[i], model->knn_model->min_diff[i], num_neighbors);
    }

    model->num_classes = num_classes;
    model->num_ex = num_ex;
    model->alpha = alpha;
    clone(model->labels, model->knn_model->labels, num_classes);
  }

  return model;
}

std::vector<int> PredictCP(const struct Problem *train, const struct Model *model, const struct Node *x, double &conf, double &cred) {
  const Parameter& param = model->param;
  int num_ex = model->num_ex;
  int num_classes = model->num_classes;
  int *labels = model->labels;
  std::vector<int> predict_label;
  int *p_values = new int[num_classes];

  if (param.measure_type == KNN) {
    int num_neighbors = param.knn_param->num_neighbors;

    for (int i = 0; i < num_classes; ++i) {
      int y = labels[i];
      p_values[i] = 0;

      double **min_same = new double*[num_ex+1];
      double **min_diff = new double*[num_ex+1];
      double *alpha = new double[num_ex+1];

      for (int j = 0; j < num_ex; ++j) {
        clone(min_same[j], model->knn_model->min_same[j], num_neighbors);
        clone(min_diff[j], model->knn_model->min_diff[j], num_neighbors);
        alpha[j] = model->alpha[j];
      }
      min_same[num_ex] = new double[num_neighbors];
      min_diff[num_ex] = new double[num_neighbors];
      for (int j = 0; j < num_neighbors; ++j) {
        min_same[num_ex][j] = kInf;
        min_diff[num_ex][j] = kInf;
      }
      alpha[num_ex] = 0;

      for (int j = 0; j < num_ex; ++j) {
        double dist = CalcDist(train->x[j], x);

        if (train->y[j] == y) {
          int index = CompareDist(min_same[j], dist, num_neighbors);
          if (index < num_neighbors) {
            alpha[j] = CalcAlpha(min_same[j], min_diff[j], num_neighbors);
          }
          CompareDist(min_same[num_ex], dist, num_neighbors);
        } else {
          int index = CompareDist(min_diff[j], dist, num_neighbors);
          if (index < num_neighbors) {
            alpha[j] = CalcAlpha(min_same[j], min_diff[j], num_neighbors);
          }
          CompareDist(min_diff[num_ex], dist, num_neighbors);
        }
      }
      alpha[num_ex] = CalcAlpha(min_same[num_ex], min_diff[num_ex], num_neighbors);

      for (int j = 0; j < num_ex+1; ++j) {
        if (alpha[j] >= alpha[num_ex]) {
          ++p_values[i];
        }
      }

      for (int j = 0; j < num_ex+1; ++j) {
        delete[] min_same[j];
        delete[] min_diff[j];
      }
      delete[] min_same;
      delete[] min_diff;
      delete[] alpha;
    }

  }

  int best = 0;
  cred = p_values[0];
  conf = p_values[1];
  for (int i = 1; i < num_classes; ++i) {
    if (p_values[i] > cred) {
      conf = cred;
      cred = p_values[i];
      best = i;
    } else if (p_values[i] < cred && p_values[i] > conf) {
      conf = p_values[i];
    }
  }
  cred = cred / (num_ex+1);
  conf = 1 - conf / (num_ex+1);

  predict_label.push_back(labels[best]);
  for (int i = 0; i < num_classes; ++i) {
    if (static_cast<double>(p_values[i])/(num_ex+1) > param.epsilon) {
      predict_label.push_back(labels[i]);
    }
  }

  delete[] p_values;

  return predict_label;
}

void CrossValidation(const struct Problem *prob, const struct Parameter *param,
    std::vector<int> *predict_labels, double *conf, double *cred) {
  int num_folds = param->num_folds;
  int num_ex = prob->num_ex;
  int num_classes;

  int *fold_start;
  int *perm = new int[num_ex];

  if (num_folds > num_ex) {
    num_folds = num_ex;
    std::cerr << "WARNING: number of folds > number of data. Will use number of folds = number of data instead (i.e., leave-one-out cross validation)" << std::endl;
  }
  fold_start = new int[num_folds+1];

  if (num_folds < num_ex) {
    int *start = NULL;
    int *label = NULL;
    int *count = NULL;
    GroupClasses(prob, &num_classes, &label, &start, &count, perm);

    int *fold_count = new int[num_folds];
    int *index = new int[num_ex];

    for (int i = 0; i < num_ex; ++i) {
      index[i] = perm[i];
    }
    std::random_device rd;
    std::mt19937 g(rd());
    for (int i = 0; i < num_classes; ++i) {
      std::shuffle(index+start[i], index+start[i]+count[i], g);
    }

    for (int i = 0; i < num_folds; ++i) {
      fold_count[i] = 0;
      for (int c = 0; c < num_classes; ++c) {
        fold_count[i] += (i+1)*count[c]/num_folds - i*count[c]/num_folds;
      }
    }

    fold_start[0] = 0;
    for (int i = 1; i <= num_folds; ++i) {
      fold_start[i] = fold_start[i-1] + fold_count[i-1];
    }
    for (int c = 0; c < num_classes; ++c) {
      for (int i = 0; i < num_folds; ++i) {
        int begin = start[c] + i*count[c]/num_folds;
        int end = start[c] + (i+1)*count[c]/num_folds;
        for (int j = begin; j < end; ++j) {
          perm[fold_start[i]] = index[j];
          fold_start[i]++;
        }
      }
    }
    fold_start[0] = 0;
    for (int i = 1; i <= num_folds; ++i) {
      fold_start[i] = fold_start[i-1] + fold_count[i-1];
    }
    delete[] start;
    delete[] label;
    delete[] count;
    delete[] index;
    delete[] fold_count;

  } else {

    for (int i = 0; i < num_ex; ++i) {
      perm[i] = i;
    }
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(perm, perm+num_ex, g);
    fold_start[0] = 0;
    for (int i = 1; i <= num_folds; ++i) {
      fold_start[i] = fold_start[i-1] + (i+1)*num_ex/num_folds - i*num_ex/num_folds;
    }
  }

  for (int i = 0; i < num_folds; ++i) {
    int begin = fold_start[i];
    int end = fold_start[i+1];
    int k = 0;
    struct Problem subprob;

    subprob.num_ex = num_ex - (end-begin);
    subprob.x = new Node*[subprob.num_ex];
    subprob.y = new double[subprob.num_ex];

    for (int j = 0; j < begin; ++j) {
      subprob.x[k] = prob->x[perm[j]];
      subprob.y[k] = prob->y[perm[j]];
      ++k;
    }
    for (int j = end; j < num_ex; ++j) {
      subprob.x[k] = prob->x[perm[j]];
      subprob.y[k] = prob->y[perm[j]];
      ++k;
    }

    Model *submodel = TrainCP(&subprob, param);

    for (int j = begin; j < end; ++j) {
      predict_labels[perm[j]] = PredictCP(&subprob, submodel, prob->x[perm[j]], conf[perm[j]], cred[perm[j]]);
    }
    FreeModel(submodel);
    delete[] subprob.x;
    delete[] subprob.y;
  }
  delete[] fold_start;
  delete[] perm;

  return;
}

static const char *kMeasureTypeTable[] = { "knn", NULL };

int SaveModel(const char *model_file_name, const struct Model *model) {
  std::ofstream model_file(model_file_name);
  if (!model_file.is_open()) {
    std::cerr << "Unable to open model file: " << model_file_name << std::endl;
    return -1;
  }

  const Parameter &param = model->param;

  model_file << "measure_type " << kMeasureTypeTable[param.measure_type] << '\n';
  model_file << "epsilon " << param.epsilon << '\n';

  if (param.measure_type == KNN) {
    SaveKNNModel(model_file, model->knn_model);
  }

  if (model->alpha) {
    model_file << "alpha\n";
    for (int i = 0; i < model->num_ex; ++i) {
      model_file << model->alpha[i] << ' ';
    }
    model_file << '\n';
  }

  if (model_file.bad() || model_file.fail()) {
    model_file.close();
    return -1;
  }

  model_file.close();

  return 0;
}

Model *LoadModel(const char *model_file_name) {
  std::ifstream model_file(model_file_name);
  if (!model_file.is_open()) {
    std::cerr << "Unable to open model file: " << model_file_name << std::endl;
    return NULL;
  }

  Model *model = new Model;

  Parameter &param = model->param;
  param.load_model = 1;
  model->labels = NULL;
  model->alpha = NULL;

  char cmd[80];
  while (1) {
    model_file >> cmd;

    if (std::strcmp(cmd, "measure_type") == 0) {
      model_file >> cmd;
      int i;
      for (i = 0; kMeasureTypeTable[i]; ++i) {
        if (std::strcmp(kMeasureTypeTable[i], cmd) == 0) {
          param.measure_type = i;
          break;
        }
      }
      if (kMeasureTypeTable[i] == NULL) {
        std::cerr << "Unknown measure type.\n" << std::endl;
        FreeModel(model);
        delete model;
        model_file.close();
        return NULL;
      }
    } else
    if (std::strcmp(cmd, "epsilon") == 0) {
      model_file >> param.epsilon;
    } else
    if (std::strcmp(cmd, "knn_model") == 0) {
      model->knn_model = LoadKNNModel(model_file);
      if (model->knn_model == NULL) {
        FreeModel(model);
        delete model;
        model_file.close();
        return NULL;
      }
      model->num_ex = model->knn_model->num_ex;
      model->num_classes = model->knn_model->num_classes;
      clone(model->labels, model->knn_model->labels, model->num_classes);
      model->param.knn_param = &model->knn_model->param;
    } else
    if (std::strcmp(cmd, "alpha") == 0) {
      int num_ex = model->num_ex;
      model->alpha = new double[num_ex];
      for (int i = 0; i < num_ex; ++i) {
        model_file >> model->alpha[i];
      }
      break;
    } else {
      std::cerr << "Unknown text in model file: " << cmd << std::endl;
      FreeModel(model);
      delete model;
      model_file.close();
      return NULL;
    }
  }
  model_file.close();

  return model;
}

void FreeModel(struct Model *model) {
  if (model->param.measure_type == KNN &&
      model->knn_model != NULL) {
    FreeKNNModel(model->knn_model);
    model->knn_model = NULL;
  }

  if (model->labels != NULL) {
    delete[] model->labels;
    model->labels = NULL;
  }

  if (model->alpha != NULL) {
    delete[] model->alpha;
    model->alpha = NULL;
  }

  delete model;
  model = NULL;

  return;
}

void FreeParam(struct Parameter *param) {
  if (param->measure_type == KNN &&
      param->knn_param != NULL) {
    FreeKNNParam(param->knn_param);
    param->knn_param = NULL;
  }

  return;
}

const char *CheckParameter(const struct Parameter *param) {
  if (param->save_model == 1 && param->load_model == 1) {
    return "cannot save and load model at the same time";
  }

  if (param->epsilon < 0 || param->epsilon > 1) {
    return "epsilon should be between 0 and 1";
  }

  if (param->measure_type == KNN) {
    if (param->knn_param == NULL) {
      return "no knn parameter";
    }
    return CheckKNNParameter(param->knn_param);
  }

  if (param->measure_type > 0) {
    return "no such taxonomy type";
  }

  return NULL;
}