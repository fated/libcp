#include "cp.h"
#include <iostream>
#include <fstream>
#include <cmath>
#include <random>

Model *TrainCP(const struct Problem *train, const struct Parameter *param) {
  Model *model = new Model;
  model->param = *param;
  int num_ex = train->num_ex;

  if (param->taxonomy_type == KNN) {
    int num_neighbors = param->knn_param->num_neighbors;

    int *categories = new int[num_ex];
    for (int i = 0; i < num_ex; ++i) {
      categories[i] = -1;
    }

    model->knn_model = TrainKNN(train, param->knn_param);

    int num_classes = model->knn_model->num_classes;
    int num_categories = param->num_categories;
    if (num_categories != num_classes) {
      std::cerr << "WARNING: number of categories should be the same as number of classes in KNN. See README for details." << std::endl;
      num_categories = num_classes;
    }

    for (int i = 0; i < num_ex; ++i) {
      categories[i] = FindMostFrequent(model->knn_model->label_neighbors[i], num_neighbors);
    }

    model->num_classes = num_classes;
    model->num_ex = num_ex;
    model->num_categories = num_categories;
    model->categories = categories;
    clone(model->labels, model->knn_model->labels, num_classes);
  }

  if (param->taxonomy_type == SVM_EL ||
      param->taxonomy_type == SVM_ES ||
      param->taxonomy_type == SVM_KM) {
    int num_categories = param->num_categories;
    int *categories = new int[num_ex];
    double *combined_decision_values = new double[num_ex];

    for (int i = 0; i < num_ex; ++i) {
      categories[i] = -1;
      combined_decision_values[i] = 0;
    }

    model->svm_model = TrainSVM(train, param->svm_param);

    int num_classes = model->svm_model->num_classes;
    if (num_classes == 1) {
      std::cerr << "WARNING: training set only has one class. See README for details." << std::endl;
    }
    if (num_classes > 2 && num_categories < num_classes) {
      std::cerr << "WARNING: number of categories should be the same as number of classes in Multi-Class case. See README for details." << std::endl;
      num_categories = num_classes;
    }

    for (int i = 0; i < num_ex; ++i) {
      double *decision_values = new double[num_classes*(num_classes-1)/2];
      int label = 0;
      double predict_label = PredictSVMValues(model->svm_model, train->x[i], decision_values);

      for (int j = 0; j < num_classes; ++j) {
        if (predict_label == model->svm_model->labels[j]) {
          label = j;
          break;
        }
      }
      combined_decision_values[i] = CalcCombinedDecisionValues(decision_values, num_classes, label);
      delete[] decision_values;
    }

    if (param->taxonomy_type == SVM_EL) {
      for (int i = 0; i < num_ex; ++i) {
        categories[i] = GetEqualLengthCategory(combined_decision_values[i], num_categories, num_classes);
      }
    }
    if (param->taxonomy_type == SVM_ES) {
      if (num_classes == 1) {
        for (int i = 0; i < num_ex; ++i) {
          categories[i] = 0;
        }
        model->points = new double[num_categories];
        for (int i = 0; i < num_categories; ++i) {
          model->points[i] = 0;
        }
      } else {
        double *points;
        points = GetEqualSizeCategory(combined_decision_values, categories, num_categories, num_ex);
        clone(model->points, points, num_categories);
        delete[] points;
      }
    }
    if (param->taxonomy_type == SVM_KM) {
      double *points;
      points = GetKMeansCategory(combined_decision_values, categories, num_categories, num_ex, kEpsilon);
      clone(model->points, points, num_categories);
      delete[] points;
    }
    delete[] combined_decision_values;
    model->num_classes = num_classes;
    model->num_ex = num_ex;
    model->categories = categories;
    model->num_categories = num_categories;
    clone(model->labels, model->svm_model->labels, num_classes);
  }

  return model;
}

double PredictCP(const struct Problem *train, const struct Model *model, const struct Node *x, double &lower, double &upper, double **avg_prob) {
  const Parameter& param = model->param;
  int num_ex = model->num_ex;
  int num_classes = model->num_classes;
  int num_categories = model->num_categories;
  int *labels = model->labels;
  double predict_label;
  int **f_matrix = new int*[num_classes];
  int *alter_labels = new int[num_ex];

  for (int i = 0; i < num_classes; ++i) {
    for (int j = 0; j < num_ex; ++j) {
      if (labels[i] == train->y[j]) {
        alter_labels[j] = i;
      }
    }
  }

  if (param.taxonomy_type == KNN) {
    int num_neighbors = param.knn_param->num_neighbors;
    for (int i = 0; i < num_classes; ++i) {
      int *categories = new int[num_ex+1];
      double **dist_neighbors = new double*[num_ex+1];
      int **label_neighbors = new int*[num_ex+1];
      f_matrix[i] = new int[num_classes];
      for (int j = 0; j < num_classes; ++j) {
        f_matrix[i][j] = 0;
      }

      for (int j = 0; j < num_ex; ++j) {
        clone(dist_neighbors[j], model->knn_model->dist_neighbors[j], num_neighbors);
        clone(label_neighbors[j], model->knn_model->label_neighbors[j], num_neighbors);
        categories[j] = model->categories[j];
      }
      dist_neighbors[num_ex] = new double[num_neighbors];
      label_neighbors[num_ex] = new int[num_neighbors];
      for (int j = 0; j < num_neighbors; ++j) {
        dist_neighbors[num_ex][j] = kInf;
        label_neighbors[num_ex][j] = -1;
      }
      categories[num_ex] = -1;

      for (int j = 0; j < num_ex; ++j) {
        double dist = CalcDist(train->x[j], x);
        int index;
        index = CompareDist(dist_neighbors[j], dist, num_neighbors);
        if (index < num_neighbors) {
          InsertLabel(label_neighbors[j], i, num_neighbors, index);
        }
        index = CompareDist(dist_neighbors[num_ex], dist, num_neighbors);
        if (index < num_neighbors) {
          InsertLabel(label_neighbors[num_ex], alter_labels[j], num_neighbors, index);
        }
      }

      for (int j = 0; j < num_ex+1; ++j) {
        categories[j] = FindMostFrequent(label_neighbors[j], num_neighbors);
      }

      for (int j = 0; j < num_ex; ++j) {
        if (categories[j] == categories[num_ex]) {
          ++f_matrix[i][alter_labels[j]];
        }
      }
      f_matrix[i][i]++;

      for (int j = 0; j < num_ex+1; ++j) {
        delete[] dist_neighbors[j];
        delete[] label_neighbors[j];
      }

      delete[] dist_neighbors;
      delete[] label_neighbors;
      delete[] categories;
    }
  }

  if (param.taxonomy_type == SVM_EL ||
      param.taxonomy_type == SVM_ES ||
      param.taxonomy_type == SVM_KM) {
    for (int i = 0; i < num_classes; ++i) {
      int *categories = new int[num_ex+1];
      f_matrix[i] = new int[num_classes];
      for (int j = 0; j < num_classes; ++j) {
        f_matrix[i][j] = 0;
      }

      for (int j = 0; j < num_ex; ++j) {
        categories[j] = model->categories[j];
      }
      categories[num_ex] = -1;

      double *decision_values = new double[num_classes*(num_classes-1)/2];
      int label = 0;
      double predict_label = PredictSVMValues(model->svm_model, x, decision_values);
      for (int j = 0; j < num_classes; ++j) {
        if (predict_label == labels[j]) {
          label = j;
          break;
        }
      }
      double combined_decision_values = CalcCombinedDecisionValues(decision_values, num_classes, label);
      if (param.taxonomy_type == SVM_EL) {
        categories[num_ex] = GetEqualLengthCategory(combined_decision_values, num_categories, num_classes);
      }
      if (param.taxonomy_type == SVM_ES) {
        if (num_classes == 1) {
          categories[num_ex] = 0;
        } else {
          int j;
          for (j = 0; j < num_categories; ++j) {
            if (combined_decision_values <= model->points[j]) {
              categories[num_ex] = j;
              break;
            }
          }
          if (j == num_categories) {
            categories[num_ex] = num_categories - 1;
          }
        }
      }
      if (param.taxonomy_type == SVM_KM) {
        categories[num_ex] = AssignCluster(num_categories, combined_decision_values, model->points);
      }
      delete[] decision_values;
      for (int j = 0; j < num_ex; ++j) {
        if (categories[j] == categories[num_ex]) {
          ++f_matrix[i][alter_labels[j]];
        }
      }
      f_matrix[i][i]++;

      delete[] categories;
    }
  }

  double **matrix = new double*[num_classes];
  for (int i = 0; i < num_classes; ++i) {
    matrix[i] = new double[num_classes];
    int sum = 0;
    for (int j = 0; j < num_classes; ++j) {
      sum += f_matrix[i][j];
    }
    for (int j = 0; j < num_classes; ++j) {
      matrix[i][j] = static_cast<double>(f_matrix[i][j]) / sum;
    }
  }

  double *quality = new double[num_classes];
  *avg_prob = new double[num_classes];
  for (int j = 0; j < num_classes; ++j) {
    quality[j] = matrix[0][j];
    (*avg_prob)[j] = matrix[0][j];
    for (int i = 1; i < num_classes; ++i) {
      if (matrix[i][j] < quality[j]) {
        quality[j] = matrix[i][j];
      }
      (*avg_prob)[j] += matrix[i][j];
    }
    (*avg_prob)[j] /= num_classes;
  }

  int best = 0;
  for (int i = 1; i < num_classes; ++i) {
    if (quality[i] > quality[best]) {
      best = i;
    }
  }

  lower = quality[best];
  upper = matrix[0][best];
  for (int i = 1; i < num_classes; ++i) {
    if (matrix[i][best] > upper) {
      upper = matrix[i][best];
    }
  }

  predict_label = labels[best];

  delete[] alter_labels;
  delete[] quality;
  for (int i = 0; i < num_classes; ++i) {
    delete[] f_matrix[i];
    delete[] matrix[i];
  }
  delete[] f_matrix;
  delete[] matrix;

  return predict_label;
}

void CrossValidation(const struct Problem *prob, const struct Parameter *param,
    double *predict_labels, double *lower_bounds, double *upper_bounds,
    double *brier, double *logloss) {
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

    struct Model *submodel = TrainCP(&subprob, param);

    if (param->probability == 1) {
      for (int j = 0; j < submodel->num_classes; ++j) {
        std::cout << submodel->labels[j] << "        ";
      }
      std::cout << '\n';
    }

    for (int j = begin; j < end; ++j) {
      double *avg_prob = NULL;
      brier[perm[j]] = 0;

      predict_labels[perm[j]] = PredictCP(&subprob, submodel, prob->x[perm[j]], lower_bounds[perm[j]], upper_bounds[perm[j]], &avg_prob);

      for (k = 0; k < submodel->num_classes; ++k) {
        if (submodel->labels[k] == prob->y[perm[j]]) {
          brier[perm[j]] += (1-avg_prob[k]) * (1-avg_prob[k]);
          double tmp = std::fmax(std::fmin(avg_prob[k], 1-kEpsilon), kEpsilon);
          logloss[perm[j]] = - std::log(tmp);
        } else {
          brier[perm[j]] += avg_prob[k] * avg_prob[k];
        }
      }
      if (param->probability == 1) {
        for (k = 0; k < submodel->num_classes; ++k) {
          std::cout << avg_prob[k] << ' ';
        }
        std::cout << '\n';
      }
      delete[] avg_prob;
    }
    FreeModel(submodel);
    delete[] subprob.x;
    delete[] subprob.y;
  }
  delete[] fold_start;
  delete[] perm;

  return;
}

void OnlinePredict(const struct Problem *prob, const struct Parameter *param,
    double *predict_labels, int *indices,
    double *lower_bounds, double *upper_bounds,
    double *brier, double *logloss) {
  int num_ex = prob->num_ex;
  int num_classes = 0;

  for (int i = 0; i < num_ex; ++i) {
    indices[i] = i;
  }
  std::random_device rd;
  std::mt19937 g(rd());
  std::shuffle(indices, indices+num_ex, g);

  if (param->taxonomy_type == KNN) {
    int num_neighbors = param->knn_param->num_neighbors;
    int *alter_labels = new int[num_ex];
    std::vector<int> labels;

    int *categories = new int[num_ex];
    double **dist_neighbors = new double*[num_ex];
    int **label_neighbors = new int*[num_ex];

    for (int i = 0; i < num_ex; ++i) {
      dist_neighbors[i] = new double[num_neighbors];
      label_neighbors[i] = new int[num_neighbors];
      for (int j = 0; j < num_neighbors; ++j) {
        dist_neighbors[i][j] = kInf;
        label_neighbors[i][j] = -1;
      }
      categories[i] = -1;
    }

    int this_label = static_cast<int>(prob->y[indices[0]]);
    labels.push_back(this_label);
    alter_labels[0] = 0;
    num_classes = 1;

    for (int i = 1; i < num_ex; ++i) {
      if (num_classes == 1)
        std::cerr <<
          "WARNING: training set only has one class. See README for details."
                  << std::endl;

      int **f_matrix = new int*[num_classes];

      for (int j = 0; j < num_classes; ++j) {
        f_matrix[j] = new int[num_classes];
        for (int k = 0; k < num_classes; ++k) {
          f_matrix[j][k] = 0;
        }

        double **dist_neighbors_ = new double*[i+1];
        int **label_neighbors_ = new int*[i+1];

        for (int j = 0; j < i; ++j) {
          clone(dist_neighbors_[j], dist_neighbors[j], num_neighbors);
          clone(label_neighbors_[j], label_neighbors[j], num_neighbors);
        }
        dist_neighbors_[i] = new double[num_neighbors];
        label_neighbors_[i] = new int[num_neighbors];
        for (int j = 0; j < num_neighbors; ++j) {
          dist_neighbors_[i][j] = kInf;
          label_neighbors_[i][j] = -1;
        }

        for (int k = 0; k < i; ++k) {
          double dist = CalcDist(prob->x[indices[k]], prob->x[indices[i]]);
          int index;

          index = CompareDist(dist_neighbors_[i], dist, num_neighbors);
          if (index < num_neighbors) {
            InsertLabel(label_neighbors_[i], alter_labels[k], num_neighbors, index);
          }
          index = CompareDist(dist_neighbors_[k], dist, num_neighbors);
          if (index < num_neighbors) {
            InsertLabel(label_neighbors_[k], j, num_neighbors, index);
          }
        }
        for (int k = 0; k <= i; ++k) {
          categories[k] = FindMostFrequent(label_neighbors_[k], num_neighbors);
        }

        for (int k = 0; k < i; ++k) {
          if (categories[k] == categories[i]) {
            ++f_matrix[j][alter_labels[k]];
          }
        }
        f_matrix[j][j]++;

        for (int j = 0; j < num_neighbors; ++j) {
          dist_neighbors[i][j] = dist_neighbors_[i][j];
          label_neighbors[i][j] = label_neighbors_[i][j];
        }
        for (int j = 0; j < i+1; ++j) {
          delete[] dist_neighbors_[j];
          delete[] label_neighbors_[j];
        }
        delete[] dist_neighbors_;
        delete[] label_neighbors_;
      }

      double **matrix = new double*[num_classes];
      for (int j = 0; j < num_classes; ++j) {
        matrix[j] = new double[num_classes];
        int sum = 0;
        for (int k = 0; k < num_classes; ++k)
          sum += f_matrix[j][k];
        for (int k = 0; k < num_classes; ++k)
          matrix[j][k] = static_cast<double>(f_matrix[j][k]) / sum;
      }

      double *quality = new double[num_classes];
      double *avg_prob = new double[num_classes];
      for (int j = 0; j < num_classes; ++j) {
        quality[j] = matrix[0][j];
        avg_prob[j] = matrix[0][j];
        for (int k = 1; k < num_classes; ++k) {
          if (matrix[k][j] < quality[j]) {
            quality[j] = matrix[k][j];
          }
          avg_prob[j] += matrix[k][j];
        }
        avg_prob[j] /= num_classes;
      }

      int best = 0;
      for (int j = 1; j < num_classes; ++j) {
        if (quality[j] > quality[best]) {
          best = j;
        }
      }

      lower_bounds[i] = quality[best];
      upper_bounds[i] = matrix[0][best];
      for (int j = 1; j < num_classes; ++j) {
        if (matrix[j][best] > upper_bounds[i]) {
          upper_bounds[i] = matrix[j][best];
        }
      }

      brier[i] = 0;
      for (int j = 0; j < num_classes; ++j) {
        if (labels[static_cast<std::size_t>(j)] == prob->y[indices[i]]) {
          brier[i] += (1-avg_prob[j])*(1-avg_prob[j]);
          double tmp = std::fmax(std::fmin(avg_prob[j], 1-kEpsilon), kEpsilon);
          logloss[i] = - std::log(tmp);
        } else {
          brier[i] += avg_prob[j]*avg_prob[j];
        }
      }

      if (param->probability == 1) {
        for (int j = 0; j < num_classes; ++j) {
          std::cout << avg_prob[j] << ' ';
        }
        std::cout << '\n';
      }

      delete[] avg_prob;

      predict_labels[i] = labels[static_cast<std::size_t>(best)];

      delete[] quality;
      for (int j = 0; j < num_classes; ++j) {
        delete[] f_matrix[j];
        delete[] matrix[j];
      }
      delete[] f_matrix;
      delete[] matrix;

      this_label = static_cast<int>(prob->y[indices[i]]);
      std::size_t j;
      for (j = 0; j < num_classes; ++j) {
        if (this_label == labels[j]) break;
      }
      alter_labels[i] = static_cast<int>(j);
      if (j == num_classes) {
        labels.push_back(this_label);
        ++num_classes;
      }

      for (int j = 0; j < i; ++j) {
        double dist = CalcDist(prob->x[indices[j]], prob->x[indices[i]]);
        int index = CompareDist(dist_neighbors[j], dist, num_neighbors);
        if (index < num_neighbors) {
          InsertLabel(label_neighbors[j], alter_labels[i], num_neighbors, index);
        }
      }

    }
    if (param->probability == 1) {
      for (std::size_t j = 0; j < num_classes; ++j) {
        std::cout << labels[j] << "        ";
      }
      std::cout << '\n';
    }

    for (int i = 0; i < num_ex; ++i) {
      delete[] dist_neighbors[i];
      delete[] label_neighbors[i];
    }

    delete[] dist_neighbors;
    delete[] label_neighbors;
    delete[] categories;
    delete[] alter_labels;
    std::vector<int>(labels).swap(labels);
  }

  if (param->taxonomy_type == SVM_EL ||
      param->taxonomy_type == SVM_ES ||
      param->taxonomy_type == SVM_KM) {
    Problem subprob;
    subprob.x = new Node*[num_ex];
    subprob.y = new double[num_ex];

    for (int i = 0; i < num_ex; ++i) {
      subprob.x[i] = prob->x[indices[i]];
      subprob.y[i] = prob->y[indices[i]];
    }

    for (int i = 1; i < num_ex; ++i) {
      double *avg_prob = NULL;
      brier[i] = 0;
      subprob.num_ex = i;
      Model *submodel = TrainCP(&subprob, param);
      predict_labels[i] = PredictCP(&subprob, submodel, subprob.x[i],
                                    lower_bounds[i], upper_bounds[i], &avg_prob);
      for (int j = 0; j < submodel->num_classes; ++j) {
        if (submodel->labels[j] == subprob.y[i]) {
          brier[i] += (1-avg_prob[j]) * (1-avg_prob[j]);
          double tmp = std::fmax(std::fmin(avg_prob[j], 1-kEpsilon), kEpsilon);
          logloss[i] = - std::log(tmp);
        } else {
          brier[i] += avg_prob[j] * avg_prob[j];
        }
      }
      if (param->probability == 1) {
        for (int j = 0; j < submodel->num_classes; ++j) {
          std::cout << avg_prob[j] << ' ';
        }
        std::cout << '\n';
      }
      FreeModel(submodel);
      delete[] avg_prob;
    }
    delete[] subprob.x;
    delete[] subprob.y;
  }

  return;
}

static const char *kTaxonomyTypeTable[] = { "knn", "svm_el", "svm_es", "svm_km", NULL };

int SaveModel(const char *model_file_name, const struct Model *model) {
  std::ofstream model_file(model_file_name);
  if (!model_file.is_open()) {
    std::cerr << "Unable to open model file: " << model_file_name << std::endl;
    return -1;
  }

  const Parameter &param = model->param;

  model_file << "taxonomy_type " << kTaxonomyTypeTable[param.taxonomy_type] << '\n';
  model_file << "num_categories " << model->num_categories << '\n';

  if (param.taxonomy_type == KNN) {
    SaveKNNModel(model_file, model->knn_model);
  }
  if (param.taxonomy_type == SVM_EL ||
      param.taxonomy_type == SVM_ES ||
      param.taxonomy_type == SVM_KM) {
    SaveSVMModel(model_file, model->svm_model);
  }

  if ((param.taxonomy_type == SVM_ES ||
       param.taxonomy_type == SVM_KM) &&
      model->points) {
    model_file << "points\n";
    for (int i = 0; i < model->num_categories; ++i) {
      model_file << model->points[i] << ' ';
    }
    model_file << '\n';
  }

  if (model->categories) {
    model_file << "categories\n";
    for (int i = 0; i < model->num_ex; ++i) {
      model_file << model->categories[i] << ' ';
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
  model->categories = NULL;

  char cmd[80];
  while (1) {
    model_file >> cmd;

    if (std::strcmp(cmd, "taxonomy_type") == 0) {
      model_file >> cmd;
      int i;
      for (i = 0; kTaxonomyTypeTable[i]; ++i) {
        if (std::strcmp(kTaxonomyTypeTable[i], cmd) == 0) {
          param.taxonomy_type = i;
          break;
        }
      }
      if (kTaxonomyTypeTable[i] == NULL) {
        std::cerr << "Unknown taxonomy type.\n" << std::endl;
        FreeModel(model);
        delete model;
        model_file.close();
        return NULL;
      }
    } else
    if (std::strcmp(cmd, "num_categories") == 0) {
      model_file >> param.num_categories;
      model->num_categories = param.num_categories;
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
    if (std::strcmp(cmd, "svm_model") == 0) {
      model->svm_model = LoadSVMModel(model_file);
      if (model->svm_model == NULL) {
        FreeModel(model);
        delete model;
        model_file.close();
        return NULL;
      }
      model->num_ex = model->svm_model->num_ex;
      model->num_classes = model->svm_model->num_classes;
      clone(model->labels, model->svm_model->labels, model->num_classes);
      model->param.svm_param = &model->svm_model->param;
    } else
    if (std::strcmp(cmd, "points") == 0) {
      int num_categories = model->num_categories;
      model->points = new double[num_categories];
      for (int i = 0; i < num_categories; ++i) {
        model_file >> model->points[i];
      }
    } else
    if (std::strcmp(cmd, "categories") == 0) {
      int num_ex = model->num_ex;
      model->categories = new int[num_ex];
      for (int i = 0; i < num_ex; ++i) {
        model_file >> model->categories[i];
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
  if (model->param.taxonomy_type == KNN &&
      model->knn_model != NULL) {
    FreeKNNModel(model->knn_model);
    delete model->knn_model;
    model->knn_model = NULL;
  }

  if ((model->param.taxonomy_type == SVM_EL ||
       model->param.taxonomy_type == SVM_ES ||
       model->param.taxonomy_type == SVM_KM) &&
      model->svm_model != NULL) {
    FreeSVMModel(&(model->svm_model));
    delete model->svm_model;
    model->svm_model = NULL;
  }

  if (model->labels != NULL) {
    delete[] model->labels;
    model->labels = NULL;
  }

  if (model->param.taxonomy_type == SVM_ES &&
      model->points != NULL) {
    delete[] model->points;
    model->points = NULL;
  }
  if (model->categories != NULL) {
    delete[] model->categories;
    model->categories = NULL;
  }

  delete model;
  model = NULL;

  return;
}

void FreeParam(struct Parameter *param) {
  if (param->taxonomy_type == KNN &&
      param->knn_param != NULL) {
    FreeKNNParam(param->knn_param);
    param->knn_param = NULL;
  }

  if ((param->taxonomy_type == SVM_EL ||
       param->taxonomy_type == SVM_ES ||
       param->taxonomy_type == SVM_KM) &&
      param->svm_param != NULL) {
    FreeSVMParam(param->svm_param);
    param->svm_param = NULL;
  }

  return;
}

const char *CheckParameter(const struct Parameter *param) {
  if (param->save_model == 1 && param->load_model == 1) {
    return "cannot save and load model at the same time";
  }

  if (param->num_categories == 0) {
    return "no. of categories cannot be less than 1";
  }

  if (param->taxonomy_type == KNN) {
    if (param->knn_param == NULL) {
      return "no knn parameter";
    }
    return CheckKNNParameter(param->knn_param);
  }

  if (param->taxonomy_type == SVM_EL ||
      param->taxonomy_type == SVM_ES ||
      param->taxonomy_type == SVM_KM) {
    if (param->svm_param == NULL) {
      return "no svm parameter";
    }
    return CheckSVMParameter(param->svm_param);
  }

  if (param->taxonomy_type > 3) {
    return "no such taxonomy type";
  }

  return NULL;
}