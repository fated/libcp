#include "knn.h"
#include <cmath>
#include <fstream>

double CalcDist(const struct Node *x1, const struct Node *x2) {
  double sum = 0;

  while (x1->index != -1 && x2->index != -1) {
    if (x1->index == x2->index) {
      sum += (x1->value - x2->value) * (x1->value - x2->value);
      ++x1;
      ++x2;
    } else {
      if(x1->index > x2->index) {
        sum += x2->value * x2->value;
        ++x2;
      } else {
        sum += x1->value * x1->value;
        ++x1;
      }
    }
  }
  if (x1->index == -1) {
    while (x2->index != -1) {
      sum += x2->value * x2->value;
      ++x2;
    }
  }
  if (x2->index == -1) {
    while (x1->index != -1) {
      sum += x1->value * x1->value;
      ++x1;
    }
  }

  return sqrt(sum);
}

int CompareDist(double *neighbors, double dist, int num_neighbors) {
  int i = 0;

  while (i < num_neighbors) {
    if (dist < neighbors[i])
      break;
    ++i;
  }
  if (i == num_neighbors)
    return i;
  for (int j = num_neighbors-1; j > i; --j)
    neighbors[j] = neighbors[j-1];
  neighbors[i] = dist;

  return i;
}

KNNModel *TrainKNN(const struct Problem *prob, const struct KNNParameter *param) {
  KNNModel *model = new KNNModel;
  model->param = *param;
  model->labels = NULL;
  model->dist_neighbors = NULL;
  model->label_neighbors = NULL;
  model->min_same = NULL;
  model->min_diff = NULL;

  int num_classes = 0;
  int num_ex = prob->num_ex;
  int num_neighbors = param->num_neighbors;
  int *labels = NULL;
  int *alter_labels = new int[num_ex];

  std::vector<int> unique_labels;
  for (int i = 0; i < num_ex; ++i) {
    int this_label = static_cast<int>(prob->y[i]);
    std::size_t j;
    for (j = 0; j < num_classes; ++j) {
      if (this_label == unique_labels[j]) break;
    }
    alter_labels[i] = static_cast<int>(j);
    if (j == num_classes) {
      unique_labels.push_back(this_label);
      ++num_classes;
    }
  }
  labels = new int[num_classes];
  for (std::size_t i = 0; i < unique_labels.size(); ++i) {
    labels[i] = unique_labels[i];
  }
  std::vector<int>(unique_labels).swap(unique_labels);

  if (num_classes == 1) {
    std::cerr << "WARNING: training set only has one class. See README for details." << std::endl;
  }

  if (param->is_cp == 0) {
    double **dist_neighbors = new double*[num_ex];
    int **label_neighbors = new int*[num_ex];

    for (int i = 0; i < num_ex; ++i) {
      dist_neighbors[i] = new double[num_neighbors];
      label_neighbors[i] = new int[num_neighbors];
      for (int j = 0; j < num_neighbors; ++j) {
        dist_neighbors[i][j] = kInf;
        label_neighbors[i][j] = -1;
      }
    }

    for (int i = 0; i < num_ex-1; ++i) {
      for (int j = i+1; j < num_ex; ++j) {
        double dist = CalcDist(prob->x[i], prob->x[j]);
        int index;
        index = CompareDist(dist_neighbors[i], dist, num_neighbors);
        if (index < num_neighbors) {
          InsertLabel(label_neighbors[i], alter_labels[j], num_neighbors, index);
        }
        index = CompareDist(dist_neighbors[j], dist, num_neighbors);
        if (index < num_neighbors) {
          InsertLabel(label_neighbors[j], alter_labels[i], num_neighbors, index);
        }
      }
    }

    model->dist_neighbors = dist_neighbors;
    model->label_neighbors = label_neighbors;

  } else {

    double **min_same = new double*[num_ex];
    double **min_diff = new double*[num_ex];

    for (int i = 0; i < num_ex; ++i) {
      min_same[i] = new double[num_neighbors];
      min_diff[i] = new double[num_neighbors];
      for (int j = 0; j < num_neighbors; ++j) {
        min_same[i][j] = kInf;
        min_diff[i][j] = kInf;
      }
    }

    for (int i = 0; i < num_ex-1; ++i) {
      for (int j = i+1; j < num_ex; ++j) {
        double dist = CalcDist(prob->x[i], prob->x[j]);

        if (prob->y[i] == prob->y[j]) {
          CompareDist(min_same[i], dist, num_neighbors);
          CompareDist(min_same[j], dist, num_neighbors);
        } else {
          CompareDist(min_diff[i], dist, num_neighbors);
          CompareDist(min_diff[j], dist, num_neighbors);
        }
      }
    }

    model->min_same = min_same;
    model->min_diff = min_diff;
  }

  delete[] alter_labels;

  model->num_ex = num_ex;
  model->num_classes = num_classes;
  model->labels = labels;
  model->param = *param;

  return model;
}

double PredictKNN(struct Problem *train, struct Node *x, const int num_neighbors) {
  double neighbors[num_neighbors];
  double labels[num_neighbors];

  for (int i = 0; i < num_neighbors; ++i) {
    neighbors[i] = kInf;
    labels[i] = -1;
  }
  for (int i = 0; i < train->num_ex; ++i) {
    double dist = CalcDist(train->x[i], x);
    int index = CompareDist(neighbors, dist, num_neighbors);
    if (index < num_neighbors) {
      InsertLabel(labels, train->y[i], num_neighbors, index);
    }
  }
  double predict_label = FindMostFrequent(labels, num_neighbors);

  return predict_label;
}

int SaveKNNModel(std::ofstream &model_file, const struct KNNModel *model) {
  model_file << "knn_model\n";
  model_file << "num_examples " << model->num_ex << '\n';
  model_file << "num_classes " << model->num_classes << '\n';
  model_file << "num_neighbors " << model->param.num_neighbors << '\n';

  if (model->labels) {
    model_file << "labels";
    for (int i = 0; i < model->num_classes; ++i) {
      model_file << ' ' << model->labels[i];
    }
    model_file << '\n';
  }

  if (model->dist_neighbors) {
    model_file << "dist_neighbors\n";
    for (int i = 0; i < model->num_ex; ++i) {
      for (int j = 0; j < model->param.num_neighbors; ++j) {
        model_file << model->dist_neighbors[i][j] << ' ';
      }
    }
    model_file << '\n';
  }

  if (model->label_neighbors) {
    model_file << "label_neighbors\n";
    for (int i = 0; i < model->num_ex; ++i) {
      for (int j = 0; j < model->param.num_neighbors; ++j) {
        model_file << model->label_neighbors[i][j] << ' ';
      }
    }
    model_file << '\n';
  }

  if (model->min_same) {
    model_file << "min_same\n";
    for (int i = 0; i < model->num_ex; ++i) {
      for (int j = 0; j < model->param.num_neighbors; ++j) {
        model_file << model->min_same[i][j] << ' ';
      }
    }
    model_file << '\n';
  }

  if (model->min_diff) {
    model_file << "min_diff\n";
    for (int i = 0; i < model->num_ex; ++i) {
      for (int j = 0; j < model->param.num_neighbors; ++j) {
        model_file << model->min_diff[i][j] << ' ';
      }
    }
    model_file << '\n';
  }

  return 0;
}

KNNModel *LoadKNNModel(std::ifstream &model_file) {
  KNNModel *model = new KNNModel;
  KNNParameter &param = model->param;

  char cmd[80];
  while (1) {
    model_file >> cmd;

    if (std::strcmp(cmd, "num_examples") == 0) {
      model_file >> model->num_ex;
    } else
    if (std::strcmp(cmd, "num_classes") == 0) {
      model_file >> model->num_classes;
    } else
    if (std::strcmp(cmd, "num_neighbors") == 0) {
      model_file >> param.num_neighbors;
    } else
    if (std::strcmp(cmd, "labels") == 0) {
      int n = model->num_classes;
      model->labels = new int[n];
      for (int i = 0; i < n; ++i) {
        model_file >> model->labels[i];
      }
    } else
    if (std::strcmp(cmd, "dist_neighbors") == 0) {
      int n = param.num_neighbors;
      int num_ex = model->num_ex;
      model->dist_neighbors = new double*[num_ex];
      for (int i = 0; i < num_ex; ++i) {
        model->dist_neighbors[i] = new double[n];
        for (int j = 0; j < n; ++j) {
          model_file >> model->dist_neighbors[i][j];
        }
      }
    } else
    if (std::strcmp(cmd, "label_neighbors") == 0) {
      int n = param.num_neighbors;
      int num_ex = model->num_ex;
      model->label_neighbors = new int*[num_ex];
      for (int i = 0; i < num_ex; ++i) {
        model->label_neighbors[i] = new int[n];
        for (int j = 0; j < n; ++j) {
          model_file >> model->label_neighbors[i][j];
        }
      }
    } else
    if (std::strcmp(cmd, "min_same") == 0) {
      int n = param.num_neighbors;
      int num_ex = model->num_ex;
      model->min_same = new double*[num_ex];
      for (int i = 0; i < num_ex; ++i) {
        model->min_same[i] = new double[n];
        for (int j = 0; j < n; ++j) {
          model_file >> model->min_same[i][j];
        }
      }
    } else
    if (std::strcmp(cmd, "min_diff") == 0) {
      int n = param.num_neighbors;
      int num_ex = model->num_ex;
      model->min_diff = new double*[num_ex];
      for (int i = 0; i < num_ex; ++i) {
        model->min_diff[i] = new double[n];
        for (int j = 0; j < n; ++j) {
          model_file >> model->min_diff[i][j];
        }
      }
      break;
    } else {
      std::cerr << "Unknown text in knn_model file: " << cmd << std::endl;
      FreeKNNModel(model);
      return NULL;
    }
  }

  return model;
}

void FreeKNNModel(struct KNNModel *model) {
  if (model->labels != NULL) {
    delete[] model->labels;
    model->labels = NULL;
  }

  if (model->dist_neighbors != NULL) {
    for (int i = 0; i < model->num_ex; ++i) {
      delete[] model->dist_neighbors[i];
    }
    delete[] model->dist_neighbors;
    model->dist_neighbors = NULL;
  }

  if (model->label_neighbors != NULL) {
    for (int i = 0; i < model->num_ex; ++i) {
      delete[] model->label_neighbors[i];
    }
    delete[] model->label_neighbors;
    model->label_neighbors = NULL;
  }

  if (model->min_same != NULL) {
    for (int i = 0; i < model->num_ex; ++i) {
      delete[] model->min_same[i];
    }
    delete[] model->min_same;
    model->min_same = NULL;
  }

  if (model->min_diff != NULL) {
    for (int i = 0; i < model->num_ex; ++i) {
      delete[] model->min_diff[i];
    }
    delete[] model->min_diff;
    model->min_diff = NULL;
  }

  if (model != NULL) {
    delete model;
    model = NULL;
  }

  return;
}

void FreeKNNParam(struct KNNParameter *param) {
  return;
}

void InitKNNParam(struct KNNParameter *param) {
  param->num_neighbors = 1;
  param->is_cp = 1;

  return;
}

const char *CheckKNNParameter(const struct KNNParameter *param) {
  if (param->num_neighbors < 1) {
    return "num_neighbors should be greater than 0";
  }

  return NULL;
}