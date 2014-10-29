#include "cp.h"
#include <iostream>
#include <fstream>
#include <iomanip>
#include <cmath>

void ExitWithHelp();
void ParseCommandLine(int argc, char *argv[], char *train_file_name, char *test_file_name, char *output_file_name, char *model_file_name);

struct Parameter param;

int main(int argc, char *argv[]) {
  char train_file_name[256];
  char test_file_name[256];
  char output_file_name[256];
  char model_file_name[256];
  struct Problem *train, *test;
  struct Model *model;
  int num_correct = 0;
  double avg_lower_bound = 0, avg_upper_bound = 0, avg_brier = 0, avg_logloss = 0;
  const char *error_message;

  ParseCommandLine(argc, argv, train_file_name, test_file_name, output_file_name, model_file_name);
  error_message = CheckParameter(&param);

  if (error_message != NULL) {
    std::cerr << error_message << std::endl;
    exit(EXIT_FAILURE);
  }

  train = ReadProblem(train_file_name);
  test = ReadProblem(test_file_name);

  if ((param.measure_type == SVM_EL ||
       param.measure_type == SVM_ES ||
       param.measure_type == SVM_KM) &&
      param.svm_param->gamma == 0) {
    param.svm_param->gamma = 1.0 / train->max_index;
  }

  std::ofstream output_file(output_file_name);
  if (!output_file.is_open()) {
    std::cerr << "Unable to open output file: " << output_file_name << std::endl;
    exit(EXIT_FAILURE);
  }

  std::chrono::time_point<std::chrono::steady_clock> start_time = std::chrono::high_resolution_clock::now();

  if (param.load_model == 1) {
    model = LoadModel(model_file_name);
    if (model == NULL) {
      exit(EXIT_FAILURE);
    }
  } else {
    model = TrainCP(train, &param);
  }

  if (param.save_model == 1) {
    if (SaveModel(model_file_name, model) != 0) {
      std::cerr << "Unable to save model file" << std::endl;
    }
  }

  if (param.probability == 1) {
    output_file << "                      ";
    for (int i = 0; i < model->num_classes; ++i) {
      output_file << model->labels[i] << "        ";
    }
    output_file << '\n';
  }

  for (int i = 0; i < test->num_ex; ++i) {
    double predict_label, lower_bound, upper_bound, logloss, brier = 0, *avg_prob = NULL;

    predict_label = PredictCP(train, model, test->x[i], lower_bound, upper_bound, &avg_prob);

    for (int j = 0; j < model->num_classes; ++j) {
      if (model->labels[j] == test->y[i]) {
        brier += (1-avg_prob[j])*(1-avg_prob[j]);
        double tmp = std::fmax(std::fmin(avg_prob[j], 1-kEpsilon), kEpsilon);
        logloss = - std::log(tmp);
      } else {
        brier += avg_prob[j]*avg_prob[j];
      }
    }
    avg_lower_bound += lower_bound;
    avg_upper_bound += upper_bound;
    avg_brier += brier;
    avg_logloss += logloss;

    output_file << std::resetiosflags(std::ios::fixed) << test->y[i] << ' ' << predict_label << ' '
                << std::setiosflags(std::ios::fixed) << lower_bound << ' ' << upper_bound;
    if (param.probability == 1) {
      for (int j = 0; j < model->num_classes; ++j) {
        output_file << ' ' << avg_prob[j];
      }
    }
    output_file << '\n';
    if (predict_label == test->y[i]) {
      ++num_correct;
    }
    delete[] avg_prob;
  }
  avg_lower_bound /= test->num_ex;
  avg_upper_bound /= test->num_ex;
  avg_brier /= test->num_ex;
  avg_logloss /= test->num_ex;

  std::chrono::time_point<std::chrono::steady_clock> end_time = std::chrono::high_resolution_clock::now();

  std::cout << "Accuracy: " << 100.0*num_correct/test->num_ex << '%'
            << " (" << num_correct << '/' << test->num_ex << ") "
            << "Probabilities: [" << std::fixed << std::setprecision(4) << 100*avg_lower_bound << "%, "
            << 100*avg_upper_bound << "%] "
            << "Brier Score: " << avg_brier << ' '
            << "Logarithmic Loss: " << avg_logloss << '\n';
  output_file.close();

  std::cout << "Time cost: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count()/1000.0 << " s\n";

  FreeProblem(train);
  FreeProblem(test);
  FreeModel(model);
  FreeParam(&param);

  return 0;
}

void ExitWithHelp() {
  std::cout << "Usage: cp-offline [options] train_file test_file [output_file]\n"
            << "options:\n"
            << "  -t non-conformity measure : set type of NCM (default 0)\n"
            << "    0 -- k-nearest neighbors (KNN)\n"
            << "    1 -- support vector machine with equal length (SVM_EL)\n"
            << "    2 -- support vector machine with equal size (SVM_ES)\n"
            << "    3 -- support vector machine with k-means clustering (SVM_KM)\n"
            << "  -k num_neighbors : set number of neighbors in kNN (default 1)\n"
            << "  -s model_file_name : save model\n"
            << "  -l model_file_name : load model\n"
            << "  -b probability estimates : whether to output probability estimates for all labels, 0 or 1 (default 0)\n"
            << "  -p : prefix of options to set parameters for SVM\n"
            << "    -ps svm_type : set type of SVM (default 0)\n"
            << "      0 -- C-SVC    (multi-class classification)\n"
            << "      1 -- nu-SVC   (multi-class classification)\n"
            << "    -pt kernel_type : set type of kernel function (default 2)\n"
            << "      0 -- linear: u'*v\n"
            << "      1 -- polynomial: (gamma*u'*v + coef0)^degree\n"
            << "      2 -- radial basis function: exp(-gamma*|u-v|^2)\n"
            << "      3 -- sigmoid: tanh(gamma*u'*v + coef0)\n"
            << "      4 -- precomputed kernel (kernel values in training_set_file)\n"
            << "    -pd degree : set degree in kernel function (default 3)\n"
            << "    -pg gamma : set gamma in kernel function (default 1/num_features)\n"
            << "    -pr coef0 : set coef0 in kernel function (default 0)\n"
            << "    -pc cost : set the parameter C of C-SVC (default 1)\n"
            << "    -pn nu : set the parameter nu of nu-SVC (default 0.5)\n"
            << "    -pm cachesize : set cache memory size in MB (default 100)\n"
            << "    -pe epsilon : set tolerance of termination criterion (default 0.001)\n"
            << "    -ph shrinking : whether to use the shrinking heuristics, 0 or 1 (default 1)\n"
            << "    -pwi weights : set the parameter C of class i to weight*C, for C-SVC (default 1)\n"
            << "    -pq : quiet mode (no outputs)\n";
  exit(EXIT_FAILURE);
}

void ParseCommandLine(int argc, char **argv, char *train_file_name, char *test_file_name, char *output_file_name, char *model_file_name) {
  int i;
  param.measure_type = KNN;
  param.save_model = 0;
  param.load_model = 0;
  param.probability = 0;
  param.knn_param = new KNNParameter;
  param.svm_param = NULL;
  InitKNNParam(param.knn_param);

  for (i = 1; i < argc; ++i) {
    if (argv[i][0] != '-') break;
    if ((i+2) >= argc)
      ExitWithHelp();
    switch (argv[i][1]) {
      case 't': {
        ++i;
        param.measure_type = std::atoi(argv[i]);
        if (param.measure_type == SVM_EL ||
            param.measure_type == SVM_ES ||
            param.measure_type == SVM_KM) {
          FreeKNNParam(param.knn_param);
          delete param.knn_param;
          param.svm_param = new SVMParameter;
          InitSVMParam(param.svm_param);
        }
        break;
      }
      case 'k': {
        ++i;
        param.knn_param->num_neighbors = std::atoi(argv[i]);
        break;
      }
      case 's': {
        ++i;
        param.save_model = 1;
        std::strcpy(model_file_name, argv[i]);
        break;
      }
      case 'l': {
        ++i;
        param.load_model = 1;
        std::strcpy(model_file_name, argv[i]);
        break;
      }
      case 'b': {
        ++i;
        param.probability = std::atoi(argv[i]);
        break;
      }
      case 'p': {
        if (argv[i][2]) {
          switch (argv[i][2]) {
            case 's': {
              ++i;
              param.svm_param->svm_type = std::atoi(argv[i]);
              break;
            }
            case 't': {
              ++i;
              param.svm_param->kernel_type = std::atoi(argv[i]);
              break;
            }
            case 'd': {
              ++i;
              param.svm_param->degree = std::atoi(argv[i]);
              break;
            }
            case 'g': {
              ++i;
              param.svm_param->gamma = std::atof(argv[i]);
              break;
            }
            case 'r': {
              ++i;
              param.svm_param->coef0 = std::atof(argv[i]);
              break;
            }
            case 'n': {
              ++i;
              param.svm_param->nu = std::atof(argv[i]);
              break;
            }
            case 'm': {
              ++i;
              param.svm_param->cache_size = std::atof(argv[i]);
              break;
            }
            case 'c': {
              ++i;
              param.svm_param->C = std::atof(argv[i]);
              break;
            }
            case 'e': {
              ++i;
              param.svm_param->eps = std::atof(argv[i]);
              break;
            }
            case 'h': {
              ++i;
              param.svm_param->shrinking = std::atoi(argv[i]);
              break;
            }
            case 'q': {
              SetPrintNull();
              break;
            }
            case 'w': {  // weights [option]: '-w1' means weight of '1'
              ++i;
              ++param.svm_param->num_weights;
              param.svm_param->weight_labels = (int *)realloc(param.svm_param->weight_labels, sizeof(int)*static_cast<unsigned long int>(param.svm_param->num_weights));
              param.svm_param->weights = (double *)realloc(param.svm_param->weights, sizeof(double)*static_cast<unsigned long int>(param.svm_param->num_weights));
              param.svm_param->weight_labels[param.svm_param->num_weights-1] = std::atoi(&argv[i-1][3]); // get and convert 'i' to int
              param.svm_param->weights[param.svm_param->num_weights-1] = std::atof(argv[i]);
              break;
              // TODO: change realloc function
            }
            default: {
              std::cerr << "Unknown SVM option: " << argv[i] << std::endl;
              ExitWithHelp();
            }
          }
        }
        break;
      }
      default: {
        std::cerr << "Unknown option: -" << argv[i][1] << std::endl;
        ExitWithHelp();
      }
    }
  }

  if ((i+1) >= argc)
    ExitWithHelp();
  std::strcpy(train_file_name, argv[i]);
  std::strcpy(test_file_name, argv[i+1]);
  if ((i+2) < argc) {
    std::strcpy(output_file_name, argv[i+2]);
  } else {
    char *p = std::strrchr(argv[i+1],'/');
    if (p == NULL) {
      p = argv[i+1];
    } else {
      ++p;
    }
    std::sprintf(output_file_name, "%s_output", p);
  }

  return;
}