#include "cp.h"
#include <iostream>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <vector>

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
  int num_correct = 0, num_empty = 0, num_multi = 0, num_incl = 0;
  double avg_conf = 0, avg_cred = 0;
  const char *error_message;

  ParseCommandLine(argc, argv, train_file_name, test_file_name, output_file_name, model_file_name);
  error_message = CheckParameter(&param);

  if (error_message != NULL) {
    std::cerr << error_message << std::endl;
    exit(EXIT_FAILURE);
  }

  train = ReadProblem(train_file_name);
  test = ReadProblem(test_file_name);

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

  for (int i = 0; i < test->num_ex; ++i) {
    double conf, cred;
    std::vector<int> predict_label;

    predict_label = PredictCP(train, model, test->x[i], conf, cred);

    avg_conf += conf;
    avg_cred += cred;

    output_file << std::resetiosflags(std::ios::fixed) << test->y[i] << ' ' << predict_label[0] << ' '
                << std::setiosflags(std::ios::fixed) << conf << ' ' << cred;
    if (predict_label[0] == test->y[i]) {
      ++num_correct;
    }

    if (predict_label.size() == 1) {
      ++num_empty;
      output_file << " Empty\n";
    } else {
      output_file << " set:";
      for (size_t j = 1; j < predict_label.size(); ++j) {
        output_file << ' ' << predict_label[j];
        if (predict_label[j] == test->y[i]) {
          ++num_incl;
        }
      }
      if (predict_label.size() > 2) {
        ++num_multi;
        output_file << " Multi\n";
      } else {
        output_file << " Single\n";
      }
    }
    std::vector<int>().swap(predict_label);
  }
  avg_conf /= test->num_ex;
  avg_cred /= test->num_ex;

  std::chrono::time_point<std::chrono::steady_clock> end_time = std::chrono::high_resolution_clock::now();

  std::cout << "Simple Accuracy: " << 100.0*num_correct/test->num_ex << '%'
            << " (" << num_correct << '/' << test->num_ex << ") "
            << "Mean Confidence: " << std::fixed << std::setprecision(4) << 100*avg_conf << "%, "
            << "Mean Credibility: " << 100*avg_cred << "% " << '\n';
  std::cout << "Accuracy: " << 100.0*num_incl/test->num_ex << '%'
            << " (" << num_incl << '/' << test->num_ex << ") "
            << "Multi Prediction: " << std::fixed << std::setprecision(4) << 100.0*num_multi/test->num_ex << "%, "
            << "Empty Prediction: " << 100.0*num_empty/test->num_ex << "% " << '\n';
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
            << "  -k num_neighbors : set number of neighbors in kNN (default 1)\n"
            << "  -s model_file_name : save model\n"
            << "  -l model_file_name : load model\n"
            << "  -e epsilon : set significance level (default 0.05)\n";
  exit(EXIT_FAILURE);
}

void ParseCommandLine(int argc, char **argv, char *train_file_name, char *test_file_name, char *output_file_name, char *model_file_name) {
  int i;
  param.measure_type = KNN;
  param.save_model = 0;
  param.load_model = 0;
  param.epsilon = 0.05;
  param.knn_param = new KNNParameter;
  InitKNNParam(param.knn_param);

  for (i = 1; i < argc; ++i) {
    if (argv[i][0] != '-') break;
    if ((i+2) >= argc)
      ExitWithHelp();
    switch (argv[i][1]) {
      case 't': {
        ++i;
        param.measure_type = std::atoi(argv[i]);
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
      case 'e': {
        ++i;
        param.epsilon = std::atof(argv[i]);
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