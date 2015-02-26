#include "cp.h"
#include <iostream>
#include <fstream>
#include <iomanip>

void ExitWithHelp();
void ParseCommandLine(int argc, char *argv[], char *data_file_name, char *output_file_name);

struct Parameter param;

int main(int argc, char *argv[]) {
  char data_file_name[256];
  char output_file_name[256];
  struct Problem *prob;
  int num_correct = 0, num_empty = 0, num_multi = 0, num_incl = 0;
  int *indices = NULL;
  double avg_conf = 0, avg_cred = 0;
  double *conf = NULL, *cred = NULL;
  std::vector<int> *predict_labels = NULL;
  const char *error_message;

  ParseCommandLine(argc, argv, data_file_name, output_file_name);
  error_message = CheckParameter(&param);

  if (error_message != NULL) {
    std::cerr << error_message << std::endl;
    exit(EXIT_FAILURE);
  }

  prob = ReadProblem(data_file_name);

  std::ofstream output_file(output_file_name);
  if (!output_file.is_open()) {
    std::cerr << "Unable to open output file: " << output_file_name << std::endl;
    exit(EXIT_FAILURE);
  }

  predict_labels = new std::vector<int>[prob->num_ex];
  conf = new double[prob->num_ex];
  cred = new double[prob->num_ex];
  indices = new int[prob->num_ex];

  std::chrono::time_point<std::chrono::steady_clock> start_time = std::chrono::high_resolution_clock::now();

  OnlinePredict(prob, &param, predict_labels, indices, conf, cred);

  std::chrono::time_point<std::chrono::steady_clock> end_time = std::chrono::high_resolution_clock::now();

  output_file << prob->y[indices[0]] << '\n';

  for (int i = 1; i < prob->num_ex; ++i) {
    avg_conf += conf[i];
    avg_cred += cred[i];

    output_file << std::resetiosflags(std::ios::fixed) << prob->y[indices[i]] << ' ' << predict_labels[i][0] << ' '
                << std::setiosflags(std::ios::fixed) << conf[i] << ' ' << cred[i];
    if (predict_labels[i][0] == prob->y[indices[i]]) {
      ++num_correct;
    }

    if (predict_labels[i].size() == 1) {
      ++num_empty;
      output_file << " Empty\n";
    } else {
      output_file << " set:";
      for (size_t j = 1; j < predict_labels[i].size(); ++j) {
        output_file << ' ' << predict_labels[i][j];
        if (predict_labels[i][j] == prob->y[indices[i]]) {
          ++num_incl;
        }
      }
      if (predict_labels[i].size() > 2) {
        ++num_multi;
        output_file << " Multi\n";
      } else {
        output_file << " Single\n";
      }
    }
    std::vector<int>().swap(predict_labels[i]);
  }
  avg_conf /= prob->num_ex - 1;
  avg_cred /= prob->num_ex - 1;

  std::cout << "Online Accuracy: " << 100.0*num_correct/(prob->num_ex-1) << '%'
            << " (" << num_correct << '/' << prob->num_ex-1 << ") "
            << "Mean Confidence: " << std::fixed << std::setprecision(4) << 100*avg_conf << "%, "
            << "Mean Credibility: " << 100*avg_cred << "%\n";
  std::cout << "Accuracy: " << 100.0*num_incl/(prob->num_ex-1) << '%'
            << " (" << num_incl << '/' << prob->num_ex-1 << ") "
            << "Multi Prediction: " << std::fixed << std::setprecision(4) << 100.0*num_multi/(prob->num_ex-1) << "%, "
            << "Empty Prediction: " << 100.0*num_empty/(prob->num_ex-1) << "%\n";
  output_file.close();

  std::cout << "Time cost: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count()/1000.0 << " s\n";

  FreeProblem(prob);
  FreeParam(&param);
  delete[] predict_labels;
  delete[] conf;
  delete[] cred;
  delete[] indices;

  return 0;
}

void ExitWithHelp() {
  std::cout << "Usage: vm-online [options] data_file [output_file]\n"
            << "options:\n"
            << "  -t non-conformity measure : set type of NCM (default 0)\n"
            << "    0 -- k-nearest neighbors (KNN)\n"
            << "  -k num_neighbors : set number of neighbors in kNN (default 1)\n"
            << "  -e epsilon : set significance level (default 0.05)\n";
  exit(EXIT_FAILURE);
}

void ParseCommandLine(int argc, char **argv, char *data_file_name, char *output_file_name) {
  int i;
  param.measure_type = KNN;
  param.save_model = 0;
  param.load_model = 0;
  param.epsilon = 0.05;
  param.knn_param = new KNNParameter;
  InitKNNParam(param.knn_param);

  for (i = 1; i < argc; ++i) {
    if (argv[i][0] != '-') break;
    if ((i+1) >= argc)
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

  if (i >= argc)
    ExitWithHelp();
  strcpy(data_file_name, argv[i]);
  if ((i+1) < argc) {
    std::strcpy(output_file_name, argv[i+1]);
  } else {
    char *p = std::strrchr(argv[i],'/');
    if (p == NULL) {
      p = argv[i];
    } else {
      ++p;
    }
    std::sprintf(output_file_name, "%s_output", p);
  }

  return;
}