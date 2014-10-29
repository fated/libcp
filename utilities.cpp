#include "utilities.h"
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cmath>
#include <exception>

Problem *ReadProblem(const char *file_name) {
  std::ifstream input_file(file_name);
  if (!input_file.is_open()) {
    std::cerr << "Unable to open input file: " << file_name << std::endl;
    exit(EXIT_FAILURE);
  }

  int max_index, current_max_index;
  std::string line;
  Problem *problem = new Problem;
  problem->num_ex = 0;

  while (std::getline(input_file, line)) {
    ++problem->num_ex;
  }
  input_file.clear();
  input_file.seekg(0);

  problem->y = new double[problem->num_ex];
  problem->x = new Node*[problem->num_ex];

  max_index = 0;
  for (int i = 0; i < problem->num_ex; ++i) {
    std::vector<std::string> tokens;
    std::size_t prev = 0, pos;

    current_max_index = -1;
    std::getline(input_file, line);
    while ((pos = line.find_first_of(" \t\n", prev)) != std::string::npos) {
      if (pos > prev) {
        tokens.push_back(line.substr(prev, pos-prev));
      }
      prev = pos + 1;
    }
    if (prev < line.length()) {
      tokens.push_back(line.substr(prev, std::string::npos));
    }

    try
    {
      std::size_t end;

      problem->y[i] = std::stod(tokens[0], &end);
      if (end != tokens[0].length()) {
        throw std::invalid_argument("incomplete convention");
      }
    }
    catch(std::exception& e)
    {
      std::cerr << "Error: " << e.what() << " in line " << (i+1) << std::endl;
      delete[] problem->y;
      for (int j = 0; j < i; ++j) {
        delete[] problem->x[j];
      }
      delete[] problem->x;
      delete problem;
      std::vector<std::string>(tokens).swap(tokens);
      input_file.close();
      exit(EXIT_FAILURE);
    }  // TODO try not to use exception

    std::size_t elements = tokens.size();
    problem->x[i] = new Node[elements];
    prev = 0;
    for (std::size_t j = 0; j < elements-1; ++j) {
      pos = tokens[j+1].find_first_of(':');
      try
      {
        std::size_t end;
        problem->x[i][j].index = std::stoi(tokens[j+1].substr(prev, pos-prev), &end);
        if (end != (tokens[j+1].substr(prev, pos-prev)).length()) {
          throw std::invalid_argument("incomplete convention");
        }
        problem->x[i][j].value = std::stod(tokens[j+1].substr(pos+1), &end);
        if (end != (tokens[j+1].substr(pos+1)).length()) {
          throw std::invalid_argument("incomplete convention");
        }
      }
      catch(std::exception& e)
      {
        std::cerr << "Error: " << e.what() << " in line " << (i+1) << std::endl;
        delete[] problem->y;
        for (int j = 0; j < i+1; ++j) {
          delete[] problem->x[j];
        }
        delete[] problem->x;
        delete problem;
        std::vector<std::string>(tokens).swap(tokens);
        input_file.close();
        exit(EXIT_FAILURE);
      }
      current_max_index = problem->x[i][j].index;
    }

    if (current_max_index > max_index) {
      max_index = current_max_index;
    }
    problem->x[i][elements-1].index = -1;
    problem->x[i][elements-1].value = 0;
  }
  problem->max_index = max_index;

  // TODO add precomputed kernel check

  input_file.close();

  return problem;
}

void FreeProblem(struct Problem *problem) {
  if (problem->y != NULL) {
    delete[] problem->y;
  }

  for (int i = 0; i < problem->num_ex; ++i) {
    if (problem->x[i] != NULL) {
      delete[] problem->x[i];
    }
  }
  if (problem->x != NULL) {
    delete[] problem->x;
  }
  if (problem != NULL) {
    delete problem;
  }

  return;
}

// label: label name, start: begin of each class, count: #data of classes, perm: indices to the original data
// perm, length l, must be allocated before calling this subroutine
void GroupClasses(const Problem *prob, int *num_classes_ret, int **labels_ret, int **start_ret, int **count_ret, int *perm) {
  int num_ex = prob->num_ex;
  int max_num_classes = 16;
  int num_classes = 0;
  int *labels = new int[max_num_classes];
  int *count = new int[max_num_classes];
  int *data_labels = new int[num_ex];

  for (int i = 0; i < num_ex; ++i) {
    int this_label = static_cast<int>(prob->y[i]);
    int j;
    for (j = 0; j < num_classes; ++j) {
      if (this_label == labels[j]) {
        ++count[j];
        break;
      }
    }
    data_labels[i] = j;
    if (j == num_classes) {
      if (num_classes == max_num_classes) {
        max_num_classes *= 2;
        labels = (int *)realloc(labels, (unsigned long)max_num_classes*sizeof(int));
        count = (int *)realloc(count, (unsigned long)max_num_classes*sizeof(int));
      }
      labels[num_classes] = this_label;
      count[num_classes] = 1;
      ++num_classes;
    }
  }

  //
  // Labels are ordered by their first occurrence in the training set.
  // However, for two-class sets with -1/+1 labels and -1 appears first,
  // we swap labels to ensure that internally the binary SVM has positive data corresponding to the +1 instances.
  //
  if (num_classes == 2 && labels[0] == -1 && labels[1] == 1) {
    std::swap(labels[0], labels[1]);
    std::swap(count[0], count[1]);
    for (int i = 0; i < num_ex; ++i) {
      if (data_labels[i] == 0) {
        data_labels[i] = 1;
      } else {
        data_labels[i] = 0;
      }
    }
  }

  int *start = new int[num_classes];
  start[0] = 0;
  for (int i = 1; i < num_classes; ++i) {
    start[i] = start[i-1] + count[i-1];
  }
  for (int i = 0; i < num_ex; ++i) {
    perm[start[data_labels[i]]] = i;
    ++start[data_labels[i]];
  }
  start[0] = 0;
  for (int i = 1; i < num_classes; ++i) {
    start[i] = start[i-1] + count[i-1];
  }

  *num_classes_ret = num_classes;
  *labels_ret = labels;
  *start_ret = start;
  *count_ret = count;
  delete[] data_labels;

  return;
}