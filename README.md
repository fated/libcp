# LIBCP -- A Library for Conformal Prediction

LibCP is a simple, easy-to-use, and efficient software for Conformal Prediction on classification, which gives prediction together with confidence and credibility. It solves conformal prediction in both online and batch mode with *k*-nearest neighbors as the underlying algorithm. This document explains the use of LibCP.

## Table of Contents

* [Installation and Data Format](#installation-and-data-format)
* ["cp-offline" Usage](#cp-offline-usage)
* ["cp-online" Usage](#cp-online-usage)
* ["cp-cv" Usage](#cp-cv-usage)
* [Tips on Practical Use](#tips-on-practical-use)
* [Examples](#examples)
* [Library Usage](#library-usage)
* [Additional Information](#additional-information)
* [Acknowledgments](#acknowledgments)

## Installation and Data Format[↩](#table-of-contents)

On Unix systems, type `make` to build the `cp-offline`, `cp-online` and `cp-cv` programs. Run them without arguments to show the usage of them.

The format of training and testing data file is:
```
<label> <index1>:<value1> <index2>:<value2> ...
...
...
...
```

Each line contains an instance and is ended by a `'\n'` character (Unix line ending). For classification, `<label>` is an integer indicating the class label (multi-class is supported). For regression, `<label>` is the target value which can be any real number. The pair `<index>:<value>` gives a feature (attribute) value: `<index>` is an integer starting from 1 and `<value>` is the value of the attribute, which could be an integer number or real number. Indices must be in **ASCENDING** order. Labels in the testing file are only used to calculate accuracies and errors. If they are unknown, just fill the first column with any numbers.

A sample classification data set included in this package is `iris_scale` for training and `iris_scale_t` for testing.

Type `cp-offline iris_scale iris_scale_t`, and the program will read the training data and testing data and then output the result into `iris_scale_t_output` file by default. The model file `iris_scale_model` will not be saved by default, however, adding `-s model_file_name` to `[option]` will save the model to `model_file_name`. The output file contains the predicted labels and the lower and upper bounds of probabilities for each predicted label.

## "cp-offline" Usage[↩](#table-of-contents)
```
Usage: cp-offline [options] train_file test_file [output_file]
options:
  -t non-conformity measure : set type of NCM (default 0)
    0 -- k-nearest neighbors (KNN)
  -k num_neighbors : set number of neighbors in kNN (default 1)
  -s model_file_name : save model
  -l model_file_name : load model
  -e epsilon : set significance level (default 0.05)
```
`train_file` is the data you want to train with.  
`test_file` is the data you want to predict.  
`cp-offline` will produce outputs in the `output_file` by default.

## "cp-online" Usage[↩](#table-of-contents)
```
Usage: cp-online [options] data_file [output_file]
options:
  -t non-conformity measure : set type of NCM (default 0)
    0 -- k-nearest neighbors (KNN)
  -k num_neighbors : set number of neighbors in kNN (default 1)
  -e epsilon : set significance level (default 0.05)
```
`data_file` is the data you want to run the online prediction on.  
`cp-online` will produce outputs in the `output_file` by default.

## "cp-cv" Usage[↩](#table-of-contents)
```
Usage: cp-cv [options] data_file [output_file]
options:
  -t non-conformity measure : set type of NCM (default 0)
    0 -- k-nearest neighbors (KNN)
  -k num_neighbors : set number of neighbors in kNN (default 1)
  -v num_folds : set number of folders in cross validation (default 5)
  -e epsilon : set significance level (default 0.05)
```
`data_file` is the data you want to run the cross validation on.  
`cp-cv` will produce outputs in the `output_file` by default.

## Tips on Practical Use[↩](#table-of-contents)
* Scale your data. For example, scale each attribute to [0,1] or [-1,+1].
* Try different non-conformity measures. Some non-conformity measures will not achieve good results on some data sets.
* Change parameters for better results.

## Examples[↩](#table-of-contents)
```
> cp-offline -k 3 train_file test_file output_file
```

Train a conformal predictor with 3-nearest neighbors as non-conformity measure from `train_file`. Then conduct this classifier to `test_file` and output the results to `output_file`.

```
> cp-online data_file
```

Train an online conformal predictor classifier using nearest neighbour as non-conformity measure from `data_file`. Then output the results to the default output file.

```
> cp-cv -v 10 data_file
```

Do a 10-fold cross validation conformal predictor using nearest neighbour as non-conformity measure from `data_file`. Then output the results to the default output file.

## Library Usage[↩](#table-of-contents)
All functions and structures are declared in different header files. There are 4 parts in this library, which are **utilities**, **knn**, **cp** and the other driver programs.

### `utilities.h` and `utilities.cpp`
The structure `Problem` for storing the data sets (including the structure `Node` for storing the attributes pair of index and value) and all the constant variables are declared in `utilities.h`.

In this file, some utilizable function templates or functions are also declared.

* `T FindMostFrequent(T *array, int size)`  
  This function is used to find the most frequent category in *k*NN taxonomy.
* `static inline void clone(T *&dest, S *src, int size)`  
  This static function is used to clone an array from `src` to `dest`.
* `void QuickSortIndex(T array[], size_t index[], size_t left, size_t right)`  
  This function is used to quicksort an array and preserve the original indices.
* `Problem *ReadProblem(const char *file_name)`  
  This function is used to read in a data set from a file named `file_name`.
* `void FreeProblem(struct Problem *problem)`  
  This function is used to free a problem stored in the memory.
* `void GroupClasses(const Problem *prob, int *num_classes_ret, int **labels_ret, int **start_ret, int **count_ret, int *perm)`  
  This function is used in Cross Validation. This function will group the examples with same label together. The last 5 parameters are using to return corresponding values. `num_classes_ret` is used to store the number of classes in the problem. `labels_ret` is an array used to store the actual label in the order of appearance. `start_ret` is an array used to store the starting index of each group of examples. `count_ret` is an array used to store the count number of each group of examples. `perm` is an array used to store the permutation of the permuted index of the problem.
* `int *GetLabels(const Problem *prob, int *num_classes_ret)`
  This function is used to get label list of `prob`. The label list will store in an integer array as the return value, and the number of classes `num_classes_ret` will also be returned.

### `knn.h` and `knn.cpp`
The structure `KNNParameter` for storing the *k*NN related parameters and the structure `KNNModel` for storing the *k*NN related model are declared in `knn.h`.

In this file, some utilizable function templates or functions are also declared.

* `static inline void InsertLabel(T *labels, T label, int num_neighbors, int index)`  
  This static function will insert `label` into the `index`-th location of the array `labels` of which the size is `num_neighbors`.
* `KNNModel *TrainKNN(const struct Problem *prob, const struct KNNParameter *param)`  
  This function is used to train a *k*NN model from a problem `prob` and the parameter `param`, it will return a model of the structure `KNNModel`.
* `double PredictKNN(struct Problem *train, struct Node *x, const int num_neighbors)`  
  This function is used to predict the label for object `x` using *k*NN classifier.
* `double CalcDist(const struct Node *x1, const struct Node *x2)`  
  This function is used to calculate the distance between two objects `x1` and `x2`, which will be used in *k*NN.
* `int CompareDist(double *neighbors, double dist, int num_neighbors)`  
  This function is used to compare a distance `dist` with the nearest neighbors' distances stored in an array `neighbors`, it will return the position of `dist`, if it is greater than all the distances in `neighbors`, it gives `num_neighbors`.
* `int SaveKNNModel(std::ofstream &model_file, const struct KNNModel *model)`
* `KNNModel *LoadKNNModel(std::ifstream &model_file)`
* `void FreeKNNModel(struct KNNModel *model)`  
  These three functions are used to manipulate the *k*NN model file, including "save to file", "load from file" and "free the model".
* `void FreeKNNParam(struct KNNParameter *param)`
* `void InitKNNParam(struct KNNParameter *param)`
* `const char *CheckKNNParameter(const struct KNNParameter *param)`  
  These three functions are used to manipulate the *k*NN parameter file, including "free the param", "initial the param" and "check the param".

### `cp.h` and `cp.cpp`
The structure `Parameter` for storing the Conformal Prediction related parameters and the structure `Model` for storing the Conformal Prediction related model are declared in `cp.h`. You need to #include "cp.h" in your C/C++ source files and link your program with `cp.cpp`. You can see `cp-offline.cpp`, `cp-online.cpp` and `cp-cv.cpp` for examples showing how to use them.

In this file, some utilizable function templates or functions are also declared.

* `Model *TrainCP(const struct Problem *train, const struct Parameter *param)` 
  This function is used to train a conformal predictor from the problem `train` and the parameter `param`.
* `std::vector<int> PredictCP(const struct Problem *train, const struct Model *model, const struct Node *x, double &conf, double &cred)`  
  This function is used to predict a new object `x` from the problem `train` and the `model`. It will return the prediction set which may contain several labels, the first element in the vector is the simple prediction (regardless of epsilon) and the following elements are prediction set. `conf` for confidence and `cred` for credibility are also returned.
* `void CrossValidation(const struct Problem *prob, const struct Parameter *param, std::vector<int> *predict_labels, double *conf, double *cred)`  
  This function is used to do a cross validation on the problem `prob` and the parameter `param`. The other 3 parameters are used to return the corresponding values.
* `void OnlinePredict(const struct Problem *prob, const struct Parameter *param, std::vector<int> *predict_labels, int *indices, double *conf, double *cred)`  
  This function is used to do a online prediction on the problem `prob` and the parameter `param`. The other 4 parameters are used to return the corresponding values.
* `int SaveModel(const char *model_file_name, const struct Model *model)`
* `Model *LoadModel(const char *model_file_name)`
* `void FreeModel(struct Model *model)`  
  These three functions are used to manipulate the model file, including "save to file", "load from file" and "free the model".
* `void FreeParam(struct Parameter *param)`
* `const char *CheckParameter(const struct Parameter *param)`  
  These two functions are used to manipulate the parameter file, including "free the param" and "check the param".
* `static double CalcAlpha(double *min_same, double *min_diff, int num_neighbors)`
  This function is used to calculate non-conformity score (alpha) in *k*NN NCM. `min_same` is the distance array of the same label, `min_diff` is the distance array of the different labels.

### `cp-offline.cpp`, `cp-online.cpp` and `cp-cv.cpp`
These three files are the driver programs for LibCP. `cp-offline.cpp` is for training and testing data sets in offline setting. `cp-online.cpp` is for doing online prediction on data sets. `cp-cv.cpp` is for doing cross validation on data sets.

The structure of these files are similar. In these programs, the command-line inputs will be parsed, the data sets will be read into the memory, the train and predict process will be called, the performance measure process will be carried out and finally the memories it claimed will be cleaned up. It includes the following functions.

* `void ExitWithHelp()`  
  This function is used to print out the usage of the executable file.
* `void ParseCommandLine(int argc, char *argv[], ...)`  
  This function is used to parse the options from the command-line input, and return the values like file names to the other parameters which is represented by `...`.

## Additional Information[↩](#table-of-contents)
For any questions and comments, please email [c.zhou@cs.rhul.ac.uk](mailto:c.zhou@cs.rhul.ac.uk)

## Acknowledgments[↩](#table-of-contents)
Special thanks to Chih-Chung Chang and Chih-Jen Lin, which are the authors of [LibSVM](http://www.csie.ntu.edu.tw/~cjlin/libsvm/).