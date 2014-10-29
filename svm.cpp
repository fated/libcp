#include "svm.h"
#include <cmath>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <cstdarg>
#include <string>
#include <vector>
#include <exception>

typedef float Qfloat;
typedef signed char schar;

static void PrintCout(const char *s) {
  std::cout << s;
  std::cout.flush();
}

static void PrintNull(const char *s) {}

static void (*PrintString) (const char *) = &PrintNull;

static void Info(const char *format, ...) {
  char buffer[BUFSIZ];
  va_list ap;
  va_start(ap, format);
  vsprintf(buffer, format, ap);
  va_end(ap);
  (*PrintString)(buffer);
}

//
// Kernel Cache
//
// l is the number of total data items
// size is the cache size limit in bytes
//
class Cache {
 public:
  Cache(int l, long int size);
  ~Cache();

  // request data [0,len)
  // return some position p where [p,len) need to be filled
  // (p >= len if nothing needs to be filled)
  int get_data(const int index, Qfloat **data, int len);
  void SwapIndex(int i, int j);

 private:
  int l_;
  long int size_;
  struct Head {
    Head *prev, *next;  // a circular list
    Qfloat *data;
    int len;  // data[0,len) is cached in this entry
  };

  Head *head_;
  Head lru_head_;
  void DeleteLRU(Head *h);
  void InsertLRU(Head *h);
};

Cache::Cache(int l, long int size) : l_(l), size_(size) {
  head_ = (Head *)calloc(static_cast<size_t>(l_), sizeof(Head));  // initialized to 0
  size_ /= sizeof(Qfloat);
  size_ -= static_cast<unsigned long>(l_) * sizeof(Head) / sizeof(Qfloat);
  size_ = std::max(size_, 2 * static_cast<long int>(l_));  // cache must be large enough for two columns
  lru_head_.next = lru_head_.prev = &lru_head_;
}

Cache::~Cache() {
  for (Head *h = lru_head_.next; h != &lru_head_; h=h->next) {
    delete[] h->data;
  }
  delete[] head_;
}

void Cache::DeleteLRU(Head *h) {
  // delete from current location
  h->prev->next = h->next;
  h->next->prev = h->prev;
}

void Cache::InsertLRU(Head *h) {
  // insert to last position
  h->next = &lru_head_;
  h->prev = lru_head_.prev;
  h->prev->next = h;
  h->next->prev = h;
}

int Cache::get_data(const int index, Qfloat **data, int len) {
  Head *h = &head_[index];
  if (h->len) {
    DeleteLRU(h);
  }
  int more = len - h->len;

  if (more > 0) {
    // free old space
    while (size_ < more) {
      Head *old = lru_head_.next;
      DeleteLRU(old);
      delete[] old->data;
      size_ += old->len;
      old->data = 0;
      old->len = 0;
    }

    // allocate new space
    h->data = (Qfloat *)realloc(h->data, sizeof(Qfloat)*static_cast<unsigned long>(len));
    size_ -= more;
    std::swap(h->len, len);
  }

  InsertLRU(h);
  *data = h->data;

  return len;
}

void Cache::SwapIndex(int i, int j) {
  if (i == j) {
    return;
  }

  if (head_[i].len) {
    DeleteLRU(&head_[i]);
  }
  if (head_[j].len) {
    DeleteLRU(&head_[j]);
  }
  std::swap(head_[i].data, head_[j].data);
  std::swap(head_[i].len, head_[j].len);
  if (head_[i].len) {
    InsertLRU(&head_[i]);
  }
  if (head_[j].len) {
    InsertLRU(&head_[j]);
  }

  if (i > j) {
    std::swap(i, j);
  }
  for (Head *h = lru_head_.next; h != &lru_head_; h = h->next) {
    if (h->len > i) {
      if (h->len > j) {
        std::swap(h->data[i], h->data[j]);
      } else {
        // give up
        DeleteLRU(h);
        delete[] h->data;
        size_ += h->len;
        h->data = 0;
        h->len = 0;
      }
    }
  }
}

//
// Kernel evaluation
//
// the static method KernelFunction is for doing single kernel evaluation
// the constructor of Kernel prepares to calculate the l*l kernel matrix
// the member function get_Q is for getting one column from the Q Matrix
//
class QMatrix {
 public:
  virtual Qfloat *get_Q(int column, int len) const = 0;
  virtual double *get_QD() const = 0;
  virtual void SwapIndex(int i, int j) const = 0;
  virtual ~QMatrix() {}
};

class Kernel : public QMatrix {
 public:
  Kernel(int l, Node *const *x, const SVMParameter& param);
  virtual ~Kernel();
  static double KernelFunction(const Node *x, const Node *y, const SVMParameter& param);
  virtual Qfloat *get_Q(int column, int len) const = 0;
  virtual double *get_QD() const = 0;
  virtual void SwapIndex(int i, int j) const {
    std::swap(x_[i], x_[j]);
    if (x_square_) {
      std::swap(x_square_[i], x_square_[j]);
    }
  }

 protected:
  double (Kernel::*kernel_function)(int i, int j) const;

 private:
  const Node **x_;
  double *x_square_;

  // SVMParameter
  const int kernel_type;
  const int degree;
  const double gamma;
  const double coef0;

  static double Dot(const Node *px, const Node *py);
  double KernelLinear(int i, int j) const {
    return Dot(x_[i], x_[j]);
  }
  double KernelPoly(int i, int j) const {
    return std::pow(gamma*Dot(x_[i], x_[j])+coef0, degree);
  }
  double KernelRBF(int i, int j) const {
    return exp(-gamma*(x_square_[i]+x_square_[j]-2*Dot(x_[i], x_[j])));
  }
  double KernelSigmoid(int i, int j) const {
    return tanh(gamma*Dot(x_[i], x_[j])+coef0);
  }
  double KernelPrecomputed(int i, int j) const {
    return x_[i][static_cast<int>(x_[j][0].value)].value;
  }
};

Kernel::Kernel(int l, Node *const *x, const SVMParameter &param)
    :kernel_type(param.kernel_type),
     degree(param.degree),
     gamma(param.gamma),
     coef0(param.coef0) {
  switch (kernel_type) {
    case LINEAR: {
      kernel_function = &Kernel::KernelLinear;
      break;
    }
    case POLY: {
      kernel_function = &Kernel::KernelPoly;
      break;
    }
    case RBF: {
      kernel_function = &Kernel::KernelRBF;
      break;
    }
    case SIGMOID: {
      kernel_function = &Kernel::KernelSigmoid;
      break;
    }
    case PRECOMPUTED: {
      kernel_function = &Kernel::KernelPrecomputed;
      break;
    }
    default: {
      // assert(false);
      break;
    }
  }

  clone(x_, x, l);

  if (kernel_type == RBF) {
    x_square_ = new double[l];
    for (int i = 0; i < l; ++i) {
      x_square_[i] = Dot(x_[i], x_[i]);
    }
  } else {
    x_square_ = 0;
  }
}

Kernel::~Kernel() {
  delete[] x_;
  delete[] x_square_;
}

double Kernel::Dot(const Node *px, const Node *py) {
  double sum = 0;
  while (px->index != -1 && py->index != -1) {
    if (px->index == py->index) {
      sum += px->value * py->value;
      ++px;
      ++py;
    } else {
      if (px->index > py->index) {
        ++py;
      } else {
        ++px;
      }
    }
  }

  return sum;
}

double Kernel::KernelFunction(const Node *x, const Node *y, const SVMParameter &param) {
  switch (param.kernel_type) {
    case LINEAR: {
      return Dot(x, y);
    }
    case POLY: {
      return std::pow(param.gamma*Dot(x, y)+param.coef0, param.degree);
    }
    case RBF: {
      double sum = 0;
      while (x->index != -1 && y->index != -1) {
        if (x->index == y->index) {
          double d = x->value - y->value;
          sum += d*d;
          ++x;
          ++y;
        } else {
          if (x->index > y->index) {
            sum += y->value * y->value;
            ++y;
          } else {
            sum += x->value * x->value;
            ++x;
          }
        }
      }

      while (x->index != -1) {
        sum += x->value * x->value;
        ++x;
      }

      while (y->index != -1) {
        sum += y->value * y->value;
        ++y;
      }

      return exp(-param.gamma*sum);
    }
    case SIGMOID: {
      return tanh(param.gamma*Dot(x, y)+param.coef0);
    }
    case PRECOMPUTED: {  //x: test (validation), y: SV
      return x[static_cast<int>(y->value)].value;
    }
    default: {
      // assert(false);
      return 0;  // Unreachable
    }
  }
}

// An SMO algorithm in Fan et al., JMLR 6(2005), p. 1889--1918
// Solves:
//
//  min 0.5(\alpha^T Q \alpha) + p^T \alpha
//
//    y^T \alpha = \delta
//    y_i = +1 or -1
//    0 <= alpha_i <= Cp for y_i = 1
//    0 <= alpha_i <= Cn for y_i = -1
//
// Given:
//
//  Q, p, y, Cp, Cn, and an initial feasible point \alpha
//  l is the size of vectors and matrices
//  eps is the stopping tolerance
//
// solution will be put in \alpha, objective value will be put in obj
//
class Solver {
 public:
  Solver() {};
  virtual ~Solver() {};

  struct SolutionInfo {
    double obj;
    double rho;
    double upper_bound_p;
    double upper_bound_n;
    double r;  // for Solver_NU
  };

  void Solve(int l, const QMatrix &Q, const double *p, const schar *y,
      double *alpha, double Cp, double Cn, double eps,
      SolutionInfo *si, int shrinking);

 protected:
  int active_size_;
  schar *y_;
  double *G_;  // gradient of objective function
  enum { LOWER_BOUND, UPPER_BOUND, FREE };
  char *alpha_status_;  // LOWER_BOUND, UPPER_BOUND, FREE
  double *alpha_;
  const QMatrix *Q_;
  const double *QD_;
  double eps_;
  double Cp_;
  double Cn_;
  double *p_;
  int *active_set_;
  double *G_bar_;  // gradient, if we treat free variables as 0
  int l_;
  bool unshrink_;  // XXX

  double get_C(int i) {
    return (y_[i] > 0) ? Cp_ : Cn_;
  }
  void UpdateAlphaStatus(int i) {
    if (alpha_[i] >= get_C(i)) {
      alpha_status_[i] = UPPER_BOUND;
    } else {
      if (alpha_[i] <= 0) {
        alpha_status_[i] = LOWER_BOUND;
      } else {
        alpha_status_[i] = FREE;
      }
    }
  }
  bool IsUpperBound(int i) {
    return alpha_status_[i] == UPPER_BOUND;
  }
  bool IsLowerBound(int i) {
    return alpha_status_[i] == LOWER_BOUND;
  }
  bool IsFree(int i) {
    return alpha_status_[i] == FREE;
  }
  void SwapIndex(int i, int j);
  void ReconstructGradient();
  virtual int SelectWorkingSet(int &i, int &j);
  virtual double CalculateRho();
  virtual void DoShrinking();

 private:
  bool IsShrunk(int i, double Gmax1, double Gmax2);
};

void Solver::SwapIndex(int i, int j) {
  Q_->SwapIndex(i, j);
  std::swap(y_[i], y_[j]);
  std::swap(G_[i], G_[j]);
  std::swap(alpha_status_[i], alpha_status_[j]);
  std::swap(alpha_[i], alpha_[j]);
  std::swap(p_[i], p_[j]);
  std::swap(active_set_[i], active_set_[j]);
  std::swap(G_bar_[i], G_bar_[j]);
}

void Solver::ReconstructGradient() {
  // reconstruct inactive elements of G from G_bar_ and free variables
  if (active_size_ == l_) {
    return;
  }

  int num_free = 0;

  for (int i = active_size_; i < l_; ++i) {
    G_[i] = G_bar_[i] + p_[i];
  }

  for (int i = 0; i < active_size_; ++i) {
    if (IsFree(i)) {
      num_free++;
    }
  }

  if (2*num_free < active_size_) {
    Info("\nWARNING: using -h 0 may be faster\n");
  }

  if (num_free*l_ > 2*active_size_*(l_-active_size_)) {
    for (int i = active_size_; i < l_; ++i) {
      const Qfloat *Q_i = Q_->get_Q(i, active_size_);
      for (int j = 0; j < active_size_; ++j) {
        if (IsFree(j)) {
          G_[i] += alpha_[j] * Q_i[j];
        }
      }
    }
  } else {
    for (int i = 0; i < active_size_; ++i) {
      if (IsFree(i)) {
        const Qfloat *Q_i = Q_->get_Q(i, l_);
        double alpha_i = alpha_[i];
        for (int j = active_size_; j < l_; ++j) {
          G_[j] += alpha_i * Q_i[j];
        }
      }
    }
  }
}

void Solver::Solve(int l, const QMatrix &Q, const double *p, const schar *y,
    double *alpha, double Cp, double Cn, double eps,
    SolutionInfo *si, int shrinking) {
  l_ = l;
  Q_ = &Q;
  QD_=Q.get_QD();
  clone(p_, p, l);
  clone(y_, y, l);
  clone(alpha_, alpha, l);
  Cp_ = Cp;
  Cn_ = Cn;
  eps_ = eps;
  unshrink_ = false;

  // initialize alpha_status_
  alpha_status_ = new char[l];
  for (int i = 0; i < l; ++i) {
    UpdateAlphaStatus(i);
  }

  // initialize active set (for shrinking)
  active_set_ = new int[l];
  for (int i = 0; i < l; ++i) {
    active_set_[i] = i;
  }
  active_size_ = l;

  // initialize gradient
  G_ = new double[l];
  G_bar_ = new double[l];
  for (int i = 0; i < l; ++i) {
    G_[i] = p_[i];
    G_bar_[i] = 0;
  }
  for (int i = 0; i < l; ++i)
    if (!IsLowerBound(i)) {
      const Qfloat *Q_i = Q.get_Q(i,l);
      double alpha_i = alpha_[i];
      for (int j = 0; j < l; ++j) {
        G_[j] += alpha_i*Q_i[j];
      }
      if (IsUpperBound(i)) {
        for (int j = 0; j < l; ++j) {
          G_bar_[j] += get_C(i) * Q_i[j];
        }
      }
    }

  // optimization step
  int iter = 0;
  int max_iter = std::max(10000000, (l>INT_MAX/100) ? (INT_MAX) : (100*l));
  int counter = std::min(l, 1000) + 1;

  while (iter < max_iter) {
    // show progress and do shrinking
    if (--counter == 0) {
      counter = std::min(l, 1000);
      if (shrinking) {
        DoShrinking();
      }
      Info(".");
    }

    int i, j;
    if (SelectWorkingSet(i, j) != 0) {
      // reconstruct the whole gradient
      ReconstructGradient();
      // reset active set size and check
      active_size_ = l;
      Info("*");
      if (SelectWorkingSet(i, j) != 0) {
        break;
      } else {
        counter = 1;  // do shrinking next iteration
      }
    }

    ++iter;

    // update alpha[i] and alpha[j], handle bounds carefully
    const Qfloat *Q_i = Q.get_Q(i, active_size_);
    const Qfloat *Q_j = Q.get_Q(j, active_size_);

    double C_i = get_C(i);
    double C_j = get_C(j);

    double old_alpha_i = alpha_[i];
    double old_alpha_j = alpha_[j];

    if (y_[i] != y_[j]) {
      double quad_coef = QD_[i] + QD_[j] + 2*Q_i[j];
      if (quad_coef <= 0) {
        quad_coef = kTau;
      }
      double delta = (-G_[i]-G_[j]) / quad_coef;
      double diff = alpha_[i] - alpha_[j];
      alpha_[i] += delta;
      alpha_[j] += delta;

      if (diff > 0) {
        if (alpha_[j] < 0) {
          alpha_[j] = 0;
          alpha_[i] = diff;
        }
      } else {
        if (alpha_[i] < 0) {
          alpha_[i] = 0;
          alpha_[j] = -diff;
        }
      }
      if (diff > C_i - C_j) {
        if (alpha_[i] > C_i) {
          alpha_[i] = C_i;
          alpha_[j] = C_i - diff;
        }
      } else {
        if (alpha_[j] > C_j) {
          alpha_[j] = C_j;
          alpha_[i] = C_j + diff;
        }
      }
    } else {
      double quad_coef = QD_[i] + QD_[j] - 2*Q_i[j];
      if (quad_coef <= 0) {
        quad_coef = kTau;
      }
      double delta = (G_[i]-G_[j]) / quad_coef;
      double sum = alpha_[i] + alpha_[j];
      alpha_[i] -= delta;
      alpha_[j] += delta;

      if (sum > C_i) {
        if (alpha_[i] > C_i) {
          alpha_[i] = C_i;
          alpha_[j] = sum - C_i;
        }
      } else {
        if (alpha_[j] < 0) {
          alpha_[j] = 0;
          alpha_[i] = sum;
        }
      }
      if (sum > C_j) {
        if (alpha_[j] > C_j) {
          alpha_[j] = C_j;
          alpha_[i] = sum - C_j;
        }
      } else {
        if (alpha_[i] < 0) {
          alpha_[i] = 0;
          alpha_[j] = sum;
        }
      }
    }

    // update G
    double delta_alpha_i = alpha_[i] - old_alpha_i;
    double delta_alpha_j = alpha_[j] - old_alpha_j;

    for (int k = 0; k < active_size_; ++k) {
      G_[k] += Q_i[k]*delta_alpha_i + Q_j[k]*delta_alpha_j;
    }

    // update alpha_status_ and G_bar_
    bool ui = IsUpperBound(i);
    bool uj = IsUpperBound(j);
    UpdateAlphaStatus(i);
    UpdateAlphaStatus(j);
    if (ui != IsUpperBound(i)) {
      Q_i = Q.get_Q(i, l);
      if (ui) {
        for (int k = 0; k < l; ++k) {
          G_bar_[k] -= C_i * Q_i[k];
        }
      } else {
        for (int k = 0; k < l; ++k) {
          G_bar_[k] += C_i * Q_i[k];
        }
      }
    }

    if (uj != IsUpperBound(j)) {
      Q_j = Q.get_Q(j, l);
      if (uj) {
        for (int k = 0; k < l; ++k) {
          G_bar_[k] -= C_j * Q_j[k];
        }
      } else {
        for (int k = 0; k < l; ++k) {
          G_bar_[k] += C_j * Q_j[k];
        }
      }
    }
  }

  if (iter >= max_iter) {
    if (active_size_ < l) {
      // reconstruct the whole gradient to calculate objective value
      ReconstructGradient();
      active_size_ = l;
      Info("*");
    }
    std::cerr << "\nWARNING: reaching max number of iterations" << std::endl;
  }

  // calculate rho
  si->rho = CalculateRho();

  // calculate objective value
  double v = 0;
  for (int i = 0; i < l; ++i) {
    v += alpha_[i] * (G_[i] + p_[i]);
  }
  si->obj = v / 2;

  // put back the solution
  for (int i = 0; i < l; ++i) {
    alpha[active_set_[i]] = alpha_[i];
  }

  // juggle everything back
  /*{
    for(int i=0;i<l;i++)
      while(active_set_[i] != i)
        SwapIndex(i,active_set_[i]);
        // or Q.SwapIndex(i,active_set_[i]);
  }*/

  si->upper_bound_p = Cp;
  si->upper_bound_n = Cn;

  Info("\noptimization finished, #iter = %d\n", iter);

  delete[] p_;
  delete[] y_;
  delete[] alpha_;
  delete[] alpha_status_;
  delete[] active_set_;
  delete[] G_;
  delete[] G_bar_;
}

// return 1 if already optimal, return 0 otherwise
int Solver::SelectWorkingSet(int &out_i, int &out_j) {
  // return i,j such that
  // i: maximizes -y_i * grad(f)_i, i in I_up(\alpha)
  // j: minimizes the decrease of obj value
  //    (if quadratic coefficeint <= 0, replace it with tau)
  //    -y_j*grad(f)_j < -y_i*grad(f)_i, j in I_low(\alpha)

  double Gmax = -kInf;
  double Gmax2 = -kInf;
  int Gmax_idx = -1;
  int Gmin_idx = -1;
  double obj_diff_min = kInf;

  for (int t = 0; t < active_size_; ++t)
    if (y_[t] == +1) {
      if (!IsUpperBound(t)) {
        if (-G_[t] >= Gmax) {
          Gmax = -G_[t];
          Gmax_idx = t;
        }
      }
    } else {
      if (!IsLowerBound(t)) {
        if (G_[t] >= Gmax) {
          Gmax = G_[t];
          Gmax_idx = t;
        }
      }
    }

  int i = Gmax_idx;
  const Qfloat *Q_i = NULL;
  if (i != -1) {// NULL Q_i not accessed: Gmax=-kInf if i=-1
    Q_i = Q_->get_Q(i, active_size_);
  }

  for (int j = 0; j < active_size_; ++j) {
    if (y_[j] == +1) {
      if (!IsLowerBound(j)) {
        double grad_diff = Gmax + G_[j];
        if (G_[j] >= Gmax2) {
          Gmax2 = G_[j];
        }
        if (grad_diff > 0) {
          double obj_diff;
          double quad_coef = QD_[i] + QD_[j] - 2.0*y_[i]*Q_i[j];
          if (quad_coef > 0) {
            obj_diff = -(grad_diff*grad_diff) / quad_coef;
          } else {
            obj_diff = -(grad_diff*grad_diff) / kTau;
          }

          if (obj_diff <= obj_diff_min) {
            Gmin_idx = j;
            obj_diff_min = obj_diff;
          }
        }
      }
    } else {
      if (!IsUpperBound(j)) {
        double grad_diff = Gmax - G_[j];
        if (-G_[j] >= Gmax2) {
          Gmax2 = -G_[j];
        }
        if (grad_diff > 0) {
          double obj_diff;
          double quad_coef = QD_[i] + QD_[j] + 2.0*y_[i]*Q_i[j];
          if (quad_coef > 0) {
            obj_diff = -(grad_diff*grad_diff) / quad_coef;
          } else {
            obj_diff = -(grad_diff*grad_diff) / kTau;
          }
          if (obj_diff <= obj_diff_min) {
            Gmin_idx = j;
            obj_diff_min = obj_diff;
          }
        }
      }
    }
  }

  if (Gmax+Gmax2 < eps_) {
    return 1;
  }

  out_i = Gmax_idx;
  out_j = Gmin_idx;

  return 0;
}

bool Solver::IsShrunk(int i, double Gmax1, double Gmax2) {
  if (IsUpperBound(i)) {
    if (y_[i] == +1) {
      return(-G_[i] > Gmax1);
    } else {
      return(-G_[i] > Gmax2);
    }
  } else {
    if (IsLowerBound(i)) {
      if (y_[i] == +1) {
        return(G_[i] > Gmax2);
      } else {
        return(G_[i] > Gmax1);
      }
    } else {
      return (false);
    }
  }
}

void Solver::DoShrinking() {
  double Gmax1 = -kInf;    // max { -y_i * grad(f)_i | i in I_up(\alpha) }
  double Gmax2 = -kInf;    // max { y_i * grad(f)_i | i in I_low(\alpha) }

  // find maximal violating pair first
  for (int i = 0; i < active_size_; ++i) {
    if (y_[i] == +1) {
      if (!IsUpperBound(i)) {
        if (-G_[i] >= Gmax1) {
          Gmax1 = -G_[i];
        }
      }
      if (!IsLowerBound(i)) {
        if (G_[i] >= Gmax2) {
          Gmax2 = G_[i];
        }
      }
    } else {
      if (!IsUpperBound(i)) {
        if (-G_[i] >= Gmax2) {
          Gmax2 = -G_[i];
        }
      }
      if (!IsLowerBound(i)) {
        if (G_[i] >= Gmax1) {
          Gmax1 = G_[i];
        }
      }
    }
  }

  if ((unshrink_ == false) && (Gmax1 + Gmax2 <= eps_*10)) {
    unshrink_ = true;
    ReconstructGradient();
    active_size_ = l_;
    Info("*");
  }

  for (int i = 0; i < active_size_; ++i) {
    if (IsShrunk(i, Gmax1, Gmax2)) {
      active_size_--;
      while (active_size_ > i) {
        if (!IsShrunk(active_size_, Gmax1, Gmax2)) {
          SwapIndex(i, active_size_);
          break;
        }
        active_size_--;
      }
    }
  }
}

double Solver::CalculateRho() {
  double r;
  int num_free = 0;
  double ub = kInf, lb = -kInf, sum_free = 0;
  for (int i = 0; i < active_size_; ++i) {
    double yG = y_[i] * G_[i];

    if (IsUpperBound(i)) {
      if (y_[i] == -1) {
        ub = std::min(ub, yG);
      } else {
        lb = std::max(lb, yG);
      }
    } else {
      if (IsLowerBound(i)) {
        if (y_[i] == +1) {
          ub = std::min(ub, yG);
        } else {
          lb = std::max(lb, yG);
        }
      } else {
        ++num_free;
        sum_free += yG;
      }
    }
  }

  if (num_free > 0) {
    r = sum_free / num_free;
  } else {
    r = (ub+lb) / 2;
  }

  return r;
}

//
// Solver for nu-svm classification and regression
//
// additional constraint: e^T \alpha = constant
//
class Solver_NU : public Solver {
 public:
  Solver_NU() {}
  void Solve(int l, const QMatrix& Q, const double *p, const schar *y,
      double *alpha, double Cp, double Cn, double eps,
      SolutionInfo* si, int shrinking) {
    si_ = si;
    Solver::Solve(l, Q, p, y, alpha, Cp, Cn, eps, si, shrinking);
  }

 private:
  SolutionInfo *si_;
  int SelectWorkingSet(int &i, int &j);
  double CalculateRho();
  bool IsShrunk(int i, double Gmax1, double Gmax2, double Gmax3, double Gmax4);
  void DoShrinking();
};

// return 1 if already optimal, return 0 otherwise
int Solver_NU::SelectWorkingSet(int &out_i, int &out_j) {
  // return i,j such that y_i = y_j and
  // i: maximizes -y_i * grad(f)_i, i in I_up(\alpha)
  // j: minimizes the decrease of obj value
  //    (if quadratic coefficeint <= 0, replace it with tau)
  //    -y_j*grad(f)_j < -y_i*grad(f)_i, j in I_low(\alpha)

  double Gmaxp = -kInf;
  double Gmaxp2 = -kInf;
  int Gmaxp_idx = -1;

  double Gmaxn = -kInf;
  double Gmaxn2 = -kInf;
  int Gmaxn_idx = -1;

  int Gmin_idx = -1;
  double obj_diff_min = kInf;

  for (int t = 0; t < active_size_; ++t)
    if (y_[t] == +1) {
      if (!IsUpperBound(t)) {
        if (-G_[t] >= Gmaxp) {
          Gmaxp = -G_[t];
          Gmaxp_idx = t;
        }
      }
    } else {
      if (!IsLowerBound(t)) {
        if (G_[t] >= Gmaxn) {
          Gmaxn = G_[t];
          Gmaxn_idx = t;
        }
      }
    }

  int ip = Gmaxp_idx;
  int in = Gmaxn_idx;
  const Qfloat *Q_ip = NULL;
  const Qfloat *Q_in = NULL;
  if (ip != -1) {// NULL Q_ip not accessed: Gmaxp=-kInf if ip=-1
    Q_ip = Q_->get_Q(ip, active_size_);
  }
  if (in != -1) {
    Q_in = Q_->get_Q(in, active_size_);
  }

  for (int j = 0; j < active_size_; ++j) {
    if (y_[j] == +1) {
      if (!IsLowerBound(j)) {
        double grad_diff = Gmaxp + G_[j];
        if (G_[j] >= Gmaxp2) {
          Gmaxp2 = G_[j];
        }
        if (grad_diff > 0) {
          double obj_diff;
          double quad_coef = QD_[ip] + QD_[j] - 2*Q_ip[j];
          if (quad_coef > 0) {
            obj_diff = -(grad_diff*grad_diff) / quad_coef;
          } else {
            obj_diff = -(grad_diff*grad_diff) / kTau;
          }
          if (obj_diff <= obj_diff_min) {
            Gmin_idx = j;
            obj_diff_min = obj_diff;
          }
        }
      }
    } else {
      if (!IsUpperBound(j)) {
        double grad_diff = Gmaxn - G_[j];
        if (-G_[j] >= Gmaxn2) {
          Gmaxn2 = -G_[j];
        }
        if (grad_diff > 0) {
          double obj_diff;
          double quad_coef = QD_[in] + QD_[j] - 2*Q_in[j];
          if (quad_coef > 0) {
            obj_diff = -(grad_diff*grad_diff) / quad_coef;
          } else {
            obj_diff = -(grad_diff*grad_diff) / kTau;
          }
          if (obj_diff <= obj_diff_min) {
            Gmin_idx = j;
            obj_diff_min = obj_diff;
          }
        }
      }
    }
  }

  if (std::max(Gmaxp+Gmaxp2, Gmaxn+Gmaxn2) < eps_) {
    return 1;
  }

  if (y_[Gmin_idx] == +1) {
    out_i = Gmaxp_idx;
  } else {
    out_i = Gmaxn_idx;
  }
  out_j = Gmin_idx;

  return 0;
}

bool Solver_NU::IsShrunk(int i, double Gmax1, double Gmax2, double Gmax3, double Gmax4) {
  if (IsUpperBound(i)) {
    if (y_[i] == +1) {
      return(-G_[i] > Gmax1);
    } else {
      return(-G_[i] > Gmax4);
    }
  } else {
    if (IsLowerBound(i)) {
      if (y_[i] == +1) {
        return(G_[i] > Gmax2);
      } else {
        return(G_[i] > Gmax3);
      }
    } else {
      return (false);
    }
  }
}

void Solver_NU::DoShrinking() {
  double Gmax1 = -kInf;  // max { -y_i * grad(f)_i | y_i = +1, i in I_up(\alpha) }
  double Gmax2 = -kInf;  // max { y_i * grad(f)_i | y_i = +1, i in I_low(\alpha) }
  double Gmax3 = -kInf;  // max { -y_i * grad(f)_i | y_i = -1, i in I_up(\alpha) }
  double Gmax4 = -kInf;  // max { y_i * grad(f)_i | y_i = -1, i in I_low(\alpha) }

  // find maximal violating pair first
  for (int i = 0; i < active_size_; ++i) {
    if (!IsUpperBound(i)) {
      if (y_[i] == +1) {
        if (-G_[i] > Gmax1) {
          Gmax1 = -G_[i];
        }
      } else {
        if (-G_[i] > Gmax4) {
          Gmax4 = -G_[i];
        }
      }
    }
    if (!IsLowerBound(i)) {
      if (y_[i] == +1) {
        if(G_[i] > Gmax2) {
          Gmax2 = G_[i];
        }
      } else {
        if (G_[i] > Gmax3) {
          Gmax3 = G_[i];
        }
      }
    }
  }

  if ((unshrink_ == false) && (std::max(Gmax1+Gmax2, Gmax3+Gmax4) <= eps_*10)) {
    unshrink_ = true;
    ReconstructGradient();
    active_size_ = l_;
  }

  for (int i = 0; i < active_size_; ++i)
    if (IsShrunk(i, Gmax1, Gmax2, Gmax3, Gmax4)) {
      active_size_--;
      while (active_size_ > i) {
        if (!IsShrunk(active_size_, Gmax1, Gmax2, Gmax3, Gmax4)) {
          SwapIndex(i, active_size_);
          break;
        }
        active_size_--;
      }
    }
}

double Solver_NU::CalculateRho() {
  int num_free1 = 0, num_free2 = 0;
  double ub1 = kInf, ub2 = kInf;
  double lb1 = -kInf, lb2 = -kInf;
  double sum_free1 = 0, sum_free2 = 0;

  for (int i = 0; i < active_size_; ++i) {
    if (y_[i] == +1) {
      if (IsUpperBound(i)) {
        lb1 = std::max(lb1, G_[i]);
      } else {
        if (IsLowerBound(i)) {
          ub1 = std::min(ub1, G_[i]);
        } else {
          ++num_free1;
          sum_free1 += G_[i];
        }
      }
    } else {
      if (IsUpperBound(i)) {
        lb2 = std::max(lb2, G_[i]);
      } else {
        if (IsLowerBound(i)) {
          ub2 = std::min(ub2, G_[i]);
        } else {
          ++num_free2;
          sum_free2 += G_[i];
        }
      }
    }
  }

  double r1, r2;
  if (num_free1 > 0) {
    r1 = sum_free1 / num_free1;
  } else {
    r1 = (ub1+lb1) / 2;
  }

  if (num_free2 > 0) {
    r2 = sum_free2 / num_free2;
  } else {
    r2 = (ub2+lb2) / 2;
  }

  si_->r = (r1+r2) / 2;
  return ((r1-r2)/2);
}

//
// Q matrices for various formulations
//
class SVC_Q : public Kernel {
 public:
  SVC_Q(const Problem &prob, const SVMParameter &param, const schar *y) : Kernel(prob.num_ex, prob.x, param) {
    clone(y_, y, prob.num_ex);
    cache_ = new Cache(prob.num_ex, static_cast<long int>(param.cache_size*(1<<20)));
    QD_ = new double[prob.num_ex];
    for (int i = 0; i < prob.num_ex; ++i)
      QD_[i] = (this->*kernel_function)(i, i);
  }

  Qfloat *get_Q(int i, int len) const {
    Qfloat *data;
    int start = cache_->get_data(i, &data, len);
    if (start < len) {
      for (int j = start; j < len; ++j)
        data[j] = static_cast<Qfloat>(y_[i]*y_[j]*(this->*kernel_function)(i, j));
    }
    return data;
  }

  double *get_QD() const {
    return QD_;
  }

  void SwapIndex(int i, int j) const {
    cache_->SwapIndex(i, j);
    Kernel::SwapIndex(i, j);
    std::swap(y_[i], y_[j]);
    std::swap(QD_[i], QD_[j]);
  }

  ~SVC_Q() {
    delete[] y_;
    delete cache_;
    delete[] QD_;
  }

 private:
  schar *y_;
  Cache *cache_;
  double *QD_;
};

//
// construct and solve various formulations
//
static void SolveCSVC(const Problem *prob, const SVMParameter *param, double *alpha, Solver::SolutionInfo *si, double Cp, double Cn) {
  int num_ex = prob->num_ex;
  double *minus_ones = new double[num_ex];
  schar *y = new schar[num_ex];

  for (int i = 0; i < num_ex; ++i) {
    alpha[i] = 0;
    minus_ones[i] = -1;
    if (prob->y[i] > 0) {
      y[i] = +1;
    } else {
      y[i] = -1;
    }
  }

  Solver s;
  s.Solve(num_ex, SVC_Q(*prob, *param, y), minus_ones, y, alpha, Cp, Cn, param->eps, si, param->shrinking);

  double sum_alpha=0;
  for (int i = 0; i < num_ex; ++i) {
    sum_alpha += alpha[i];
  }

  if (Cp == Cn) {
    Info("nu = %f\n", sum_alpha/(Cp*prob->num_ex));
  }

  for (int i = 0; i < num_ex; ++i) {
    alpha[i] *= y[i];
  }

  delete[] minus_ones;
  delete[] y;
}

static void SolveNuSVC(const Problem *prob, const SVMParameter *param, double *alpha, Solver::SolutionInfo *si) {
  int num_ex = prob->num_ex;
  double nu = param->nu;

  schar *y = new schar[num_ex];

  for (int i = 0; i < num_ex; ++i) {
    if (prob->y[i] > 0) {
      y[i] = +1;
    } else {
      y[i] = -1;
    }
  }

  double sum_pos = nu*num_ex/2;
  double sum_neg = nu*num_ex/2;

  for (int i = 0; i < num_ex; ++i) {
    if (y[i] == +1) {
      alpha[i] = std::min(1.0, sum_pos);
      sum_pos -= alpha[i];
    } else {
      alpha[i] = std::min(1.0, sum_neg);
      sum_neg -= alpha[i];
    }
  }

  double *zeros = new double[num_ex];

  for (int i = 0; i < num_ex; ++i) {
    zeros[i] = 0;
  }

  Solver_NU s;
  s.Solve(num_ex, SVC_Q(*prob, *param, y), zeros, y, alpha, 1.0, 1.0, param->eps, si, param->shrinking);
  double r = si->r;

  Info("C = %f\n", 1/r);

  for (int i = 0; i < num_ex; ++i) {
    alpha[i] *= y[i]/r;
  }

  si->rho /= r;
  si->obj /= (r*r);
  si->upper_bound_p = 1/r;
  si->upper_bound_n = 1/r;

  delete[] y;
  delete[] zeros;
}

//
// DecisionFunction
//
struct DecisionFunction {
  double *alpha;
  double rho;
};

static DecisionFunction TrainSingleSVM(const Problem *prob, const SVMParameter *param, double Cp, double Cn) {
  double *alpha = new double[prob->num_ex];
  Solver::SolutionInfo si;
  switch (param->svm_type) {
    case C_SVC: {
      SolveCSVC(prob, param, alpha, &si, Cp, Cn);
      break;
    }
    case NU_SVC: {
      SolveNuSVC(prob, param, alpha, &si);
      break;
    }
    default: {
      // assert{false};
      break;
    }
  }

  Info("obj = %f, rho = %f\n", si.obj, si.rho);

  // output SVs
  int nSV = 0;
  int nBSV = 0;
  for (int i = 0; i < prob->num_ex; ++i) {
    if (fabs(alpha[i]) > 0) {
      ++nSV;
      if (prob->y[i] > 0) {
        if (fabs(alpha[i]) >= si.upper_bound_p) {
          ++nBSV;
        }
      } else {
        if (fabs(alpha[i]) >= si.upper_bound_n) {
          ++nBSV;
        }
      }
    }
  }

  Info("nSV = %d, nBSV = %d\n", nSV, nBSV);

  DecisionFunction f;
  f.alpha = alpha;
  f.rho = si.rho;
  return f;
}

//
// Interface functions
//
SVMModel *TrainSVM(const Problem *prob, const SVMParameter *param) {
  SVMModel *model = new SVMModel;
  model->param = *param;
  model->free_sv = 0;  // XXX

  // classification
  int num_ex = prob->num_ex;
  int num_classes;
  int *labels = NULL;
  int *start = NULL;
  int *count = NULL;
  int *perm = new int[num_ex];

  // group training data of the same class
  GroupClasses(prob, &num_classes, &labels, &start, &count, perm);
  if (num_classes == 1) {
    Info("WARNING: training data in only one class. See README for details.\n");
  }

  Node **x = new Node*[num_ex];
  for (int i = 0; i < num_ex; ++i) {
    x[i] = prob->x[perm[i]];
  }

  // calculate weighted C
  double *weighted_C = new double[num_classes];
  for (int i = 0; i < num_classes; ++i) {
    weighted_C[i] = param->C;
  }
  for (int i = 0; i < param->num_weights; ++i) {
    int j;
    for (j = 0; j < num_classes; ++j) {
      if (param->weight_labels[i] == labels[j]) {
        break;
      }
    }
    if (j == num_classes) {
      std::cerr << "WARNING: class label " << param->weight_labels[i] << " specified in weight is not found" << std::endl;
    } else {
      weighted_C[j] *= param->weights[i];
    }
  }

  // train k*(k-1)/2 models
  bool *non_zero = new bool[num_ex];
  for (int i = 0; i < num_ex; ++i) {
    non_zero[i] = false;
  }
  DecisionFunction *f = new DecisionFunction[num_classes*(num_classes-1)/2];

  int p = 0;
  for (int i = 0; i < num_classes; ++i) {
    for (int j = i+1; j < num_classes; ++j) {
      Problem sub_prob;
      int si = start[i], sj = start[j];
      int ci = count[i], cj = count[j];
      sub_prob.num_ex = ci+cj;
      sub_prob.x = new Node*[sub_prob.num_ex];
      sub_prob.y = new double[sub_prob.num_ex];
      for (int k = 0; k < ci; ++k) {
        sub_prob.x[k] = x[si+k];
        sub_prob.y[k] = +1;
      }
      for (int k = 0; k < cj; ++k) {
        sub_prob.x[ci+k] = x[sj+k];
        sub_prob.y[ci+k] = -1;
      }

      f[p] = TrainSingleSVM(&sub_prob, param,weighted_C[i], weighted_C[j]);
      for (int k = 0; k < ci; ++k) {
        if (!non_zero[si+k] && fabs(f[p].alpha[k]) > 0) {
          non_zero[si+k] = true;
        }
      }
      for (int k = 0; k < cj; ++k) {
        if (!non_zero[sj+k] && fabs(f[p].alpha[ci+k]) > 0) {
          non_zero[sj+k] = true;
        }
      }
      delete[] sub_prob.x;
      delete[] sub_prob.y;
      ++p;
    }
  }

  // build output
  model->num_classes = num_classes;
  model->num_ex = num_ex;

  model->labels = new int[num_classes];
  for (int i = 0; i < num_classes; ++i) {
    model->labels[i] = labels[i];
  }

  model->rho = new double[num_classes*(num_classes-1)/2];
  for (int i = 0; i < num_classes*(num_classes-1)/2; ++i) {
    model->rho[i] = f[i].rho;
  }

  int total_sv = 0;
  int *nz_count = new int[num_classes];
  model->num_svs = new int[num_classes];
  for (int i = 0; i < num_classes; ++i) {
    int num_svs = 0;
    for (int j = 0; j < count[i]; ++j) {
      if (non_zero[start[i]+j]) {
        ++num_svs;
        ++total_sv;
      }
    }
    model->num_svs[i] = num_svs;
    nz_count[i] = num_svs;
  }

  Info("Total nSV = %d\n", total_sv);

  model->total_sv = total_sv;
  model->svs = new Node*[total_sv];
  model->sv_indices = new int[total_sv];
  p = 0;
  for (int i = 0; i < num_ex; ++i) {
    if (non_zero[i]) {
      model->svs[p] = x[i];
      model->sv_indices[p] = perm[i] + 1;
      ++p;
    }
  }

  int *nz_start = new int[num_classes];
  nz_start[0] = 0;
  for (int i = 1; i < num_classes; ++i) {
    nz_start[i] = nz_start[i-1]+nz_count[i-1];
  }

  model->sv_coef = new double*[num_classes-1];
  for (int i = 0; i < num_classes-1; ++i) {
    model->sv_coef[i] = new double[total_sv];
  }

  p = 0;
  for (int i = 0; i < num_classes; ++i) {
    for (int j = i+1; j < num_classes; ++j) {
      // classifier (i,j): coefficients with
      // i are in sv_coef[j-1][nz_start[i]...],
      // j are in sv_coef[i][nz_start[j]...]
      int si = start[i];
      int sj = start[j];
      int ci = count[i];
      int cj = count[j];

      int q = nz_start[i];
      for (int k = 0; k < ci; ++k) {
        if (non_zero[si+k]) {
          model->sv_coef[j-1][q++] = f[p].alpha[k];
        }
      }
      q = nz_start[j];
      for (int k = 0; k < cj; ++k) {
        if (non_zero[sj+k]) {
          model->sv_coef[i][q++] = f[p].alpha[ci+k];
        }
      }
      ++p;
    }
  }

  delete[] labels;
  delete[] count;
  delete[] perm;
  delete[] start;
  delete[] x;
  delete[] weighted_C;
  delete[] non_zero;
  for (int i = 0; i < num_classes*(num_classes-1)/2; ++i) {
    delete[] f[i].alpha;
  }
  delete[] f;
  delete[] nz_count;
  delete[] nz_start;

  return model;
}

double PredictSVMValues(const SVMModel *model, const Node *x, double *decision_values) {
  int num_classes = model->num_classes;
  int total_sv = model->total_sv;

  double *kvalue = new double[total_sv];
  for (int i = 0; i < total_sv; ++i) {
    kvalue[i] = Kernel::KernelFunction(x, model->svs[i], model->param);
  }

  int *start = new int[num_classes];
  start[0] = 0;
  for (int i = 1; i < num_classes; ++i) {
    start[i] = start[i-1] + model->num_svs[i-1];
  }

  int *vote = new int[num_classes];
  for (int i = 0; i < num_classes; ++i) {
    vote[i] = 0;
  }

  int p = 0;
  for (int i = 0; i < num_classes; ++i) {
    for (int j = i+1; j < num_classes; ++j) {
      double sum = 0;
      int si = start[i];
      int sj = start[j];
      int ci = model->num_svs[i];
      int cj = model->num_svs[j];

      double *coef1 = model->sv_coef[j-1];
      double *coef2 = model->sv_coef[i];
      for (int k = 0; k < ci; ++k) {
        sum += coef1[si+k] * kvalue[si+k];
      }
      for (int k = 0; k < cj; ++k) {
        sum += coef2[sj+k] * kvalue[sj+k];
      }
      sum -= model->rho[p];
      decision_values[p] = sum;

      if (decision_values[p] > 0) {
        ++vote[i];
      } else {
        ++vote[j];
      }
      ++p;
    }
  }

  int vote_max_idx = 0;
  for (int i = 1; i < num_classes; ++i) {
    if (vote[i] > vote[vote_max_idx]) {
      vote_max_idx = i;
    }
  }

  delete[] kvalue;
  delete[] start;
  delete[] vote;

  return model->labels[vote_max_idx];
}

double PredictSVM(const SVMModel *model, const Node *x) {
  int num_classes = model->num_classes;
  double *decision_values = new double[num_classes*(num_classes-1)/2];
  double pred_result = PredictSVMValues(model, x, decision_values);
  delete[] decision_values;
  return pred_result;
}

static const char *kSVMTypeTable[] = { "c_svc", "nu_svc", NULL };

static const char *kKernelTypeTable[] = { "linear", "polynomial", "rbf", "sigmoid", "precomputed", NULL };

int SaveSVMModel(std::ofstream &model_file, const struct SVMModel *model) {
  const SVMParameter &param = model->param;

  model_file << "svm_model\n";
  model_file << "svm_type " << kSVMTypeTable[param.svm_type] << '\n';
  model_file << "kernel_type " << kKernelTypeTable[param.kernel_type] << '\n';

  if (param.kernel_type == POLY) {
    model_file << "degree " << param.degree << '\n';
  }
  if (param.kernel_type == POLY ||
      param.kernel_type == RBF  ||
      param.kernel_type == SIGMOID) {
    model_file << "gamma " << param.gamma << '\n';
  }
  if (param.kernel_type == POLY ||
      param.kernel_type == SIGMOID) {
    model_file << "coef0 " << param.coef0 << '\n';
  }

  int num_classes = model->num_classes;
  int total_sv = model->total_sv;
  model_file << "num_examples " << model->num_ex << '\n';
  model_file << "num_classes " << num_classes << '\n';
  model_file << "total_SV " << total_sv << '\n';

  if (model->labels) {
    model_file << "labels";
    for (int i = 0; i < num_classes; ++i)
      model_file << ' ' << model->labels[i];
    model_file << '\n';
  }

  if (model->rho) {
    model_file << "rho";
    for (int i = 0; i < num_classes*(num_classes-1)/2; ++i)
      model_file << ' ' << model->rho[i];
    model_file << '\n';
  }

  if (model->num_svs) {
    model_file << "num_SVs";
    for (int i = 0; i < num_classes; ++i)
      model_file << ' ' << model->num_svs[i];
    model_file << '\n';
  }

  if (model->sv_indices) {
    model_file << "SV_indices\n";
    for (int i = 0; i < total_sv; ++i)
      model_file << model->sv_indices[i] << ' ';
    model_file << '\n';
  }

  model_file << "SVs\n";
  const double *const *sv_coef = model->sv_coef;
  const Node *const *svs = model->svs;

  for (int i = 0; i < total_sv; ++i) {
    for (int j = 0; j < num_classes-1; ++j)
      model_file << std::setprecision(16) << (sv_coef[j][i]+0.0) << ' ';  // add "+0.0" to avoid negative zero in output

    const Node *p = svs[i];

    if (param.kernel_type == PRECOMPUTED) {
      model_file << "0:" << static_cast<int>(p->value) << ' ';
    } else {
      while (p->index != -1) {
        model_file << p->index << ':' << std::setprecision(8) << p->value << ' ';
        ++p;
      }
    }
    model_file << '\n';
  }

  return 0;
}

SVMModel *LoadSVMModel(std::ifstream &model_file) {
  SVMModel *model = new SVMModel;
  SVMParameter &param = model->param;
  model->rho = NULL;
  model->sv_indices = NULL;
  model->labels = NULL;
  model->num_svs = NULL;

  char cmd[80];
  while (1) {
    model_file >> cmd;

    if (std::strcmp(cmd, "svm_type") == 0) {
      model_file >> cmd;
      int i;
      for (i = 0; kSVMTypeTable[i]; ++i) {
        if (std::strcmp(kSVMTypeTable[i], cmd) == 0) {
          param.svm_type = i;
          break;
        }
      }
      if (kSVMTypeTable[i] == NULL) {
        std::cerr << "Unknown SVM type.\n" << std::endl;
        return NULL;
      }
    } else
    if (std::strcmp(cmd, "kernel_type") == 0) {
      model_file >> cmd;
      int i;
      for (i = 0; kKernelTypeTable[i]; ++i) {
        if (std::strcmp(kKernelTypeTable[i], cmd) == 0) {
          param.kernel_type = i;
          break;
        }
      }
      if (kKernelTypeTable[i] == NULL) {
        std::cerr << "Unknown kernel function.\n" << std::endl;
        return NULL;
      }
    } else
    if (std::strcmp(cmd, "degree") == 0) {
      model_file >> param.degree;
    } else
    if (std::strcmp(cmd, "gamma") == 0) {
      model_file >> param.gamma;
    } else
    if (std::strcmp(cmd, "coef0") == 0) {
      model_file >> param.coef0;
    } else
    if (std::strcmp(cmd, "num_examples") == 0) {
      model_file >> model->num_ex;
    } else
    if (std::strcmp(cmd, "num_classes") == 0) {
      model_file >> model->num_classes;
    } else
    if (std::strcmp(cmd, "total_SV") == 0) {
      model_file >> model->total_sv;
    } else
    if (std::strcmp(cmd, "labels") == 0) {
      int n = model->num_classes;
      model->labels = new int[n];
      for (int i = 0; i < n; ++i) {
        model_file >> model->labels[i];
      }
    } else
    if (std::strcmp(cmd, "rho") == 0) {
      int n = model->num_classes*(model->num_classes-1)/2;
      model->rho = new double[n];
      for (int i = 0; i < n; ++i) {
        model_file >> model->rho[i];
      }
    } else
    if (std::strcmp(cmd, "num_SVs") == 0) {
      int n = model->num_classes;
      model->num_svs = new int[n];
      for (int i = 0; i < n; ++i) {
        model_file >> model->num_svs[i];
      }
    } else
    if (std::strcmp(cmd, "SV_indices") == 0) {
      int n = model->total_sv;
      model->sv_indices = new int[n];
      for (int i = 0; i < n; ++i) {
        model_file >> model->sv_indices[i];
      }
    } else
    if (std::strcmp(cmd, "SVs") == 0) {
      std::size_t m = static_cast<unsigned long>(model->num_classes)-1;
      int total_sv = model->total_sv;
      std::string line;

      if (model_file.peek() == '\n')
        model_file.get();

      model->sv_coef = new double*[m];
      for (int i = 0; i < m; ++i) {
        model->sv_coef[i] = new double[total_sv];
      }
      model->svs = new Node*[total_sv];
      for (int i = 0; i < total_sv; ++i) {
        std::vector<std::string> tokens;
        std::size_t prev = 0, pos;

        std::getline(model_file, line);
        while ((pos = line.find_first_of(" \t\n", prev)) != std::string::npos) {
          if (pos > prev)
            tokens.push_back(line.substr(prev, pos-prev));
          prev = pos + 1;
        }
        if (prev < line.length())
          tokens.push_back(line.substr(prev, std::string::npos));

        for (std::size_t j = 0; j < m; ++j) {
          try
          {
            std::size_t end;
            model->sv_coef[j][i] = std::stod(tokens[j], &end);
            if (end != tokens[j].length()) {
              throw std::invalid_argument("incomplete convention");
            }
          }
          catch(std::exception& e)
          {
            std::cerr << "Error: " << e.what() << " in SV " << (i+1) << std::endl;
            delete[] model->svs;
            for (int j = 0; j < m; ++j) {
              delete[] model->sv_coef[j];
            }
            delete[] model->sv_coef;
            std::vector<std::string>(tokens).swap(tokens);
            exit(EXIT_FAILURE);
          }  // TODO try not to use exception
        }

        std::size_t elements = tokens.size() - m + 1;
        model->svs[i] = new Node[elements];
        prev = 0;
        for (std::size_t j = 0; j < elements-1; ++j) {
          pos = tokens[j+m].find_first_of(':');
          try
          {
            std::size_t end;

            model->svs[i][j].index = std::stoi(tokens[j+m].substr(prev, pos-prev), &end);
            if (end != (tokens[j+m].substr(prev, pos-prev)).length()) {
              throw std::invalid_argument("incomplete convention");
            }
            model->svs[i][j].value = std::stod(tokens[j+m].substr(pos+1), &end);
            if (end != (tokens[j+m].substr(pos+1)).length()) {
              throw std::invalid_argument("incomplete convention");
            }
          }
          catch(std::exception& e)
          {
            std::cerr << "Error: " << e.what() << " in line " << (i+1) << std::endl;
            for (int k = 0; k < m; ++k) {
              delete[] model->sv_coef[k];
            }
            delete[] model->sv_coef;
            for (int k = 0; k < i+1; ++k) {
              delete[] model->svs[k];
            }
            delete[] model->svs;
            std::vector<std::string>(tokens).swap(tokens);
            exit(EXIT_FAILURE);
          }
        }
        model->svs[i][elements-1].index = -1;
        model->svs[i][elements-1].value = 0;
      }
      break;
    } else {
      std::cerr << "Unknown text in knn_model file: " << cmd << std::endl;
      FreeSVMModel(&model);
      return NULL;
    }
  }
  model->free_sv = 1;
  return model;
}

void FreeSVMModelContent(SVMModel *model) {
  if (model->free_sv && model->total_sv > 0 && model->svs != NULL) {
    delete[] model->svs;
    model->svs = NULL;
  }

  if (model->sv_coef) {
    for (int i = 0; i < model->num_classes-1; ++i)
      delete[] model->sv_coef[i];
  }

  if (model->svs) {
    delete[] model->svs;
    model->svs = NULL;
  }

  if (model->sv_coef) {
    delete[] model->sv_coef;
    model->sv_coef = NULL;
  }

  if (model->rho) {
    delete[] model->rho;
    model->rho = NULL;
  }

  if (model->labels) {
    delete[] model->labels;
    model->labels= NULL;
  }

  if (model->sv_indices) {
    delete[] model->sv_indices;
    model->sv_indices = NULL;
  }

  if (model->num_svs) {
    delete[] model->num_svs;
    model->num_svs = NULL;
  }
}

void FreeSVMModel(SVMModel** model)
{
  if (model != NULL && *model != NULL) {
    FreeSVMModelContent(*model);
    delete *model;
    *model = NULL;
  }

  return;
}

void FreeSVMParam(SVMParameter* param) {
  if (param->weight_labels) {
    delete[] param->weight_labels;
    param->weight_labels = NULL;
  }
  if (param->weights) {
    delete[] param->weights;
    param->weights = NULL;
  }
  delete param;
  param = NULL;

  return;
}

const char *CheckSVMParameter(const SVMParameter *param) {
  int svm_type = param->svm_type;
  if (svm_type != C_SVC &&
      svm_type != NU_SVC)
    return "unknown svm type";

  int kernel_type = param->kernel_type;
  if (kernel_type != LINEAR &&
      kernel_type != POLY &&
      kernel_type != RBF &&
      kernel_type != SIGMOID &&
      kernel_type != PRECOMPUTED)
    return "unknown kernel type";

  if (param->gamma < 0)
    return "gamma < 0";

  if (param->degree < 0)
    return "degree of polynomial kernel < 0";

  if (param->cache_size <= 0)
    return "cache_size <= 0";

  if (param->eps <= 0)
    return "eps <= 0";

  if (svm_type == C_SVC)
    if (param->C <= 0)
      return "C <= 0";

  if (svm_type == NU_SVC)
    if (param->nu <= 0 || param->nu > 1)
      return "nu <= 0 or nu > 1";

  if (param->shrinking != 0 &&
      param->shrinking != 1)
    return "shrinking != 0 and shrinking != 1";

  return NULL;
}

void InitSVMParam(struct SVMParameter *param) {
  param->svm_type = C_SVC;
  param->kernel_type = RBF;
  param->degree = 3;
  param->gamma = 0;  // default 1/num_features
  param->coef0 = 0;
  param->nu = 0.5;
  param->cache_size = 100;
  param->C = 1;
  param->eps = 1e-3;
  param->shrinking = 1;
  param->num_weights = 0;
  param->weight_labels = NULL;
  param->weights = NULL;
  SetPrintCout();

  return;
}

void SetPrintNull() {
  PrintString = &PrintNull;
}

void SetPrintCout() {
  PrintString = &PrintCout;
}