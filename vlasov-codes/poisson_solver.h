#ifndef _poisson_solver_h
#define _poisson_solver_h
#include<eigen3/Eigen/Core>
using namespace Eigen;

class PoissonSolver
{
  public:
    const int nx;
    const double dx;
    VectorXd rho;
    VectorXd phi;
    VectorXd E;

    double GetPhi(int x_index);
    double GetE(int x_index);
    PoissonSolver(int _nx, double _dx);
};

class PoissonSolverPeriodicBC: public PoissonSolver
{
  public:
    PoissonSolverPeriodicBC(int _nx, double _dx);
    void Solve(VectorXd _rho);
};

class PoissonSolverDirichletBC: public PoissonSolver
{
  public:
    PoissonSolverDirichletBC(int _nx, double _dx, double c1, double c2);
    void Solve(VectorXd _rho);
};

class PoissonSolverNaturalBC: public PoissonSolver
{
    const double d1;
    const double d2;
  public:
    PoissonSolverNaturalBC(int _nx, double _dx, double _d1, double _d2);
    void Solve(VectorXd _rho);
};

class PoissonSolverDebyeBC: public PoissonSolver
{
    const double l_D;
  public:
    PoissonSolverDebyeBC(int _nx, double _dx, double _l_D);
    void Solve(VectorXd _rho);
};
#endif
