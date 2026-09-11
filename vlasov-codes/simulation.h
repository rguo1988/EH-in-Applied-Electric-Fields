#ifndef _simulation_h
#define _simulation_h
#include"input.h"
#include"poisson_solver.h"
#include<eigen3/Eigen/Core>
#include<vector>
#include"cubic_spline_1d.h"

class Simulation: public Input
{
    Eigen::Matrix<double, nx, 1> x_samples;
    Eigen::Matrix<double, nv_e, 1> v_samples_e;
    Eigen::Matrix<double, nv_i, 1> v_samples_i;
    Eigen::Matrix<double, nx, 1> rho_e;
    Eigen::Matrix<double, nx, 1> rho_i;
    Eigen::Matrix<double, nx, 1> rho;
    Eigen::Matrix<double, nx, 1> E_Applied;
    Eigen::MatrixXd Phi_evolution;
    Eigen::MatrixXd E_evolution;

    vector<double> Ek;
    vector<double> Ek_e;
    vector<double> Ek_i;
    vector<double> Ep;
    vector<double> Et;
    vector<double> pe;
    vector<double> pi;

  public:
    //Create Poisson Solver & interpolation keywords under different BCs
#ifdef _periodic_boundary_condition
    PoissonSolverPeriodicBC poisson_solver;
    CubicSplineInterp1D::BoundaryCondition current_bc_type = CubicSplineInterp1D::periodic;
#elif defined _Dirichlet_boundary_condition
    PoissonSolverDirichletBC poisson_solver;
    CubicSplineInterp1D::BoundaryCondition current_bc_type = CubicSplineInterp1D::fp_zero;
#elif defined _Debye_boundary_condition
    PoissonSolverDebyeBC poisson_solver;
    CubicSplineInterp1D::BoundaryCondition current_bc_type = CubicSplineInterp1D::fp_zero;
#endif

    Eigen::MatrixXd fe;
    Eigen::MatrixXd fi;
    Simulation();
    void Run();
    void ShiftAPeriod(double& x, double L);
    void PrintParameters();
    void DiagnoseEveryStep(int n);
    void DiagnoseAtEnd();
};
#endif
