/***************************
 * Solver of Poisson equation
 * Author: rguo
 * Data:   2023-5-25
 * Note:   A combined PoissonSolver
 *         Periodic, Dirichlet, Natural BC by direct Thomas algorithm
 *         Debye BC by shooting method of bvp
 *         Periodic BC requires neutral total charges
 ****************************/
#include<eigen3/Eigen/Core>
#include"poisson_solver.h"
#include<iostream>

using namespace Eigen;

PoissonSolver::PoissonSolver(int _nx, double _dx): nx(_nx), dx(_dx)
{
    phi.resize(nx);
    E.resize(nx);
}

PoissonSolverPeriodicBC::PoissonSolverPeriodicBC(int _nx, double _dx): PoissonSolver(_nx, _dx) { }

//solve poisson equation with periodic condition
void PoissonSolverPeriodicBC::Solve(VectorXd _rho)
{
    rho = _rho;

    //construct laplace opretor
    //here laplace is a reduced laplace oprator
    //e.g. a block(1,1)-(3,3)
    //-2  1  0  1
    //1 -2  1  0
    //0  1 -2  1
    //1  0  1 -2

    double dx2 = dx * dx;
    double r[nx - 1];
    double b[nx - 1];
    r[0] = b[0] = 0.0;
    r[1] = b[1] = 0.0;

    for(int i = 1; i <= nx - 3; i++)
    {
        r[i + 1] = -1.0 / (r[i] - 2);
        b[i + 1] = (-rho[i] * dx2 - b[i]) / (r[i] - 2);
    }

    phi(nx - 2) = (-rho(nx - 2) * dx2 - b[nx - 2]) / (r[nx - 2] - 2);

    for(int j = nx - 2; j >= 2; j--)
    {
        phi[j - 1] = r[j] * phi[j] + b[j];
    }

    phi[0] = 0.0;
    phi[nx - 1] = phi(0);
    //zero potential can be defined as \int_L \phi \dd x = 0
    // VectorXd phi_sum = VectorXd::Constant(nx, phi.sum() / (nx - 1));
    // phi -= phi_sum;

    //calculate E
    for(int i = 0; i <= nx - 2; i++)
    {
        int i_minus = (i == 0) ? (nx - 2) : (i - 1);
        int i_plus = (i == nx - 2) ? (0) : (i + 1);
        E[i] = -(phi[i_plus] - phi[i_minus]) / (2 * dx);
    }
    E[nx - 1] = E[0];
}

PoissonSolverDirichletBC::PoissonSolverDirichletBC(int _nx, double _dx, double c1, double c2): PoissonSolver(_nx, _dx)
{
    phi[0] = c1;
    phi[nx - 1] = c2;
}

//solve poisson equation with Dirichlet condition
void PoissonSolverDirichletBC::Solve(VectorXd _rho)
{
    double dx2 = dx * dx;
    //initialize rho with BC
    rho = _rho;
    rho[1] += phi[0] / dx2;
    rho[nx - 2] += phi[nx - 1] / dx2;

    //construct laplace opretor
    //-2  1  0  0
    //1 -2  1  0
    //0  1 -2  1
    //0  0  1 -2

    double r[nx - 1];
    double b[nx - 1];
    r[0] = b[0] = 0.0;
    r[1] = b[1] = 0.0;
    for(int i = 1; i <= nx - 3; i++)
    {
        r[i + 1] = -1.0 / (r[i] - 2);
        b[i + 1] = (-rho[i] * dx2 - b[i]) / (r[i] - 2);
    }
    phi(nx - 2) = (-rho(nx - 2) * dx2 - b[nx - 2]) / (r[nx - 2] - 2);
    for(int j = nx - 2; j >= 2; j--)
    {
        phi[j - 1] = r[j] * phi[j] + b[j];
    }

    //calculate E
    for(int i = 0; i <= nx - 1; i++)
    {
        double phi_plus = (i == nx - 1) ? phi[nx - 1] : phi[i + 1];
        double phi_minus = (i == 0) ? phi[0] : phi[i - 1];
        E[i] = -(phi_plus - phi_minus) / (2 * dx);
    }
}

PoissonSolverNaturalBC::PoissonSolverNaturalBC(int _nx, double _dx, double _d1, double _d2): PoissonSolver(_nx, _dx), d1(_d1), d2(_d2) { }

//solve poisson equation with Natural condition
void PoissonSolverNaturalBC::Solve(VectorXd _rho)
{
    //initialize rho with BC
    rho = _rho;
    rho[1] -= d1 / dx;
    rho[nx - 2] += d2 / dx;

    //construct laplace opretor
    //-1  1  0  0
    // 1 -2  1  0
    // 0  1 -2  1
    // 0  0  1 -1

    double dx2 = dx * dx;
    double r[nx];
    double b[nx];
    r[0] = b[0] = 0.0; //useless parameters
    r[1] = b[1] = 0.0;
    r[2] = 1.0;
    b[2] = rho[1] * dx2;

    for(int i = 2; i <= nx - 3; i++)
    {
        r[i + 1] = -1.0 / (r[i] - 2);
        b[i + 1] = (-rho[i] * dx2 - b[i]) / (r[i] - 2);
        std::cout<<r[i+1]<<" "<<b[i+1]<<std::endl;
    }

    phi(nx - 2) = (-rho(nx - 2) * dx2 - b[nx - 2]) / (r[nx - 2] - 1.0);
    for(int j = nx - 2; j >= 2; j--)
    {
        phi[j - 1] = r[j] * phi[j] + b[j];
    }
    phi[0] = phi[1] - d1 * dx;
    phi[nx - 1] = phi[nx - 2] + d2 * dx;

    //calculate E
    for(int i = 1; i <= nx - 2; i++)
    {
        E[i] = -(phi[i + 1] - phi[i - 1]) / (2 * dx);
    }
    E[0] = -d1;
    E[nx - 1] = -d2;
}

PoissonSolverDebyeBC::PoissonSolverDebyeBC(int _nx, double _dx, double _l_D): PoissonSolver(_nx, _dx), l_D(_l_D) { }

//solve poisson equation with Debye condition: phi' = phi/lambda_D and phi' = -phi/lambda_D
void PoissonSolverDebyeBC::Solve(VectorXd rho)
{
    //ode: y''=f(x,y,y'); y'(a) = y(a)/l_D; y'(b) = -y(b)/l_D;
    //solve: y''=f(x,y,y'); y(a) = t1; y'(a) = t1/l_D; obtain: e(t1) = y'(b,t1) + y(b,t1)/l_D;
    //do k times:
    //tk = tk-1 - e(t_k-1)*(t_k-1-t_k-2)/ (e(t_k-1)-e(t_k-2))

    double t2 = 0.0;//set t2
    double t1 = 0.1; //set t1
    double t0 = 0; //set t0
    double dx2 = dx * dx;
    double phi00, phi01, phi02 = 0.0;
    double phi10, phi11, phi12 = 0.0;
    for(int j = 1; j < 100; j++)
    {
        phi00 = t0;
        phi10 = t1;
        phi01 = t0 + 2.0 * t0 / l_D * dx;
        phi11 = t1 + 2.0 * t1 / l_D * dx;
        for(int i = 1; i <= nx - 2; i++)
        {
            phi02 = -rho[i] * dx2 + 2.0 * phi01 - phi00;
            phi00 = phi01;
            phi01 = phi02;
            phi12 = -rho[i] * dx2 + 2.0 * phi11 - phi10;
            phi10 = phi11;
            phi11 = phi12;
        }
        double e0 = (phi01 - phi00) / 2.0 / dx + phi01 / l_D;
        double e1 = (phi11 - phi10) / 2.0 / dx + phi11 / l_D;
        if(abs(e1) < 1e-6)
            break;
        t2 = t1 - e1 * (t1 - t0) / (e1 - e0);
        t0 = t1;
        t1 = t2;
    }
    
    phi[0] = t1;
    phi[1] = t1 + 2.0 * t1 / l_D * dx;
    for(int i = 1; i <= nx - 2; i++)
        phi[i + 1] = -rho[i] * dx2 + 2.0 * phi[i] - phi[i - 1];

    //calculate E
    for(int i = 0; i <= nx - 1; i++)
    {
        double phi_plus = (i == nx - 1) ? phi[nx - 1] : phi[i + 1];
        double phi_minus = (i == 0) ? phi[0] : phi[i - 1];
        E[i] = -(phi_plus - phi_minus) / (2 * dx);
    }
}

double PoissonSolver::GetPhi(int x_index)
{
    return phi[x_index];
}
double PoissonSolver::GetE(int x_index)
{
    return E[x_index];
}
