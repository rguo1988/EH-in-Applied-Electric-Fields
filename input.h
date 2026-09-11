//To Use this input by changing the filename to input.h
/***********************************
 ***********************************/
#ifndef _input_h
#define _input_h
#define _USE_MATH_DEFINES
#include<cmath>
#include<iostream>
#include<iomanip>
#include<string>

#ifndef EIGEN_STACK_ALLOCATION_LIMIT
#define EIGEN_STACK_ALLOCATION_LIMIT 0
#endif

#include<eigen3/Eigen/Core>
#include<eigen3/Eigen/QR>
#include<eigen3/Eigen/Dense>
#include<eigen3/Eigen/SparseCore>
#include<eigen3/Eigen/SparseCholesky>

using namespace std;
using namespace Eigen;

class Input
{
    // functions switch
#define _ions_motion true
// #define _Debye_boundary_condition
// #define _Dirichlet_boundary_condition
#define _periodic_boundary_condition
#define _self_consistent_potetial_calculation true
#define _E_Applied true

    // parameter settings
  protected:
    //title
    const string title = "EField resists CHS self-acceleration";

    //plasma parameters
    const double e = -1.0;
    const double n = 1.0;

    const double Te = 1.0; //temperature
    const double me = 1.0;
    const double vt_e = sqrt(Te / me);
    const double wp_e = sqrt(n*e*e / me);
    const double l_e = sqrt(Te / n*e*e); //Debye length

    const double mi = 1836.0;
    const double Ti = 5.0;
    const double vt_i = sqrt(Ti / mi); // boundary speed of wb model
    const double wp_i = sqrt(n*e*e / mi);
    const double l_i = sqrt(Ti / n*e*e);

    const double cs = sqrt(Te/mi);

    //simulation constant
    const double k = 1.0;
    const int N_k = 30;
    const double L = N_k * 2.0 * M_PI / k; //simulaiton length
    static const int nx = 4000;//grid num is nx-1; grid point num is nx
    static const int nx_grids = nx - 1;
    const double dx = L / nx_grids;

    const double vmax_e = 10.0 * vt_e;
    static const int nv_e = 2000;
    static const int nv_e_grids = nv_e - 1;
    const double dv_e = 2 * vmax_e / nv_e_grids;

    const double vmax_i = 10.0 * vt_i;
    static const int nv_i = 2000;
    static const int nv_i_grids = nv_i - 1;
    const double dv_i = 2 * vmax_i / nv_i_grids;

    const double dt = 0.05;
    const int max_steps = 30000;

    //special parameters
    const double ue = 0.0 * vt_e;
    const double ui = 0.15 * vt_e;
    const double xp = L / 2;
    const double psi = 0.1 * abs(e) / Te;
    const double del = 5 * l_e;

    const double E_amp = 0.3;
    const double E_start = 5;
    const double E_end = 505;

    const double E_slp = 0.1; // slope of E in time evolution

    VectorXd phi_sc;

    //data recording
    const string data_path = "./data/";
    const int data_steps = int(max_steps / 1);
    const int data_num = max_steps / data_steps + 1;

    void CalculatePotentialSC()
    {
        phi_sc.resize(nx);
        for(int i = 0; i < nx; i++)
        {
            phi_sc(i) = psi * pow(cosh((i * dx - xp) / del), -2);
        }
        phi_sc[0] = 0.0;
        phi_sc[nx - 1] = 0.0;
    }

    double ElecDistrib(double x, double v)
    {
        int x_idx = x / dx;
        double phi = phi_sc[x_idx];
        double We = 0.5 * pow((v) / vt_e, 2) - abs(e) * phi / Te;
        double r = 0.0;

        // self-consistent EH
        double mu = mi / me;
        if ((v) / vt_e >= sqrt(2.0 * abs(e) * phi / Te))
            r = exp( -pow( sqrt(We) + ue / vt_e / M_SQRT2, 2) ) / sqrt(2.0 * M_PI) / vt_e;
        else if ((v) / vt_e <= -sqrt(2.0 * abs(e) * phi / Te))
            r = exp( -pow(-sqrt(We) + ue / vt_e / M_SQRT2, 2) ) / sqrt(2.0 * M_PI) / vt_e;
        else if (We < 0)
        {
            // passing electron contrib
            double r1 = exp(-We) * erfc(sqrt(-We)) / sqrt(M_PI) / M_SQRT2 / vt_e;
            // double r1 = (I1(sqrt(-We), ue / M_SQRT2 / vt_e) + I1(sqrt(-We), -ue / M_SQRT2 / vt_e)) / pow(M_PI, 1.5) / sqrt(2) / vt_e;
            // potential shape contrib
            double r2 = 8.0 / M_PI / del / del * sqrt(-We) * (2 * We / psi + 1) / vt_e / M_SQRT2; //sech2

            r = r1 + r2;
        }
        return r;
    }

// ion distribution for CHS/pure soliton
    double IonDistrib(double x, double v)
    {
        double Wi = 0.5 * pow((v + ui) / vt_i, 2);
        double r = exp(-Wi) / sqrt(2.0 * M_PI) / vt_i;

        return r;
        // return 1.0;
    }

    double GetElecInitDistrib(double x, double v)
    {
        return ElecDistrib(x, v);
    }

    double GetElecFreeDistrib(double x, double v)
    {
        return GetElecInitDistrib(0.0, v);
    }

    double GetIonInitDistrib(double x, double v)
    {
        return IonDistrib(x, v);
    }

    double GetIonFreeDistrib(double x, double v)
    {
        return GetIonInitDistrib(0.0, v);
    }

    double AppliedEField(double x, int n)
    {
        double r = E_amp * sin(k*x)
		   * (
                       1.0 / (1.0 + exp(-(n * dt - E_start) / E_slp))
                       + 1.0 / (1.0 + exp((n * dt - E_end) / E_slp))
                       - 1.0
                   );
        return r;
    }

    void PrintSpecialParameters()
    {
        cout << " Special Parameters: " << endl;
        cout << "    ue = " << setw(7) << setprecision(5) << ue
             << "    ui = " << setw(7) << setprecision(5) << ui
             << "    cs = " << setw(7) << setprecision(5) << cs<<endl;
        cout << "   x_c = " << setw(7) << setprecision(5) << xp
             << "   psi = " << setw(7) << setprecision(5) << psi
             << "   del = " << setw(7) << setprecision(5) << del << endl;
        cout << " E_amp = " << setw(7) << setprecision(5) << E_amp
             << " start = " << setw(7) << setprecision(5) << E_start
             << "   end = " << setw(7) << setprecision(5) << E_end << endl;
        cout << "--------------------------------------------------------------------------------" << endl;
        cout << " Parameters Max/Min: " << endl;
    }
};
#endif
