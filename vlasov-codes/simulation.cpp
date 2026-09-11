#include<iostream>
#include<iomanip>
#include<omp.h>

#include"input.h"
#include<eigen3/Eigen/Dense>
#include"simulation.h"
#include"poisson_solver.h"

using namespace Eigen;
using namespace std;

Simulation::Simulation():
#ifdef _periodic_boundary_condition
    poisson_solver(nx, dx)
#elif defined _Dirichlet_boundary_condition
    poisson_solver(nx, dx, 0.0, 0.0)
#elif defined _Debye_boundary_condition
    poisson_solver(nx, dx, l_e)
#endif
{
    x_samples.setLinSpaced(nx, 0, L);
    v_samples_e.setLinSpaced(nv_e, -vmax_e, vmax_e);
    v_samples_i.setLinSpaced(nv_i, -vmax_i, vmax_i);
    //initialize distribution
    fe.resize(nx, nv_e);
    fe.setZero();
    fi.resize(nx, nv_i);
    fi.setZero();

    //initialize Applied E Field
#if _E_Applied
    E_Applied.resize(nx);
    E_Applied.setZero();
#endif

#if _self_consistent_potetial_calculation
    CalculatePotentialSC();
#endif

    //electrons
    #pragma omp parallel for collapse(2) shared(fe)
    for (int i = 0; i < nx; i++)
    {
        for(int j = 0; j < nv_e; j++)
        {
            fe(i, j) = GetElecInitDistrib(x_samples[i], v_samples_e[j]);
        }
    }
    //ions
#if _ions_motion
    #pragma omp parallel for collapse(2) shared(fi)
    for (int i = 0; i < nx; i++)
    {
        for(int j = 0; j < nv_i; j++)
        {
            fi(i, j) = GetIonInitDistrib(x_samples[i], v_samples_i[j]);
        }
    }
#else
    for (int i = 0; i < nx; i++)
    {
        rho_i[i] = GetIonInitDistrib(x_samples[i], 0.0);
    }
#endif
    Ek.resize(max_steps + 1);
    Ek_e.resize(max_steps + 1);
    Ek_i.resize(max_steps + 1);
    Ep.resize(max_steps + 1);
    Et.resize(max_steps + 1);
    pe.resize(max_steps + 1);
    pi.resize(max_steps + 1);
    Phi_evolution.resize(nx, max_steps + 1);
    Phi_evolution.setZero();
    E_evolution.resize(nx, max_steps + 1);
    E_evolution.setZero();
}

void Simulation::Run()
{
    //set variables for electrons
    double xe_shift = 0.0;
    double ve_shift = 0.0;
    MatrixXd fe_xshift(nx, nv_e);
    fe_xshift.setZero();
    MatrixXd fe_vshift(nx, nv_e);
    fe_vshift.setZero();

#if _ions_motion //mobile ions
    //set variables for ions
    double xi_shift = 0.0;
    double vi_shift = 0.0;
    MatrixXd fi_xshift(nx, nv_i);
    fi_xshift.setZero();
    MatrixXd fi_vshift(nx, nv_i);
    fi_vshift.setZero();
#endif

    PrintParameters();

    for(int n = 0; n < max_steps + 1; n++)
    {
        //print running process
        int percent = 100 * n / (max_steps - 1);
        if(percent % 5 == 0)
        {
            cout << "\r" << "  Process: " << percent << "%" << flush;
        }

        //x shift dt/2
        #pragma omp parallel for schedule(dynamic, 20) shared(fe_xshift) private(xe_shift)
        for(int j = 0; j < nv_e; j++)
        {
            //electrons
            Matrix < double, nx, 1 > fe_fixed_v_samples = fe.col(j);
            CubicSplineInterp1D cubic_spline_interp_xe(x_samples, fe_fixed_v_samples, current_bc_type, CubicSplineInterp1D::equal_interval);

            for(int i = 0; i < nx; i++)
            {
                xe_shift = x_samples(i) - 0.5 * dt * v_samples_e(j);

#ifdef _periodic_boundary_condition
                //periodic bc
                ShiftAPeriod(xe_shift, L);
                fe_xshift(i, j) = cubic_spline_interp_xe.CalcVal(xe_shift);
                //open bc
#elif defined (_Dirichlet_boundary_condition)|| defined (_Debye_boundary_condition)
                if(xe_shift > L || xe_shift < 0.0)
                    fe_xshift(i, j) = GetElecFreeDistrib(i * dx, -vmax_e + j * dv_e);
                else
                    fe_xshift(i, j) = cubic_spline_interp_xe.CalcVal(xe_shift);
#endif

            }
        }
#if _ions_motion
        #pragma omp parallel for schedule(dynamic, 20) shared(fi_xshift) private(xi_shift)
        for(int j = 0; j < nv_i; j++)
        {
            //ions
            Matrix < double, nx, 1 > fi_fixed_v_samples = fi.col(j);
            CubicSplineInterp1D cubic_spline_interp_xi(x_samples, fi_fixed_v_samples, current_bc_type, CubicSplineInterp1D::equal_interval);

            for(int i = 0; i < nx; i++)
            {
                xi_shift = x_samples(i) - 0.5 * dt * v_samples_i(j);

#ifdef _periodic_boundary_condition
                //periodic bc
                ShiftAPeriod(xi_shift, L);
                fi_xshift(i, j) = cubic_spline_interp_xi.CalcVal(xi_shift);
                //open bc
#elif defined (_Dirichlet_boundary_condition) || defined (_Debye_boundary_condition)
                if(xi_shift > L || xi_shift < 0.0)
                    fi_xshift(i, j) = GetIonFreeDistrib(i * dx, -vmax_i + j * dv_i);
                else
                    fi_xshift(i, j) = cubic_spline_interp_xi.CalcVal(xi_shift);
#endif
            }
        }
#endif //for ion motion

        //solve rho
        rho_e = 0.5 * ( fe_xshift.block(0, 0, nx, nv_e - 1).rowwise().sum() + fe_xshift.block(0, 1, nx, nv_e - 1).rowwise().sum() ) * dv_e;
#if _ions_motion
        rho_i = 0.5 * ( fi_xshift.block(0, 0, nx, nv_i - 1).rowwise().sum() + fi_xshift.block(0, 1, nx, nv_i - 1).rowwise().sum() ) * dv_i;
#endif
        rho = rho_i - rho_e;

        //Solve Poisson equations under different BCs
        poisson_solver.Solve(rho);

        //add Applied Electric Field
#if _E_Applied
        E_Applied.setZero();
        for(int i = 0; i < nx; i++)
            E_Applied[i] = AppliedEField(i * dx, n);
        poisson_solver.E += E_Applied;
#endif

        //v shift dt
        #pragma omp parallel for schedule(dynamic, 20) shared(fe_vshift, fi_vshift) private(ve_shift, vi_shift)
        for(int i = 0; i < nx; i++)
        {
            //electrons
            Matrix < double, nv_e, 1 > fe_fixed_x_samples = fe_xshift.row(i);
            CubicSplineInterp1D cubic_spline_interp_ve(v_samples_e, fe_fixed_x_samples, CubicSplineInterp1D::fp_zero, CubicSplineInterp1D::equal_interval);
            for(int j = 1; j < nv_e; j++)
            {
                ve_shift = v_samples_e(j) - (e / me) * poisson_solver.GetE(i) * dt;
                // fe_vshift(i, j) =  0.0;
                if(ve_shift < vmax_e || ve_shift > -vmax_e)
                {
                    fe_vshift(i, j) =  cubic_spline_interp_ve.CalcVal(ve_shift);
                }
                //when v is out of range, f =0, which is set at intial
            }
#if _ions_motion
            //ions
            Matrix < double, nv_i, 1 > fi_fixed_x_samples = fi_xshift.row(i);
            CubicSplineInterp1D cubic_spline_interp_vi(v_samples_i, fi_fixed_x_samples, CubicSplineInterp1D::fp_zero, CubicSplineInterp1D::equal_interval);
            for(int j = 1; j < nv_i; j++)
            {
                vi_shift = v_samples_i(j) + (e / mi) * poisson_solver.GetE(i) * dt;
                // fi_vshift(i, j) =  0.0;
                if(vi_shift < vmax_i || vi_shift > -vmax_i)
                {
                    fi_vshift(i, j) =  cubic_spline_interp_vi.CalcVal(vi_shift);
                }
                //when v is out of range, f =0, which is set at intial
            }
#endif
        }

        //2nd x shift dt/2
        #pragma omp parallel for schedule(dynamic, 20) shared(fe) private(xe_shift)
        for(int j = 0; j < nv_e; j++)
        {
            //electrons
            Matrix < double, nx, 1 > fe_fixed_v_samples2 = fe_vshift.col(j);
            CubicSplineInterp1D cubic_spline_interp_xe2(x_samples, fe_fixed_v_samples2, current_bc_type, CubicSplineInterp1D::equal_interval);
            for(int i = 0; i < nx; i++)
            {
                xe_shift = x_samples(i) - 0.5 * dt * v_samples_e(j);

#ifdef _periodic_boundary_condition
                //periodic bc
                ShiftAPeriod(xe_shift, L);
                fe(i, j) = cubic_spline_interp_xe2.CalcVal(xe_shift);

#elif defined (_Dirichlet_boundary_condition) || defined (_Debye_boundary_condition)
                //open bc
                if(xe_shift > L || xe_shift < 0.0)
                    fe(i, j) = GetElecFreeDistrib(i * dx, -vmax_e + j * dv_e);
                else
                    fe(i, j) = cubic_spline_interp_xe2.CalcVal(xe_shift);
#endif

            }
        }
#if _ions_motion
        #pragma omp parallel for schedule(dynamic, 20) shared(fi) private(xi_shift)
        for(int j = 0; j < nv_i; j++)
        {
            //ions
            Matrix < double, nx, 1 > fi_fixed_v_samples2 = fi_vshift.col(j);
            CubicSplineInterp1D cubic_spline_interp_xi2(x_samples, fi_fixed_v_samples2, current_bc_type, CubicSplineInterp1D::equal_interval);
            for(int i = 0; i < nx; i++)
            {
                xi_shift = x_samples(i) - 0.5 * dt * v_samples_i(j);

#ifdef _periodic_boundary_condition
                //periodic bc
                ShiftAPeriod(xi_shift, L);
                fi(i, j) = cubic_spline_interp_xi2.CalcVal(xi_shift);
#elif defined (_Dirichlet_boundary_condition) || defined (_Debye_boundary_condition)
                //open bc
                if(xi_shift > L || xi_shift < 0.0)
                    fi(i, j) = GetIonFreeDistrib(i * dx, -vmax_i + j * dv_i);
                else
                    fi(i, j) = cubic_spline_interp_xi2.CalcVal(xi_shift);
#endif

            }
        }
#endif
        //diagnose at every time step
        DiagnoseEveryStep(n);
    }

    //final diagnosis
    DiagnoseAtEnd();

    cout << endl << "  Simulation Finish!" << endl;
}

void Simulation::ShiftAPeriod(double& x, double length)
{
    if(x < 0) x += length;
    else if(x >= length)  x -= length;
}

void Simulation::PrintParameters()
{
    //show information
    cout << "--------------------------------------------------------------------------------" << endl;
    cout << "  Vlasov Simulation: " << title << endl;
    cout << "  Parallelizing by " << omp_get_max_threads() << " threads" << endl;
    cout << "--------------------------------------------------------------------------------" << endl;
    cout << "  Boundary Condition: ";
#if defined _periodic_boundary_condition
    cout << "Periodic" << endl;
#elif defined _Dirichlet_boundary_condition
    cout << "Dirichlet" << endl;
#elif defined _Debye_boundary_condition
    cout << "Debye" << endl;
#endif
    cout << "--------------------------------------------------------------------------------" << endl;
    cout << "  Simulation Parameters: " << endl;
    cout.setf(ios::left);
    cout << "     L = " << setw(7) << setprecision(5) << L
         << "    nx = " << setw(7) << setprecision(5) << nx
         << "    dx = " << setw(7) << setprecision(5) << dx
         << "     k = " << setw(7) << setprecision(5) << k << endl;
    cout << "    wT = " << setw(7) << setprecision(5) << max_steps*dt
         << "    dt = " << setw(7) << setprecision(5) << dt
         << " steps = " << setw(7) << setprecision(5) << max_steps << endl;
    cout << "--------------------------------------------------------------------------------" << endl;
    cout << "  Plasma Parameters: " << endl;
    cout << "    me = " << setw(7) << setprecision(5) << me
         << "    Te = " << setw(7) << setprecision(5) << Te
         << "  vt_e = " << setw(7) << setprecision(5) << vt_e
         << "  wp_e = " << setw(7) << setprecision(5) << wp_e
         << "  l_De = " << setw(7) << setprecision(5) << l_e << endl;
    cout << "vmax_e = " << setw(7) << setprecision(5) << vmax_e
         << "  nv_e = " << setw(7) << setprecision(5) << nv_e
         << "  dv_e = " << setw(7) << setprecision(5) << dv_e << endl;
#if _ions_motion
    cout << "    mi = " << setw(7) << setprecision(5) << mi
         << "    Ti = " << setw(7) << setprecision(5) << Ti
         << "  vt_i = " << setw(7) << setprecision(5) << vt_i
         << "  wp_i = " << setw(7) << setprecision(5) << wp_i
         << "  l_Di = " << setw(7) << setprecision(5) << l_i << endl;
    cout << "vmax_i = " << setw(7) << setprecision(5) << vmax_i
         << "  nv_i = " << setw(7) << setprecision(5) << nv_i
         << "  dv_i = " << setw(7) << setprecision(5) << dv_i << endl;
#else
    cout << "    mi = " << setw(7) << setprecision(5) << "inf"
         << "    Ti = " << setw(7) << setprecision(5) << "-"
         << "  vt_i = " << setw(7) << setprecision(5) << "-"
         << "  w_pi = " << setw(7) << setprecision(5) << "-"
         << "  l_Di = " << setw(7) << setprecision(5) << "-" << endl;
    cout << "vmax_i = " << setw(7) << setprecision(5) << "-"
         << "  nv_i = " << setw(7) << setprecision(5) << "-"
         << "  dv_i = " << setw(7) << setprecision(5) << "-" << endl;
    cout << "  Noting: Ions is Immobile!" << endl;
#endif
    cout << "--------------------------------------------------------------------------------" << endl;

    //print special parameter for different experiments
    PrintSpecialParameters();

    cout << "--------------------------------------------------------------------------------" << endl;
    cout << "  Data: " << endl;
    cout << "  dataNum = " << setw(7) << setprecision(5) << data_num << endl;
    cout << "--------------------------------------------------------------------------------" << endl;
}
