#include<iostream>
#include<iomanip>
#include<fstream>
#include"diagnose.h"
#include"simulation.h"

using namespace std;

void OutputMatrix(string filename, MatrixXd f)
{
    ofstream ofile;
    ofile.open(filename.c_str());
    ofile << f;
    ofile.clear();
    ofile.close();
}
void OutputVector(string filename, vector < double > v)
{
    ofstream ofile;
    ofile.open(filename.c_str());
    for(auto out_v : v)
    {
        ofile << setprecision(13) << out_v << endl;
    }
    ofile.clear();
    ofile.close();
}

void Simulation::DiagnoseAtEnd()
{
    OutputVector(data_path + "kin_energy", Ek);
    // OutputVector(data_path + "kin_energy_e", Ek_e);
    // OutputVector(data_path + "kin_energy_i", Ek_i);
    OutputVector(data_path + "pot_energy", Ep);
    OutputVector(data_path + "tot_energy", Et);
    // OutputVector(data_path + "pe", pe);
    // OutputVector(data_path + "pi", pi);
    OutputMatrix(data_path + "Phi_evolution", Phi_evolution);
    // OutputMatrix(data_path + "E_evolution", E_evolution);
}

void Simulation::DiagnoseEveryStep(int n)
{
    //output distribution and phi
    if(n % data_steps == 0)
    {
        int nn = n / data_steps;
        string filename_e = data_path + "fe" + to_string(nn);
        OutputMatrix(filename_e, fe);
        string filename_i = data_path + "fi" + to_string(nn);
        OutputMatrix(filename_i, fi);
        // string filename = data_path + "phi" + to_string(nn);
        // OutputMatrix(filename, poisson_solver.phi);
        // string filename_EA = data_path + "EApplied" + to_string(nn);
        // OutputMatrix(filename_EA, E_Applied);
    }

    //diagnose energy & momentum
    double Ep_sum = 0.0;
    // double Ek_sum = 0.0;
    double Ek_e_sum = 0.0;
    double Ek_i_sum = 0.0;
    double pe_sum = 0.0;
    double pi_sum = 0.0;

    for(int i = 0; i < nx - 1; i++)
    {
        double E_temp = poisson_solver.GetE(i);
        Ep_sum += E_temp * E_temp;
        for(int j = 0; j < nv_e - 1; j++)
        {
            Ek_e_sum += pow(-vmax_e + j * dv_e, 2) * me * fe(i, j);
            pe_sum += me * (-vmax_e + j * dv_e) * fe(i, j);
        }
        // Ek_sum = Ek_e_sum;

#if _ions_motion
        for(int j = 0; j < nv_i - 1; j++)
        {
            Ek_i_sum += pow(-vmax_i + j * dv_i, 2) * (mi * fi(i, j));
            pi_sum += mi * (-vmax_i + j * dv_i) * fi(i, j);
        }
        // Ek_sum = Ek_e_sum + Ek_i_sum;
#endif
    }
    //average energy per particle
    Ep[n] = 0.5 * Ep_sum * dx / L;
    Ek_e[n] = 0.5 * Ek_e_sum * dx * dv_e / L;
    Ek_i[n] = 0.5 * Ek_i_sum * dx * dv_i / L;
    Ek[n] = Ek_e[n] + Ek_i[n];
    Et[n] = Ep[n] + Ek[n];

    pe[n] = pe_sum * dx * dv_e / L;
    pi[n] = pi_sum * dx * dv_i / L;

    Phi_evolution.col(n) = poisson_solver.phi;
    E_evolution.col(n) = poisson_solver.E;
}
