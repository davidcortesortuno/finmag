/**
 * FinMag - a thin layer on top of FEniCS to enable micromagnetic multi-physics simulations
 * Copyright (C) 2012 University of Southampton
 * Do not distribute
 *
 * CONTACT: h.fangohr@soton.ac.uk
 *
 * AUTHOR(S) OF THIS FILE:
    Dmitri Chernyshenko (d.chernyshenko@soton.ac.uk)
    Marc-Antonio Bisotti (mb8g11@soton.ac.uk)
 */

#include "finmag_includes.h"

#include "llg.h"

#include "util/python_threading.h"

namespace finmag { namespace llg {
    namespace {
        double const e = 1.602176565e-19; // elementary charge in As
        double const h_bar = 1.054571726e-34; // reduced Plank constant in Js
        double const pi = 4 * atan(1);
        double const mu_B = 9.27400968e-24; //Bohr magneton
        double const mu_0 = pi * 4e-7; // Vacuum permeability in Vs/(Am)

        // Components of the cross product
        inline double cross0(double a0, double a1, double a2, double b0, double b1, double b2) { return a1*b2 - a2*b1; }
        inline double cross1(double a0, double a1, double a2, double b0, double b1, double b2) { return a2*b0 - a0*b2; }
        inline double cross2(double a0, double a1, double a2, double b0, double b1, double b2) { return a0*b1 - a1*b0; }

        /*
        Compute the damping for one node.
        */
        void damping_i(
                double const alpha, double const gamma,
                double &m0, double &m1, double &m2,
                double &h0, double &h1, double &h2,
                double &dm0, double &dm1, double &dm2) {
            double const gamma_LL = gamma / (1 + alpha * alpha);
            double const damping_coeff = - alpha * gamma_LL;
            double const mh = m0 * h0 + m1 * h1 + m2 * h2;
            double const mm = m0 * m0 + m1 * m1 + m2 * m2;
            dm0 += damping_coeff * (m0 * mh - h0 * mm); 
            dm1 += damping_coeff * (m1 * mh - h1 * mm); 
            dm2 += damping_coeff * (m2 * mh - h2 * mm); 
        }

        /*
        Compute the derivative of the damping term for one node.
        m x (m x H) = (m*H)m - (m*m)H -->
            (mp*H + m*Hp)*m + (m*H)mp - 2(m*mp)H - (m*m)*Hp
        */
        void dm_damping_i(
                double const alpha, double const gamma,
                double const &m0, double const &m1, double const &m2,
                double const &mp0, double const &mp1, double const &mp2,
                double const &h0, double const &h1, double const &h2,
                double const &hp0, double const &hp1, double const &hp2,
                double &jtimes0, double &jtimes1, double &jtimes2) {
            double const gamma_LL = gamma / (1 + alpha * alpha);
            double const mph_mhp = mp0 * h0 + mp1 * h1 + mp2 * h2 + m0 * hp0 + m1 * hp1 + m2 * hp2;
            double const mh = m0 * h0 + m1 * h1 + m2 * h2;
            double const mm = m0 * m0 + m1 * m1 + m2 * m2;
            double const mmp = m0 * mp0 + m1 * mp1 + m2 * mp2;
            jtimes0 += - alpha * gamma_LL * (mph_mhp * m0 + mh * mp0 - 2 * mmp * h0 - mm * hp0);
            jtimes1 += - alpha * gamma_LL * (mph_mhp * m1 + mh * mp1 - 2 * mmp * h1 - mm * hp1);
            jtimes2 += - alpha * gamma_LL * (mph_mhp * m2 + mh * mp2 - 2 * mmp * h2 - mm * hp2);
        }

        /*
        Compute the precession for one node.
        */
        void precession_i(
                double const alpha, double const gamma,
                double &m0, double &m1, double &m2,
                double &h0, double &h1, double &h2,
                double &dm0, double &dm1, double &dm2) {
            double const gamma_LL = gamma / (1 + alpha * alpha);
            dm0 += - gamma_LL * cross0(m0, m1, m2, h0, h1, h2);
            dm1 += - gamma_LL * cross1(m0, m1, m2, h0, h1, h2);
            dm2 += - gamma_LL * cross2(m0, m1, m2, h0, h1, h2);
        }

        /*
        Derivative of the precessional term for one node.
        m x H --> m' x H + m x H'
        */
        void dm_precession_i(
                double const alpha, double const gamma,
                double const &m0, double const &m1, double const &m2,
                double const &mp0, double const &mp1, double const &mp2,
                double const &h0, double const &h1, double const &h2,
                double const &hp0, double const &hp1, double const &hp2,
                double &jtimes0, double &jtimes1, double &jtimes2) {
            double const gamma_LL = gamma / (1 + alpha * alpha);
            jtimes0 += - gamma_LL * (cross0(mp0, mp1, mp2, h0, h1, h2) + cross0(m0, m1, m2, hp0, hp1, hp2));
            jtimes1 += - gamma_LL * (cross1(mp0, mp1, mp2, h0, h1, h2) + cross1(m0, m1, m2, hp0, hp1, hp2));
            jtimes2 += - gamma_LL * (cross2(mp0, mp1, mp2, h0, h1, h2) + cross2(m0, m1, m2, hp0, hp1, hp2));
        }

        /*
        Compute the relaxation for one node.
        */
        void relaxation_i(
                double const c,
                double &m0, double &m1, double &m2,
                double &dm0, double &dm1, double &dm2) {
            double const mm = m0 * m0 + m1 * m1 + m2 * m2;
            double const relax_coeff = c * (1.0 - mm); 
            dm0 += relax_coeff * m0;
            dm1 += relax_coeff * m1;
            dm2 += relax_coeff * m2;
        }

        /*
        Compute the derivative of the relaxation term for one node.
        (1 - m*m) * m --> - 2 * m*mp * m + (1 - m*m) * mp
        */
        void dm_relaxation_i(
                double const relax_coeff,
                double const &m0, double const &m1, double const &m2,
                double const &mp0, double const &mp1, double const &mp2,
                double &jtimes0, double &jtimes1, double &jtimes2) {
            double const mm = m0 * m0 + m1 * m1 + m2 * m2;
            double const mmp = m0 * mp0 + m1 * mp1 + m2 * mp2;
            jtimes0 += relax_coeff * (-2 * mmp * m0 + (1.0 - mm) * mp0);
            jtimes1 += relax_coeff * (-2 * mmp * m1 + (1.0 - mm) * mp1);
            jtimes2 += relax_coeff * (-2 * mmp * m2 + (1.0 - mm) * mp2);
        }

        /*
        Set the values of dm/dt to zero for all nodes in pins.
        */
        void pin(const np_array<double> &dmdt, const np_array<long> &pins) {
            double *dm0 = dmdt(0), *dm1 = dmdt(1), *dm2 = dmdt(2);

            pins.check_ndim(1, "pins");
            int const nb_pins = pins.dim()[0];
            int const nodes = dmdt.dim()[1];

            int pin;
            for (int i = 0; i < nb_pins; i++) {
                pin = * pins[i];
                if ( pin >= 0 && pin < nodes ) {
                    dm0[pin] = 0;
                    dm1[pin] = 0;
                    dm2[pin] = 0;
                }
            }
        }

        /*
        Check if the dimensions of m, H, dmdt and alpha are mutually compatible.
        Returns the number of nodes.
        */
        int check_dimensions(
                const np_array<double> &alpha,
                const np_array<double> &m,
                const np_array<double> &H,
                const np_array<double> &dmdt) {
            m.check_ndim(2, "check_dimensions: m");
            int const nodes = m.dim()[1];

            m.check_shape(3, nodes, "check_dimensions: m");
            H.check_shape(3, nodes, "check_dimensions: H");
            dmdt.check_shape(3, nodes, "check_dimensions: dmdt");

            alpha.check_ndim(1, "check_dimensions: alpha");
            alpha.check_shape(nodes, "check_dimensions: alpha");

            return nodes;
        }

        void calc_llg_dmdt(
                const np_array<double> &m,
                const np_array<double> &H,
                double t,
                const np_array<double> &dmdt,
                const np_array<long> &pins,
                double gamma,
                const np_array<double> &alpha,
                double char_time,
                bool do_precession) {
            int const nodes = check_dimensions(alpha, m, H, dmdt); 
            double *m0 = m(0), *m1 = m(1), *m2 = m(2);
            double *h0 = H(0), *h1 = H(1), *h2 = H(2);
            double *dm0 = dmdt(0), *dm1 = dmdt(1), *dm2 = dmdt(2);

            finmag::util::scoped_gil_release release_gil;

            // calculate dmdt
            #pragma omp parallel for schedule(guided)
            for (int i=0; i < nodes; i++) {
                damping_i(*alpha[i], gamma, m0[i], m1[i], m2[i], h0[i], h1[i], h2[i], dm0[i], dm1[i], dm2[i]);
                relaxation_i(0.1/char_time, m0[i], m1[i], m2[i], dm0[i], dm1[i], dm2[i]);

                if (do_precession)
                    precession_i(*alpha[i], gamma, m0[i], m1[i], m2[i], h0[i], h1[i], h2[i], dm0[i], dm1[i], dm2[i]);
            }
            pin(dmdt, pins);
        }
    
        /*
         * Compute the Slonczewski/Xiao spin-torque term.
         */
        void slonczewski_xiao_i(
                double const alpha, double const gamma,
                double const lambda, double const epsilonprime,
                double const J, double const P, double const d, double const Ms,
                double const &m0, double const &m1, double const &m2,
                double const &p0, double const &p1, double const &p2,
                double &dm0, double &dm1, double &dm2) {
            double const gamma_LL = gamma / (1 + alpha * alpha);
            double lambda_sq = lambda * lambda;
            double const beta = J * h_bar / (mu_0 * Ms * e * d);
            double const epsilon = P * lambda_sq / (lambda_sq + 1 + (lambda_sq - 1) * (m0*p0 + m1*p1 + m2*p2));

            double const mm = m0 * m0 + m1 * m1 + m2 * m2; /* for the vector triple product expansion */
            double const mp = m0 * p0 + m1 * p1 + m2 * p2; /* also known as Lagrange's formula */

            /* gamma_LL * beta * {(alpha eps - eps') m x p - (eps - alpha eps') m x (m x p)} */
            double const perpendicular = alpha * epsilon - epsilonprime;
            double const parallel = epsilon - alpha * epsilonprime;
            dm0 += gamma_LL * beta * (perpendicular * cross0(m0, m1, m2, p0, p1, p2) - parallel * (mp * m0 - mm * p0));
            dm1 += gamma_LL * beta * (perpendicular * cross1(m0, m1, m2, p0, p1, p2) - parallel * (mp * m1 - mm * p1));
            dm2 += gamma_LL * beta * (perpendicular * cross2(m0, m1, m2, p0, p1, p2) - parallel * (mp * m2 - mm * p2));
        }

        /*
        Compute the Slonczewski spin-torque term for one node.
        */
        void slonczewski_i(
                double const alpha, double const gamma,
                double const J, double const P, double const d, double const Ms,
                double const &m0, double const &m1, double const &m2,
                double const &p0, double const &p1, double const &p2,
                double &dm0, double &dm1, double &dm2) {

            double const gamma_LL = gamma / (1 + alpha * alpha);
            double const mm = m0 * m0 + m1 * m1 + m2 * m2;
            double const mp = m0 * p0 + m1 * p1 + m2 * p2;

            double const a_P = 4 * pow(sqrt(P) / (1 + P), 3);
            double const stt_coeff = gamma_LL * J * h_bar / (mu_0 * Ms * e * d) * a_P / (3 + mp - 4 * a_P); 

            dm0 += stt_coeff * (alpha * cross0(m0, m1, m2, p0, p1, p2) - (mp * m0 - mm * p0));
            dm1 += stt_coeff * (alpha * cross1(m0, m1, m2, p0, p1, p2) - (mp * m1 - mm * p1));
            dm2 += stt_coeff * (alpha * cross2(m0, m1, m2, p0, p1, p2) - (mp * m2 - mm * p2));
        }

        void calc_llg_slonczewski_dmdt(
                const np_array<double> &m,
                const np_array<double> &H,
                double t,
                const np_array<double> &dmdt,
                const np_array<long> &pins,
                double gamma,
                const np_array<double> &alpha,
                double char_time,
                double const lambda,
                double const epsilonprime,
                const np_array<double> &J,
                double const P,
                double const d,
                const np_array<double> &Ms,
                const np_array<double> &p) {
            int const nodes = check_dimensions(alpha, m, H, dmdt);
            double *m0 = m(0), *m1 = m(1), *m2 = m(2);
            double *h0 = H(0), *h1 = H(1), *h2 = H(2);
            double *dm0 = dmdt(0), *dm1 = dmdt(1), *dm2 = dmdt(2);

            p.check_shape(3, nodes, "calc_llg_dmdt: p");
            double *p0 = p(0), *p1 = p(1), *p2 = p(2); 

            J.check_ndim(1, "calc_llg_slonczewski_dmdt: J");
            J.check_shape(nodes, "calc_llg_slonczewski_dmdt: J");

            Ms.check_ndim(1, "calc_llg_slonczewski_dmdt: Ms");
            Ms.check_shape(nodes, "calc_llg_slonczewski_dmdt: Ms");

            finmag::util::scoped_gil_release release_gil;

            // calculate dmdt
            #pragma omp parallel for schedule(guided)
            for (int i=0; i < nodes; i++) {
                damping_i(*alpha[i], gamma, m0[i], m1[i], m2[i], h0[i], h1[i], h2[i], dm0[i], dm1[i], dm2[i]);
                relaxation_i(0.1/char_time, m0[i], m1[i], m2[i], dm0[i], dm1[i], dm2[i]);
                precession_i(*alpha[i], gamma, m0[i], m1[i], m2[i], h0[i], h1[i], h2[i], dm0[i], dm1[i], dm2[i]);
                slonczewski_xiao_i(*alpha[i], gamma, lambda, epsilonprime, *J[i], P, d, *Ms[i], m0[i], m1[i], m2[i], p0[i], p1[i], p2[i], dm0[i], dm1[i], dm2[i]);
            }
            pin(dmdt, pins);
        }

        void calc_llg_jtimes(
                const np_array<double> &m,
                const np_array<double> &H,
                const np_array<double> &mp,
                const np_array<double> &Hp,
                double t,
                const np_array<double> &jtimes,
                double gamma,
                const np_array<double> &alpha,
                double char_time,
                bool do_precession,
                const np_array<long> &pins) {

           //* m - magnetisation
           //* H - effective field
           //* mp - the magnetisation vector to compute the product with (read it as m-prime, m')
           //* Hp - the derivative of the effective field in the direction of mp, i.e. d H_eff(m + a m')/da | a = 0
           //* jtimes - the output Jacobean product, i.e. d rhs(m + a m')/da | a = 0

            int nodes = check_dimensions(alpha, m, H, jtimes);
            mp.check_shape(3, nodes, "calc_llg_jtimes: mp");
            Hp.check_shape(3, nodes, "calc_llg_jtimes: Hp");

            double *m0 = m(0), *m1 = m(1), *m2 = m(2);
            double *mp0 = mp(0), *mp1 = mp(1), *mp2 = mp(2);
            double *h0 = H(0), *h1 = H(1), *h2 = H(2);
            double *hp0 = Hp(0), *hp1 = Hp(1), *hp2 = Hp(2);
            double *jtimes0 = jtimes(0), *jtimes1 = jtimes(1), *jtimes2 = jtimes(2);

            finmag::util::scoped_gil_release release_gil;

            #pragma omp parallel for schedule(guided)
            for (int i = 0; i < nodes; i++) {
                jtimes0[i] = 0; jtimes1[i] = 0; jtimes2[i] = 0;

                if ( do_precession ) {
                    dm_precession_i(*alpha[i], gamma, m0[i], m1[i], m2[i], mp0[i], mp1[i], mp2[i],
                        h0[i], h1[i], h2[i], hp0[i], hp1[i], hp2[i], jtimes0[i], jtimes1[i], jtimes2[i]);
                }
                dm_damping_i(*alpha[i], gamma, m0[i], m1[i], m2[i], mp0[i], mp1[i], mp2[i],
                        h0[i], h1[i], h2[i], hp0[i], hp1[i], hp2[i], jtimes0[i], jtimes1[i], jtimes2[i]);
                dm_relaxation_i(0.1/char_time, m0[i], m1[i], m2[i], mp0[i], mp1[i], mp2[i], jtimes0[i], jtimes1[i], jtimes2[i]);
            }
            pin(jtimes, pins);
        }

        /*
            Computes the solid angle subtended by the triangular mesh Ts, as seen from xs
              r - 3 x m array of points in space
              T - 3 x 3 x n array of triangular coordinates, first index is node number, second index is spatial coordinate
              a (output, optional) -  m vector of computed solid angles

            Return a, or a newly allocated result vector if a is not provided
        */
        np_array<double> compute_solid_angle(const np_array<double> &r_arr, const np_array<double> &T_arr, bp::object a_obj) {
            r_arr.check_ndim(2, "compute_solid_angle: r");
            T_arr.check_ndim(3, "compute_solid_angle: T");
            int m = r_arr.dim()[1];
            int n = T_arr.dim()[2];

            r_arr.check_shape(3, m, "compute_solid_angle: r");
            T_arr.check_shape(3, 3, n, "compute_solid_angle: T");

            np_array<double> a_arr = a_obj.is_none() ? np_array<double>(m) : bp::extract<np_array<double> >(a_obj);
            a_arr.check_shape(m, "compute_solid_angle: a");

            // set up the pointers
            double *r_x = r_arr(0), *r_y = r_arr(1), *r_z = r_arr(2);
            double *T1_x = T_arr(0, 0), *T1_y = T_arr(0, 1), *T1_z = T_arr(0, 2);
            double *T2_x = T_arr(1, 0), *T2_y = T_arr(1, 1), *T2_z = T_arr(1, 2);
            double *T3_x = T_arr(2, 0), *T3_y = T_arr(2, 1), *T3_z = T_arr(2, 2);
            double *a = a_arr.data();

            // i runs over points, j runs over triangles
    //        #pragma omp parallel for schedule(guided)
            for (int i = 0; i < m; i++) {
                double omega = 0;
                for (int j = 0; j < n; j++) {
                    double R1_x = T1_x[j] - r_x[i], R1_y = T1_y[j] - r_y[i], R1_z = T1_z[j] - r_z[i];
                    double R2_x = T2_x[j] - r_x[i], R2_y = T2_y[j] - r_y[i], R2_z = T2_z[j] - r_z[i];
                    double R3_x = T3_x[j] - r_x[i], R3_y = T3_y[j] - r_y[i], R3_z = T3_z[j] - r_z[i];
                    // Wikipedia cites
                    // Van Oosterom, A; Strackee, J (1983). "The Solid Angle of a Plane Triangle". IEEE Trans. Biom. Eng. BME-30 (2): 125–126. doi:10.1109/TBME.1983.325207
                    // Omega = 2*atan(p/q) where
                    // p = R1.R2xR3
                    double p = R1_x*(R2_y*R3_z - R2_z*R3_y) - R2_x*(R1_y*R3_z - R1_z*R3_y) + R3_x*(R1_y*R2_z - R1_z*R2_y);
                    // q = |R1||R2||R3| + |R3|R1.R2 + |R2|R1.R3 + |R1|R2.R3
                    double R1_norm = sqrt(R1_x*R1_x + R1_y*R1_y + R1_z*R1_z);
                    double R2_norm = sqrt(R2_x*R2_x + R2_y*R2_y + R2_z*R2_z);
                    double R3_norm = sqrt(R3_x*R3_x + R3_y*R3_y + R3_z*R3_z);
                    double R1_R2 = R1_x*R2_x + R1_y*R2_y + R1_z*R2_z;
                    double R1_R3 = R1_x*R3_x + R1_y*R3_y + R1_z*R3_z;
                    double R2_R3 = R2_x*R3_x + R2_y*R3_y + R2_z*R3_z;
                    double q = R1_norm*R2_norm*R3_norm + R3_norm*R1_R2 + R2_norm*R1_R3 + R1_norm*R2_R3;

                    double at = atan2(p, q);
                    omega += 2*at;
                }
                a[i] = omega;
            }

            return a_arr;
        }

        std::string demo_hello(std::string name) {
            std::stringstream ss;
            ss << "Hello" << name;
            return ss.str();
        }


        void calc_llg_zhang_li_dmdt(
                const np_array<double> &m,
                const np_array<double> &H,
                const np_array<double> &H_gradm,
                double t,
                const np_array<double> &dmdt,
                const np_array<long> &pins,
                double gamma,
                const np_array<double> &alpha_arr,
                double char_time,
                double const u0,
                double const beta,
                const np_array<double> &Ms) {

            int const nodes = check_dimensions(alpha_arr, m, H, dmdt);
            double *m0 = m(0), *m1 = m(1), *m2 = m(2);
            double *h0 = H(0), *h1 = H(1), *h2 = H(2);
            double *dm0 = dmdt(0), *dm1 = dmdt(1), *dm2 = dmdt(2);

            H_gradm.check_shape(3, nodes, "calc_llg_zhang_li_dmdt: H_gradm");
            double *hg0 = H_gradm(0), *hg1 = H_gradm(1), *hg2 = H_gradm(2);

            double *alpha = alpha_arr.data();

            Ms.check_ndim(1, "calc_llg_zhang_li_dmdt: Ms");
            Ms.check_shape(nodes, "calc_llg_zhang_li_dmdt: Ms");

            finmag::util::scoped_gil_release release_gil;

            // calculate dmdt
            #pragma omp parallel for schedule(guided)
            for (int i=0; i < nodes; i++) {

            	double coeff_stt = u0/(1+alpha[i]*alpha[i]);

            	if ((*Ms[i])==0){
            		coeff_stt = 0;
            	}else{
            		coeff_stt/=(*Ms[i]);
            	}

            	double mht = m0[i]*hg0[i] + m1[i]*hg1[i] + m2[i]*hg2[i];

            	double hp0 = hg0[i] - mht*m0[i];
            	double hp1 = hg1[i] - mht*m1[i];
            	double hp2 = hg2[i] - mht*m2[i];

            	double mth0 = cross0(m0[i],m1[i],m2[i],hp0,hp1,hp2);
            	double mth1 = cross1(m0[i],m1[i],m2[i],hp0,hp1,hp2);
            	double mth2 = cross2(m0[i],m1[i],m2[i],hp0,hp1,hp2);

            	dm0[i] = coeff_stt*((1 + alpha[i]*beta)*hp0 - (beta - alpha[i])*mth0);
            	dm1[i] = coeff_stt*((1 + alpha[i]*beta)*hp1 - (beta - alpha[i])*mth1);
            	dm2[i] = coeff_stt*((1 + alpha[i]*beta)*hp2 - (beta - alpha[i])*mth2);


            	damping_i(alpha[i], gamma, m0[i], m1[i], m2[i], h0[i], h1[i], h2[i], dm0[i], dm1[i], dm2[i]);
            	relaxation_i(0.1/char_time, m0[i], m1[i], m2[i], dm0[i], dm1[i], dm2[i]);
            	precession_i(alpha[i], gamma, m0[i], m1[i], m2[i], h0[i], h1[i], h2[i], dm0[i], dm1[i], dm2[i]);


            }
            pin(dmdt, pins);
        }


        void calc_llg_nonlocal_stt_dmdt(
                const np_array<double> &m,
                const np_array<double> &mt,
                const np_array<double> &H,
                const np_array<double> &H_diff,
                const np_array<double> &H_gradm,
                const np_array<double> &dmdt,
                const np_array<long> &pins,
                double const gamma,
                const np_array<double> &alpha_arr,
                double const char_time,
                double const P,
                double const tau_sd,
                double const tau_sf,
                const np_array<double> &Ms) {


            m.check_ndim(2, "check_dimensions: m");
            int const nodes = m.dim()[1];

            m.check_shape(3, nodes, "check_dimensions: m");
            mt.check_shape(3, nodes, "check_dimensions: m");
            H.check_shape(3, nodes, "check_dimensions: H");
            dmdt.check_shape(6, nodes, "check_dimensions: dmdt");
            H_gradm.check_shape(3, nodes, "check_dimensions: H_gradm");
            H_diff.check_shape(3, nodes, "check_dimensions: H_gradm");

            alpha_arr.check_ndim(1, "check_dimensions: alpha");
            alpha_arr.check_shape(nodes, "check_dimensions: alpha");


            double *m0 = m(0), *m1 = m(1), *m2 = m(2);
            double *mt0 = mt(0), *mt1 = mt(1), *mt2 = mt(2);
            double *h0 = H(0), *h1 = H(1), *h2 = H(2);
            double *dm0 = dmdt(0), *dm1 = dmdt(1), *dm2 = dmdt(2);
            double *dmt0 = dmdt(3), *dmt1 = dmdt(4), *dmt2 = dmdt(5);


            double *hg0 = H_gradm(0), *hg1 = H_gradm(1), *hg2 = H_gradm(2);
            double *hd0 = H_diff(0), *hd1 = H_diff(1), *hd2 = H_diff(2);

            double *alpha = alpha_arr.data();

            Ms.check_ndim(1, "calc_llg_zhang_li_dmdt: Ms");
            Ms.check_shape(nodes, "calc_llg_zhang_li_dmdt: Ms");

            finmag::util::scoped_gil_release release_gil;

            double u0 = P*mu_B/e; // P g mu_B/(2 e Ms) and g=2 for electrons

            // calculate dmdt
            #pragma omp parallel for schedule(guided)
            for (int i=0; i < nodes; i++) {

            	double coeff_stt =0 ;
            	double coeff = 1/(1+alpha[i]*alpha[i]) ;

            	if ((*Ms[i])>0){
            		coeff_stt=u0/(*Ms[i]);
            	}

            	double mmt = m0[i]*mt0[i] + m1[i]*mt1[i] + m2[i]*mt2[i];
            	double mtp0 = mt0[i]-mmt*m0[i];
            	double mtp1 = mt1[i]-mmt*m1[i];
            	double mtp2 = mt2[i]-mmt*m2[i];


            	double mmt0 = cross0(m0[i],m1[i],m2[i],mt0[i],mt1[i],mt2[i])/tau_sd;
            	double mmt1 = cross1(m0[i],m1[i],m2[i],mt0[i],mt1[i],mt2[i])/tau_sd;
            	double mmt2 = cross2(m0[i],m1[i],m2[i],mt0[i],mt1[i],mt2[i])/tau_sd;

            	//printf("%g  %g  %g\n",mt0[i],mt1[i],mt2[i]);
            	dm0[i] = (alpha[i]/tau_sd*mtp0-mmt0)*coeff;
            	dm1[i] = (alpha[i]/tau_sd*mtp1-mmt1)*coeff;
                dm2[i] = (alpha[i]/tau_sd*mtp2-mmt2)*coeff;

            	damping_i(alpha[i], gamma, m0[i], m1[i], m2[i], h0[i], h1[i], h2[i], dm0[i], dm1[i], dm2[i]);
            	relaxation_i(0.1/char_time, m0[i], m1[i], m2[i], dm0[i], dm1[i], dm2[i]);
            	precession_i(alpha[i], gamma, m0[i], m1[i], m2[i], h0[i], h1[i], h2[i], dm0[i], dm1[i], dm2[i]);

            	double mht = m0[i]*hg0[i] + m1[i]*hg1[i] + m2[i]*hg2[i];

            	double hp0 = hg0[i] - mht*m0[i];
            	double hp1 = hg1[i] - mht*m1[i];
            	double hp2 = hg2[i] - mht*m2[i];

            	dmt0[i] = hd0[i]+mmt0-mtp0/tau_sf+coeff_stt*hp0;
            	dmt1[i] = hd1[i]+mmt1-mtp1/tau_sf+coeff_stt*hp1;
            	dmt2[i] = hd2[i]+mmt2-mtp2/tau_sf+coeff_stt*hp2;

            }
            //Todo: fix the pin function here
            //pin(dmdt, pins);
        }



        /*
        Compute the Baryakhtar term for cubic crystal anisotropy
        and uniaxial crystal anisotropy, suppose nx=(1,0,0),
        type=1 corresponding to cubic and 0 for uniaxial.
        dM/dt = - gamma M x H  +  gamma M0 (alpha H - beta Delta H)
        */
        void calc_baryakhtar_dmdt(
                const np_array<double> &M,
                const np_array<double> &H,
                const np_array<double> &delta_H,
                const np_array<double> &dmdt,
                const np_array<double> &alpha,
                const np_array<double> &beta,
                double M0, 
                double gamma,
                double type,
                bool do_precession,
                const np_array<long> &pins) {
            
           
            finmag::util::scoped_gil_release release_gil;
            
            double *m=M.data();
            double *h=H.data();
            double *delta_h=delta_H.data();
    	    double *dm_dt=dmdt.data();
            double *a=alpha.data();
            double *b=beta.data();
            long int *pin=pins.data();
            
            double precession_coeff = -gamma;
            double damping_coeff = 0;
            damping_coeff = gamma * M0;
            
            assert(M.size()%3==0);
            
            int length=M.size()/3;         
   
            #pragma omp parallel for schedule(guided)
            for (int i = 0; i < length; i++) {
                int j=length+i;
                int k=length+j;
               
                if(pin[i]>0){
        		  dm_dt[i]=0;
        		  dm_dt[j]=0;
        		  dm_dt[k]=0;
                }{

                	dm_dt[i]=damping_coeff*(a[i]*h[i]*type - b[i]*delta_h[i]);
                	dm_dt[j]=damping_coeff*(a[i]*h[j] - b[i]*delta_h[j]);
                	dm_dt[k]=damping_coeff*(a[i]*h[k] - b[i]*delta_h[k]);
                
                	// add precession: m x H, multiplied by -gamma
                	if (do_precession) {
                		dm_dt[i] += precession_coeff*(m[j]*h[k]-m[k]*h[j]);
                		dm_dt[j] += precession_coeff*(m[k]*h[i]-m[i]*h[k]);
                		dm_dt[k] += precession_coeff*(m[i]*h[j]-m[j]*h[i]);
                } 

        	  }
            }

        }

        void baryakhtar_helper_M2(
                const np_array<double> &M,
                const np_array<double> &Ms){
        
            double *m=M.data();
            double *ms=Ms.data();
            
            assert(M.size()==Ms.size());
            assert(M.size()%3==0);
            
            int length=M.size()/3;         

            for (int i = 0; i < length; i++) {
                int j=length+i;
                int k=length+j;
                
                double tmp=m[i]*m[i]+m[j]*m[j]+m[k]*m[k];

                ms[i]=tmp;
                ms[j]=tmp;
                ms[k]=tmp;
            }
        
        }

	void calc_baryakhtar_jtimes(
		const np_array<double> &M,
                const np_array<double> &H,
                const np_array<double> &Mp,
                const np_array<double> &jtimes,
                double gamma,
                double chi,
                double M0,
                bool do_precession,
                const np_array<long> &pins) {

	    	double *m=M.data();
            double *h=H.data();
            double *mp=Mp.data();
            double *jt=jtimes.data();
            long int *pin=pins.data();

            double p[3][3],q[3][3];
            double tmp,m2;

            double coeff1=-gamma;
            double coeff2=-1.0/(chi*M0*M0);

            assert(H.size()%3==0);
            
            int length=H.size()/3;
            int i1,i2,i3;

           if (!do_precession) {
        	   for (int i = 0; i < 3*length; i++) {
                   jt[i]=0;
                }
        	   return;
           }


            for (int i = 0; i < length; i++) {
                i1=i;
                i2=length+i1;
                i3=length+i2;
 		
                m2=m[i1]*m[i1]+m[i2]*m[i2]+m[i3]*m[i3];
               
                tmp=(m2-M0*M0)/2.0;
                q[0][0]=coeff2*(m[i1]*m[i1]+tmp);
                q[0][1]=coeff2*m[i2]*m[i1];
                q[0][2]=coeff2*m[i3]*m[i1];

                q[1][0]=coeff2*m[i1]*m[i2];
                q[1][1]=coeff2*(m[i2]*m[i2]+tmp);
                q[1][2]=coeff2*m[i3]*m[i2];

                q[2][0]=coeff2*m[i1]*m[i3];
                q[2][1]=coeff2*m[i2]*m[i3];
                q[2][2]=coeff2*(m[i3]*m[i3]+tmp);



                p[0][0]=m[i2]*q[2][0]-m[i3]*q[1][0];
                p[0][1]=m[i2]*q[2][1]-m[i3]*q[1][1]+h[i3];
                p[0][2]=m[i2]*q[2][2]-m[i3]*q[1][2]-h[i2];

                p[1][0]=-m[i1]*q[2][0]+m[i3]*q[0][0]-h[i3];
                p[1][1]=-m[i1]*q[2][1]+m[i3]*q[0][1];
                p[1][2]=-m[i1]*q[2][2]+m[i3]*q[0][2]+h[i1];

                p[2][0]=m[i1]*q[1][0]-m[i2]*q[0][0]+h[i2];
                p[2][1]=m[i1]*q[1][1]-m[i2]*q[0][1]-h[i1];
                p[2][2]=m[i1]*q[1][2]-m[i2]*q[0][2];

	
                jt[i1]= coeff1*(p[0][0]*mp[i1]+p[0][1]*mp[i2]+p[0][2]*mp[i3]);
                jt[i2]= coeff1*(p[1][0]*mp[i1]+p[1][1]*mp[i2]+p[1][2]*mp[i3]);
                jt[i3]= coeff1*(p[2][0]*mp[i1]+p[2][1]*mp[i2]+p[2][2]*mp[i3]);

                //printf("%g  %g  %g\n",jt[i1],jt[i2],jt[i3]);

                if(pin[i]>0){
                	jt[i1]=0;
                    jt[i2]=0;
                    jt[i3]=0;
                }

            }
       }

    }

    void register_llg() {
        using namespace bp;

        def("calc_llg_dmdt", &calc_llg_dmdt, (
            arg("m"),
            arg("H"),
            arg("t"),
            arg("dmdt"),
            arg("pins"),
            arg("gamma_LL"),
            arg("alpha"),
            arg("char_time"),
            arg("do_precession")
        ));
        def("calc_llg_slonczewski_dmdt", &calc_llg_slonczewski_dmdt, (
            arg("m"),
            arg("H"),
            arg("t"),
            arg("dmdt"),
            arg("pins"),
            arg("gamma_LL"),
            arg("alpha"),
            arg("char_time"),
            arg("lambda"),
            arg("epsilonprime"),
            arg("J"),
            arg("P"),
            arg("d"),
            arg("Ms"),
            arg("p")
        ));
        def("calc_llg_jtimes", &calc_llg_jtimes, (
            arg("m"),
            arg("H"),
            arg("mp"),
            arg("Hp"),
            arg("t"),
            arg("jtimes"),
            arg("gamma_LL"),
            arg("alpha"),
            arg("char_time"),
            arg("do_precession"),
            arg("pins")
        ));
        def("compute_solid_angle", &compute_solid_angle, (
            arg("r"),
            arg("T"),
            arg("a")=object()
        ));
        //Towards some simple examples on using Boost.Python:
        def("demo_hello", &demo_hello, (
                arg("name")
                    ));

        def("calc_llg_zhang_li_dmdt", &calc_llg_zhang_li_dmdt, (
        		arg("m"),
        		arg("H"),
        		arg("H_gradm"),
        		arg("t"),
        		arg("dmdt"),
        		arg("pins"),
        		arg("gamma"),
        		arg("alpha_arr"),
        		arg("char_time"),
        		arg("u0"),
        		arg("beta"),
        		arg("Ms")
         ));


        def("calc_llg_nonlocal_stt_dmdt", &calc_llg_nonlocal_stt_dmdt, (
        		arg("m"),
        		arg("mt"),
        		arg("H"),
        		arg("H_diff"),
        		arg("H_gradm"),
        		arg("dmdt"),
        		arg("pins"),
        		arg("gamma"),
        		arg("alpha_arr"),
        		arg("char_time"),
        		arg("P"),
        		arg("tau_sd"),
        		arg("tau_sf"),
        		arg("Ms")
         ));


	   def("calc_baryakhtar_dmdt", &calc_baryakhtar_dmdt, (
            arg("M"),
            arg("H"),
	        arg("delta_H"),
            arg("dmdt"),
            arg("alpha"),
	        arg("beta"),
            arg("M0"),
            arg("gamma"),
            arg("type"),
            arg("do_precession"),
            arg("pins")
        ));


        
        def("baryakhtar_helper_M2", &baryakhtar_helper_M2, (
            arg("M"),
            arg("Ms")
        ));

        def("calc_baryakhtar_jtimes", &calc_baryakhtar_jtimes, (
        	arg("M"),
            arg("H"),
            arg("Mp"),
            arg("jtimes"),
            arg("gamma"),
            arg("chi"),
            arg("M0"),
            arg("do_precession"),
            arg("pins")
        ));
    }
}}
