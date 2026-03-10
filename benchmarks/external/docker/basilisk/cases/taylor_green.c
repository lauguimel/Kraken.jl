/**
# Taylor-Green vortex decay
Benchmark: periodic domain [0, 2*pi]^2, nu = 0.01, t_final = 1.0
Exact solution: u =  cos(x)*sin(y)*exp(-2*nu*t)
                v = -sin(x)*cos(y)*exp(-2*nu*t)
*/

#include "navier-stokes/centered.h"

#define NU 0.01
#define LEVEL 6
#define TFINAL 1.0

int main() {
    L0 = 2.*M_PI;
    origin(0., 0.);
    init_grid(1 << LEVEL);

    periodic(right);
    periodic(top);

    const face vector muc[] = {NU, NU};
    mu = muc;

    DT = 0.01;

    run();
}

event init(t = 0) {
    foreach() {
        u.x[] = cos(x)*sin(y);
        u.y[] = -sin(x)*cos(y);
    }
}

event logfile(i += 100; t <= TFINAL) {
    double decay = exp(-2.*NU*t);
    double errtot = 0., normtot = 0.;
    foreach(reduction(+:errtot) reduction(+:normtot)) {
        double ue = cos(x)*sin(y)*decay;
        double ve = -sin(x)*cos(y)*decay;
        errtot += sq(u.x[] - ue) + sq(u.y[] - ve);
        normtot += sq(ue) + sq(ve);
    }
    double denom = normtot > 0. ? normtot : 1.;
    fprintf(stderr, "t = %g, L2_error = %g\n", t, sqrt(errtot/denom));
}

event end_profile(t = end) {
    double decay = exp(-2.*NU*TFINAL);
    double umax = 0., errtot = 0., normtot = 0.;
    foreach(reduction(max:umax) reduction(+:errtot) reduction(+:normtot)) {
        double ue = cos(x)*sin(y)*decay;
        double ve = -sin(x)*cos(y)*decay;
        errtot += sq(u.x[] - ue) + sq(u.y[] - ve);
        normtot += sq(ue) + sq(ve);
        if (fabs(u.x[]) > umax) umax = fabs(u.x[]);
    }
    double denom = normtot > 0. ? normtot : 1.;

    FILE *fp = fopen("taylor_green_results.dat", "w");
    fprintf(fp, "# L2_error umax\n");
    fprintf(fp, "%.10g %.10g\n", sqrt(errtot/denom), umax);
    fclose(fp);
}
