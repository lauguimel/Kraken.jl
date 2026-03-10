/**
# Lid-driven cavity Re=100
Benchmark case for comparison with Kraken.jl and Ghia et al. 1982.
Grid: 2^6 = 64 cells per dimension.
Domain: [0,1]^2 with top-lid moving at u=1.
*/

#include "navier-stokes/centered.h"

#define RE 100.
#define LEVEL 6

// Top lid velocity (tangential = 1)
u.t[top] = dirichlet(1.);
u.t[bottom] = dirichlet(0.);
u.t[left] = dirichlet(0.);
u.t[right] = dirichlet(0.);

// Normal velocity = 0 on all walls
u.n[top] = dirichlet(0.);
u.n[bottom] = dirichlet(0.);
u.n[left] = dirichlet(0.);
u.n[right] = dirichlet(0.);

scalar un[];

int main() {
    // Domain is [0,1]^2
    size(1.);
    origin(0., 0.);
    init_grid(1 << LEVEL);

    // Viscosity: 1/Re
    const face vector muc[] = {1./RE, 1./RE};
    mu = muc;

    run();
}

event init(t = 0) {
    // Zero initial velocity
    foreach() {
        u.x[] = 0.;
        u.y[] = 0.;
        un[] = 0.;
    }
}

// Track previous u.x for convergence check
event update_un(i++) {
    foreach()
        un[] = u.x[];
}

// Check convergence every 100 iterations
event logfile(i += 100; t <= 50.) {
    double du = change(u.x, un);
    fprintf(stderr, "i = %d, t = %g, du = %g\n", i, t, du);
    if (i > 0 && du < 1e-7)
        return 1; // converged, stop
}

// Output u-velocity profile along x=0.5 at the end
event profile(t = end) {
    FILE *fp = fopen("cavity_profile.dat", "w");
    fprintf(fp, "# y u\n");
    for (double y = 0; y <= 1.0; y += 1.0/(1 << LEVEL))
        fprintf(fp, "%.10g %.10g\n", y, interpolate(u.x, 0.5, y));
    fclose(fp);

    // Also output v along y=0.5
    FILE *fp2 = fopen("cavity_profile_v.dat", "w");
    fprintf(fp2, "# x v\n");
    for (double x = 0; x <= 1.0; x += 1.0/(1 << LEVEL))
        fprintf(fp2, "%.10g %.10g\n", x, interpolate(u.y, x, 0.5));
    fclose(fp2);
}
