import * as vscode from 'vscode';

const HOVER_DOCS: Record<string, string> = {
    // Physics parameters
    'nu': 'Kinematic viscosity (lattice units).\n\nRelated: τ = 3ν + 0.5, ω = 1/τ\n\nLow ν → high Re (turbulent). Keep τ ∈ [0.51, 2.0] for stability.',
    'alpha': 'Thermal diffusivity (lattice units).\n\nα = ν / Pr. Controls heat conduction rate.\n\nRequires `Module thermal`.',
    'Pr': 'Prandtl number.\n\nPr = ν/α. Ratio of momentum to thermal diffusivity.\n\nRequires `Module thermal`.\n\nTypical: air ≈ 0.71, water ≈ 7.0.',
    'Ra': 'Rayleigh number.\n\nRa = βg·ΔT·H³/(ν·α). Controls convection intensity.\n\nRa > Ra_c triggers convection onset. Requires `Module thermal`.',
    'Fx': 'Body force in x-direction (lattice units).\n\nUsed for pressure-driven flows (Poiseuille).',
    'Fy': 'Body force in y-direction (lattice units).\n\nUsed for gravity-driven flows.',
    'Fz': 'Body force in z-direction (lattice units).\n\n3D only.',
    'sigma': 'Surface tension coefficient (lattice units).\n\nControls interface curvature pressure. Requires `Module twophase_vof`.',
    'rho_l': 'Liquid density.\n\nUsed in two-phase simulations. Requires `Module twophase_vof`.',
    'rho_g': 'Gas density.\n\nUsed in two-phase simulations. Requires `Module twophase_vof`.',
    'beta_g': 'Thermal expansion coefficient.\n\nUsed in Boussinesq approximation: F = ρ·β·g·(T - T_ref).\n\nRequires `Module thermal`.',
    'g_x': 'Gravity component in x-direction.',
    'g_y': 'Gravity component in y-direction.\n\nTypically -1.0 for downward gravity.',
    'g_z': 'Gravity component in z-direction.',

    // Lattice types
    'D2Q9': '2D lattice with 9 velocities.\n\nStandard lattice for 2D LBM simulations. 1 rest + 4 cardinal + 4 diagonal.',
    'D3Q19': '3D lattice with 19 velocities.\n\nStandard lattice for 3D LBM simulations. 1 rest + 6 face + 12 edge.',

    // Boundary types
    'wall': 'No-slip wall boundary (bounce-back).\n\nZero velocity at the wall. Standard for solid boundaries.',
    'velocity': 'Velocity inlet/lid boundary.\n\nSyntax: `velocity(ux = ..., uy = ...)`\n\nSets a prescribed velocity at the boundary (Zou-He).',
    'pressure': 'Pressure boundary (Zou-He).\n\nSyntax: `pressure(rho = ...)`\n\nSets a prescribed density/pressure at the boundary.',
    'periodic': 'Periodic boundary condition.\n\nApplied to an axis (x, y, z). Wraps flow from one side to the other.',

    // Boundary faces
    'north': 'Top boundary face (y = L_y).',
    'south': 'Bottom boundary face (y = 0).',
    'east': 'Right boundary face (x = L_x).',
    'west': 'Left boundary face (x = 0).',
    'top': 'Top boundary face in 3D (z = L_z).',
    'bottom': 'Bottom boundary face in 3D (z = 0).',

    // Keywords
    'Simulation': 'Declare a new simulation.\n\nSyntax: `Simulation <name> <lattice>`\n\nExample: `Simulation cavity D2Q9`',
    'Domain': 'Define the simulation domain.\n\nSyntax: `Domain L = Lx x Ly  N = Nx x Ny`\n\nL = physical size, N = grid resolution.',
    'Physics': 'Physical parameters for the simulation.\n\nList parameters as `key = value` pairs.\n\nExample: `Physics nu = 0.01 Fx = 1e-5`',
    'Boundary': 'Set boundary condition on a face.\n\nSyntax: `Boundary <face> <type>`\n\nFaces: north, south, east, west, top, bottom.',
    'Obstacle': 'Define an obstacle using an implicit equation.\n\nSyntax: `Obstacle <name> { <equation> }`\n\nExample: `Obstacle cyl { (x-5)^2 + (y-2)^2 < 1^2 }`',
    'Refine': 'Define a grid refinement patch.\n\nSyntax: `Refine <name> { level = L x1 = ... y1 = ... x2 = ... y2 = ... }`\n\nNested refinement with factor 2^level.',
    'Initial': 'Set initial conditions.\n\nSyntax: `Initial { ux = expr  uy = expr }`\n\nSupports mathematical expressions with x, y, pi.',
    'Module': 'Enable a physics module.\n\nModules: `thermal`, `axisymmetric`, `twophase_vof`, `advection_only`.',
    'Run': 'Execute the simulation.\n\nSyntax: `Run <N> steps`',
    'Output': 'Configure output.\n\nSyntax: `Output <format> every <N> [fields]`\n\nFormats: vtk, png, gif. Fields: rho, ux, uy, uz, T.',
    'Diagnostics': 'Enable diagnostic outputs.\n\nMonitor convergence, forces, Nusselt number, etc.',
    'Rheology': 'Set rheology model for non-Newtonian fluids.\n\nModels: power_law, carreau, cross, bingham, herschel_bulkley.',
    'Setup': 'Helper for parameter computation.\n\nSyntax: `Setup reynolds Re = 100 L_ref = 128 u_ref = 0.1`\n\nComputes nu from Re, L_ref, u_ref.',
    'Preset': 'Use a predefined simulation configuration.\n\nPresets: cavity_2d, poiseuille_2d, couette_2d, taylor_green_2d, rayleigh_benard_2d.',
    'Sweep': 'Run a parameter sweep.\n\nSyntax: `Sweep <param> [val1, val2, ...]`\n\nRuns multiple simulations with different parameter values.',
    'Define': 'Define a reusable variable.\n\nSyntax: `Define <name> = <value>`',
    'Fluid': 'Fluid properties definition.',
    'Velocity': 'Velocity field specification.',

    // Modules
    'thermal': 'Thermal module (Boussinesq approximation).\n\nAdds temperature field and buoyancy force. Requires Pr or alpha in Physics.',
    'axisymmetric': 'Axisymmetric coordinate module.\n\nConverts 2D simulation to axisymmetric (cylindrical) coordinates.',
    'twophase_vof': 'Two-phase Volume of Fluid module.\n\nInterface tracking with VOF method. Requires sigma, rho_l, rho_g.',
    'advection_only': 'Passive scalar advection module.\n\nAdvects a scalar field without feedback on the flow.',

    // Presets
    'cavity_2d': 'Lid-driven cavity preset.\n\nClassic benchmark: square domain, top lid moves at constant velocity.',
    'poiseuille_2d': 'Poiseuille flow preset.\n\nChannel flow driven by body force between parallel walls.',
    'couette_2d': 'Couette flow preset.\n\nFlow between a moving top wall and a stationary bottom wall.',
    'taylor_green_2d': 'Taylor-Green vortex preset.\n\nDecaying vortex benchmark for accuracy testing.',
    'rayleigh_benard_2d': 'Rayleigh-Benard convection preset.\n\nThermal convection between heated bottom and cooled top plates.',

    // Output formats
    'vtk': 'VTK output format.\n\nCompatible with ParaView. Standard for CFD visualization.',
    'png': 'PNG image output.\n\nDirect field visualization as images.',
    'gif': 'Animated GIF output.\n\nCreates animation from field snapshots.',

    // Rheology models
    'power_law': 'Power-law rheology model.\n\nμ = K · γ̇^(n-1)\n\nParameters: K (consistency), n (power-law index).',
    'carreau': 'Carreau rheology model.\n\nμ = μ_∞ + (μ_0 - μ_∞)(1 + (λγ̇)²)^((n-1)/2)\n\nShear-thinning with Newtonian plateaus.',
    'cross': 'Cross rheology model.\n\nμ = μ_∞ + (μ_0 - μ_∞)/(1 + (λγ̇)^n)\n\nAlternative to Carreau.',
    'bingham': 'Bingham plastic model.\n\nτ = τ_y + μ_p · γ̇ for τ > τ_y\n\nYield stress fluid (e.g., concrete, toothpaste).',
    'herschel_bulkley': 'Herschel-Bulkley model.\n\nτ = τ_y + K · γ̇^n for τ > τ_y\n\nGeneralized yield stress + power-law.',

    // Common fields
    'rho': 'Density field.\n\nIn LBM: ρ = Σf_i. Incompressible limit: ρ ≈ 1.',
    'ux': 'Velocity component in x-direction.',
    'uy': 'Velocity component in y-direction.',
    'uz': 'Velocity component in z-direction (3D only).',
    'T': 'Temperature field.\n\nRequires `Module thermal`.',
    'steps': 'Number of time steps to simulate.',
    'every': 'Output frequency (in time steps).',
    'level': 'Refinement level.\n\nGrid spacing is divided by 2^level.',
};

export class KrkHoverProvider implements vscode.HoverProvider {
    provideHover(
        document: vscode.TextDocument,
        position: vscode.Position,
        _token: vscode.CancellationToken
    ): vscode.Hover | null {
        const wordRange = document.getWordRangeAtPosition(position, /[A-Za-z_][A-Za-z0-9_]*/);
        if (!wordRange) {
            return null;
        }

        const word = document.getText(wordRange);
        const doc = HOVER_DOCS[word];
        if (!doc) {
            return null;
        }

        const markdown = new vscode.MarkdownString();
        markdown.appendMarkdown(`**${word}**\n\n${doc}`);
        return new vscode.Hover(markdown, wordRange);
    }
}
