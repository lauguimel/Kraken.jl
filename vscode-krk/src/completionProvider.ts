import * as vscode from 'vscode';

const KEYWORDS = [
    { label: 'Simulation', detail: 'Declare simulation name and lattice', insertText: 'Simulation ${1:name} ${2|D2Q9,D3Q19|}' },
    { label: 'Domain', detail: 'Domain size and resolution', insertText: 'Domain  L = ${1:1.0} x ${2:1.0}  N = ${3:128} x ${4:128}' },
    { label: 'Physics', detail: 'Physical parameters', insertText: 'Physics ${1:nu = 0.01}' },
    { label: 'Boundary', detail: 'Boundary condition', insertText: 'Boundary ${1|north,south,east,west,top,bottom|} ${2|wall,velocity(ux = 0.0\\, uy = 0.0),pressure(rho = 1.0),periodic|}' },
    { label: 'Obstacle', detail: 'Obstacle definition', insertText: 'Obstacle ${1:name} { ${2:equation} }' },
    { label: 'Refine', detail: 'Refinement patch', insertText: 'Refine ${1:patch} { level = ${2:1} x1 = ${3:0.0} y1 = ${4:0.0} x2 = ${5:1.0} y2 = ${6:1.0} }' },
    { label: 'Initial', detail: 'Initial conditions block', insertText: 'Initial {\n  ${1:ux = 0.0}\n  ${2:uy = 0.0}\n}' },
    { label: 'Module', detail: 'Enable module', insertText: 'Module ${1|thermal,axisymmetric,twophase_vof,advection_only|}' },
    { label: 'Run', detail: 'Run simulation', insertText: 'Run ${1:10000} steps' },
    { label: 'Output', detail: 'Output configuration', insertText: 'Output ${1|vtk,png,gif|} every ${2:1000} [${3:rho, ux, uy}]' },
    { label: 'Diagnostics', detail: 'Diagnostics configuration', insertText: 'Diagnostics ${1:every 1000}' },
    { label: 'Rheology', detail: 'Rheology model', insertText: 'Rheology ${1|power_law,carreau,cross,bingham,herschel_bulkley|}' },
    { label: 'Setup', detail: 'Setup helper', insertText: 'Setup reynolds Re = ${1:100} L_ref = ${2:128} u_ref = ${3:0.1}' },
    { label: 'Preset', detail: 'Use a preset configuration', insertText: 'Preset ${1|cavity_2d,poiseuille_2d,couette_2d,taylor_green_2d,rayleigh_benard_2d|}' },
    { label: 'Sweep', detail: 'Parameter sweep', insertText: 'Sweep ${1:nu} [${2:0.01, 0.02, 0.05}]' },
    { label: 'Define', detail: 'Define a variable', insertText: 'Define ${1:name} = ${2:value}' },
    { label: 'Fluid', detail: 'Fluid properties', insertText: 'Fluid ${1:properties}' },
    { label: 'Velocity', detail: 'Velocity definition', insertText: 'Velocity ${1:specification}' },
];

const PHYSICS_PARAMS = [
    { label: 'nu', detail: 'Kinematic viscosity' },
    { label: 'alpha', detail: 'Thermal diffusivity' },
    { label: 'Pr', detail: 'Prandtl number' },
    { label: 'Ra', detail: 'Rayleigh number' },
    { label: 'Fx', detail: 'Body force (x)' },
    { label: 'Fy', detail: 'Body force (y)' },
    { label: 'Fz', detail: 'Body force (z)' },
    { label: 'sigma', detail: 'Surface tension' },
    { label: 'rho_l', detail: 'Liquid density' },
    { label: 'rho_g', detail: 'Gas density' },
    { label: 'beta_g', detail: 'Thermal expansion coefficient' },
    { label: 'g_x', detail: 'Gravity (x)' },
    { label: 'g_y', detail: 'Gravity (y)' },
    { label: 'g_z', detail: 'Gravity (z)' },
];

const BOUNDARY_TYPES = [
    { label: 'wall', detail: 'No-slip wall (bounce-back)' },
    { label: 'velocity', detail: 'Velocity boundary', insertText: 'velocity(ux = ${1:0.0}, uy = ${2:0.0})' },
    { label: 'pressure', detail: 'Pressure boundary', insertText: 'pressure(rho = ${1:1.0})' },
    { label: 'periodic', detail: 'Periodic boundary' },
];

const MODULES = [
    { label: 'thermal', detail: 'Thermal module (Boussinesq)' },
    { label: 'axisymmetric', detail: 'Axisymmetric coordinates' },
    { label: 'twophase_vof', detail: 'Two-phase VOF module' },
    { label: 'advection_only', detail: 'Passive scalar advection' },
];

const LATTICES = [
    { label: 'D2Q9', detail: '2D lattice, 9 velocities' },
    { label: 'D3Q19', detail: '3D lattice, 19 velocities' },
];

const OUTPUT_FORMATS = [
    { label: 'vtk', detail: 'VTK output format' },
    { label: 'png', detail: 'PNG image output' },
    { label: 'gif', detail: 'Animated GIF output' },
];

const PRESETS = [
    { label: 'cavity_2d', detail: 'Lid-driven cavity' },
    { label: 'poiseuille_2d', detail: 'Poiseuille channel flow' },
    { label: 'couette_2d', detail: 'Couette flow' },
    { label: 'taylor_green_2d', detail: 'Taylor-Green vortex' },
    { label: 'rayleigh_benard_2d', detail: 'Rayleigh-Benard convection' },
];

const RHEOLOGY_MODELS = [
    { label: 'power_law', detail: 'Power-law model' },
    { label: 'carreau', detail: 'Carreau model' },
    { label: 'cross', detail: 'Cross model' },
    { label: 'bingham', detail: 'Bingham plastic' },
    { label: 'herschel_bulkley', detail: 'Herschel-Bulkley model' },
];

const BOUNDARY_FACES = [
    { label: 'north', detail: 'Top boundary' },
    { label: 'south', detail: 'Bottom boundary' },
    { label: 'east', detail: 'Right boundary' },
    { label: 'west', detail: 'Left boundary' },
    { label: 'top', detail: 'Top boundary (3D)' },
    { label: 'bottom', detail: 'Bottom boundary (3D)' },
];

function makeItems(
    items: Array<{ label: string; detail: string; insertText?: string }>,
    kind: vscode.CompletionItemKind
): vscode.CompletionItem[] {
    return items.map(item => {
        const ci = new vscode.CompletionItem(item.label, kind);
        ci.detail = item.detail;
        if (item.insertText) {
            ci.insertText = new vscode.SnippetString(item.insertText);
        }
        return ci;
    });
}

export class KrkCompletionProvider implements vscode.CompletionItemProvider {
    provideCompletionItems(
        document: vscode.TextDocument,
        position: vscode.Position,
        _token: vscode.CancellationToken,
        _context: vscode.CompletionContext
    ): vscode.CompletionItem[] {
        const lineText = document.lineAt(position).text;
        const textBefore = lineText.substring(0, position.character).trimStart();

        // Empty line or just started typing: suggest keywords
        if (textBefore === '' || /^[A-Z]\w*$/.test(textBefore)) {
            return makeItems(KEYWORDS, vscode.CompletionItemKind.Keyword);
        }

        // After "Simulation <name> ": suggest lattices
        if (/^Simulation\s+\S+\s+/i.test(textBefore)) {
            return makeItems(LATTICES, vscode.CompletionItemKind.EnumMember);
        }

        // After "Physics ": suggest parameters
        if (/^Physics\s+/i.test(textBefore)) {
            return makeItems(PHYSICS_PARAMS, vscode.CompletionItemKind.Property);
        }

        // After "Boundary ": suggest faces first, then types
        if (/^Boundary\s+$/i.test(textBefore)) {
            return makeItems(BOUNDARY_FACES, vscode.CompletionItemKind.EnumMember);
        }

        // After "Boundary <face> ": suggest boundary types
        if (/^Boundary\s+\S+\s+/i.test(textBefore)) {
            return makeItems(BOUNDARY_TYPES, vscode.CompletionItemKind.Function);
        }

        // After "Module ": suggest modules
        if (/^Module\s+/i.test(textBefore)) {
            return makeItems(MODULES, vscode.CompletionItemKind.Module);
        }

        // After "Output ": suggest formats
        if (/^Output\s+$/i.test(textBefore)) {
            return makeItems(OUTPUT_FORMATS, vscode.CompletionItemKind.EnumMember);
        }

        // After "Preset ": suggest presets
        if (/^Preset\s+/i.test(textBefore)) {
            return makeItems(PRESETS, vscode.CompletionItemKind.Value);
        }

        // After "Rheology ": suggest models
        if (/^Rheology\s+/i.test(textBefore)) {
            return makeItems(RHEOLOGY_MODELS, vscode.CompletionItemKind.Value);
        }

        return [];
    }
}
