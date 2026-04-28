"use strict";
var __create = Object.create;
var __defProp = Object.defineProperty;
var __getOwnPropDesc = Object.getOwnPropertyDescriptor;
var __getOwnPropNames = Object.getOwnPropertyNames;
var __getProtoOf = Object.getPrototypeOf;
var __hasOwnProp = Object.prototype.hasOwnProperty;
var __export = (target, all) => {
  for (var name in all)
    __defProp(target, name, { get: all[name], enumerable: true });
};
var __copyProps = (to, from, except, desc) => {
  if (from && typeof from === "object" || typeof from === "function") {
    for (let key of __getOwnPropNames(from))
      if (!__hasOwnProp.call(to, key) && key !== except)
        __defProp(to, key, { get: () => from[key], enumerable: !(desc = __getOwnPropDesc(from, key)) || desc.enumerable });
  }
  return to;
};
var __toESM = (mod, isNodeMode, target) => (target = mod != null ? __create(__getProtoOf(mod)) : {}, __copyProps(
  // If the importer is in node compatibility mode or this is not an ESM
  // file that has been converted to a CommonJS file using a Babel-
  // compatible transform (i.e. "__esModule" has not been set), then set
  // "default" to the CommonJS "module.exports" for node compatibility.
  isNodeMode || !mod || !mod.__esModule ? __defProp(target, "default", { value: mod, enumerable: true }) : target,
  mod
));
var __toCommonJS = (mod) => __copyProps(__defProp({}, "__esModule", { value: true }), mod);

// src/extension.ts
var extension_exports = {};
__export(extension_exports, {
  activate: () => activate,
  deactivate: () => deactivate
});
module.exports = __toCommonJS(extension_exports);
var vscode3 = __toESM(require("vscode"));

// src/validator.ts
var import_vscode = require("vscode");
var TOP_LEVEL_KEYWORDS = [
  "Simulation",
  "Domain",
  "Physics",
  "Define",
  "Obstacle",
  "Fluid",
  "Boundary",
  "Refine",
  "Initial",
  "Velocity",
  "Module",
  "Run",
  "Output",
  "Diagnostics",
  "Rheology",
  "Setup",
  "Preset",
  "Sweep"
];
function levenshtein(a, b) {
  const m = a.length;
  const n = b.length;
  const dp = Array.from({ length: m + 1 }, () => Array(n + 1).fill(0));
  for (let i = 0; i <= m; i++) {
    dp[i][0] = i;
  }
  for (let j = 0; j <= n; j++) {
    dp[0][j] = j;
  }
  for (let i = 1; i <= m; i++) {
    for (let j = 1; j <= n; j++) {
      dp[i][j] = a[i - 1] === b[j - 1] ? dp[i - 1][j - 1] : 1 + Math.min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]);
    }
  }
  return dp[m][n];
}
function suggestKeyword(word) {
  let best = "";
  let bestDist = Infinity;
  for (const kw of TOP_LEVEL_KEYWORDS) {
    const d = levenshtein(word.toLowerCase(), kw.toLowerCase());
    if (d < bestDist) {
      bestDist = d;
      best = kw;
    }
  }
  return bestDist <= 3 ? best : null;
}
function validateDocument(text) {
  const diagnostics = [];
  const lines = text.split("\n");
  let hasSimulation = false;
  let hasPreset = false;
  let hasDomain = false;
  let hasRun = false;
  let hasThermalModule = false;
  let hasPr = false;
  let hasAlpha = false;
  let nuValue = null;
  let nuLine = -1;
  let insideBlock = false;
  let braceDepth = 0;
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    const trimmed = line.trim();
    for (const ch of line) {
      if (ch === "{") {
        braceDepth++;
        insideBlock = true;
      }
      if (ch === "}") {
        braceDepth--;
        if (braceDepth <= 0) {
          insideBlock = false;
          braceDepth = 0;
        }
      }
    }
    if (trimmed === "" || trimmed.startsWith("#")) {
      continue;
    }
    if (insideBlock && !trimmed.match(/^\s*(Simulation|Domain|Physics|Define|Obstacle|Fluid|Boundary|Refine|Initial|Velocity|Module|Run|Output|Diagnostics|Rheology|Setup|Preset|Sweep)\b/)) {
      continue;
    }
    const firstWordMatch = trimmed.match(/^([A-Za-z_]\w*)\b/);
    if (!firstWordMatch) {
      continue;
    }
    const firstWord = firstWordMatch[1];
    if (!TOP_LEVEL_KEYWORDS.includes(firstWord)) {
      if (firstWord[0] === firstWord[0].toUpperCase() && firstWord.length > 2) {
        const suggestion = suggestKeyword(firstWord);
        const msg = suggestion ? `Unknown keyword "${firstWord}". Did you mean "${suggestion}"?` : `Unknown keyword "${firstWord}".`;
        diagnostics.push({
          line: i,
          endCol: firstWord.length,
          message: msg,
          severity: import_vscode.DiagnosticSeverity.Error,
          source: "krk"
        });
      }
      continue;
    }
    if (firstWord === "Simulation") {
      hasSimulation = true;
    }
    if (firstWord === "Preset") {
      hasPreset = true;
    }
    if (firstWord === "Domain") {
      hasDomain = true;
    }
    if (firstWord === "Run") {
      hasRun = true;
    }
    if (firstWord === "Module" && /\bthermal\b/.test(trimmed)) {
      hasThermalModule = true;
    }
    if (firstWord === "Physics") {
      const nuMatch = trimmed.match(/\bnu\s*=\s*([0-9eE.+-]+)/);
      if (nuMatch) {
        nuValue = parseFloat(nuMatch[1]);
        nuLine = i;
        if (isNaN(nuValue) || nuValue <= 0) {
          diagnostics.push({
            line: i,
            message: `nu must be positive (got ${nuMatch[1]}).`,
            severity: import_vscode.DiagnosticSeverity.Error,
            source: "krk"
          });
        }
      }
      if (/\bPr\s*=/.test(trimmed)) {
        hasPr = true;
      }
      if (/\balpha\s*=/.test(trimmed)) {
        hasAlpha = true;
      }
    }
  }
  if (!hasSimulation && !hasPreset) {
    diagnostics.push({
      line: 0,
      message: 'Missing "Simulation" or "Preset" declaration.',
      severity: import_vscode.DiagnosticSeverity.Error,
      source: "krk"
    });
  }
  if (hasSimulation && !hasPreset && !hasDomain) {
    diagnostics.push({
      line: 0,
      message: 'Missing "Domain" definition. Required when using Simulation.',
      severity: import_vscode.DiagnosticSeverity.Error,
      source: "krk"
    });
  }
  if (!hasRun) {
    diagnostics.push({
      line: lines.length - 1,
      message: 'Missing "Run" command.',
      severity: import_vscode.DiagnosticSeverity.Warning,
      source: "krk"
    });
  }
  if (nuValue !== null && nuValue > 0 && nuLine >= 0) {
    const tau = 3 * nuValue + 0.5;
    if (tau < 0.51) {
      diagnostics.push({
        line: nuLine,
        message: `Stability warning: \u03C4 = ${tau.toFixed(4)} is very close to 0.5. Simulation may be unstable.`,
        severity: import_vscode.DiagnosticSeverity.Warning,
        source: "krk-stability"
      });
    }
    if (tau > 2) {
      diagnostics.push({
        line: nuLine,
        message: `Stability warning: \u03C4 = ${tau.toFixed(4)} > 2.0. High viscosity may cause inaccuracies.`,
        severity: import_vscode.DiagnosticSeverity.Warning,
        source: "krk-stability"
      });
    }
  }
  if (hasThermalModule && !hasPr && !hasAlpha) {
    diagnostics.push({
      line: 0,
      message: 'Module thermal requires "Pr" or "alpha" in Physics.',
      severity: import_vscode.DiagnosticSeverity.Warning,
      source: "krk"
    });
  }
  return diagnostics;
}

// src/completionProvider.ts
var vscode = __toESM(require("vscode"));
var KEYWORDS = [
  { label: "Simulation", detail: "Declare simulation name and lattice", insertText: "Simulation ${1:name} ${2|D2Q9,D3Q19|}" },
  { label: "Domain", detail: "Domain size and resolution", insertText: "Domain  L = ${1:1.0} x ${2:1.0}  N = ${3:128} x ${4:128}" },
  { label: "Physics", detail: "Physical parameters", insertText: "Physics ${1:nu = 0.01}" },
  { label: "Boundary", detail: "Boundary condition", insertText: "Boundary ${1|north,south,east,west,top,bottom|} ${2|wall,velocity(ux = 0.0\\, uy = 0.0),pressure(rho = 1.0),periodic|}" },
  { label: "Obstacle", detail: "Obstacle definition", insertText: "Obstacle ${1:name} { ${2:equation} }" },
  { label: "Refine", detail: "Refinement patch", insertText: "Refine ${1:patch} { level = ${2:1} x1 = ${3:0.0} y1 = ${4:0.0} x2 = ${5:1.0} y2 = ${6:1.0} }" },
  { label: "Initial", detail: "Initial conditions block", insertText: "Initial {\n  ${1:ux = 0.0}\n  ${2:uy = 0.0}\n}" },
  { label: "Module", detail: "Enable module", insertText: "Module ${1|thermal,axisymmetric,twophase_vof,advection_only|}" },
  { label: "Run", detail: "Run simulation", insertText: "Run ${1:10000} steps" },
  { label: "Output", detail: "Output configuration", insertText: "Output ${1|vtk,png,gif|} every ${2:1000} [${3:rho, ux, uy}]" },
  { label: "Diagnostics", detail: "Diagnostics configuration", insertText: "Diagnostics ${1:every 1000}" },
  { label: "Rheology", detail: "Rheology model", insertText: "Rheology ${1|power_law,carreau,cross,bingham,herschel_bulkley|}" },
  { label: "Setup", detail: "Setup helper", insertText: "Setup reynolds Re = ${1:100} L_ref = ${2:128} u_ref = ${3:0.1}" },
  { label: "Preset", detail: "Use a preset configuration", insertText: "Preset ${1|cavity_2d,poiseuille_2d,couette_2d,taylor_green_2d,rayleigh_benard_2d|}" },
  { label: "Sweep", detail: "Parameter sweep", insertText: "Sweep ${1:nu} [${2:0.01, 0.02, 0.05}]" },
  { label: "Define", detail: "Define a variable", insertText: "Define ${1:name} = ${2:value}" },
  { label: "Fluid", detail: "Fluid properties", insertText: "Fluid ${1:properties}" },
  { label: "Velocity", detail: "Velocity definition", insertText: "Velocity ${1:specification}" }
];
var PHYSICS_PARAMS = [
  { label: "nu", detail: "Kinematic viscosity" },
  { label: "alpha", detail: "Thermal diffusivity" },
  { label: "Pr", detail: "Prandtl number" },
  { label: "Ra", detail: "Rayleigh number" },
  { label: "Fx", detail: "Body force (x)" },
  { label: "Fy", detail: "Body force (y)" },
  { label: "Fz", detail: "Body force (z)" },
  { label: "sigma", detail: "Surface tension" },
  { label: "rho_l", detail: "Liquid density" },
  { label: "rho_g", detail: "Gas density" },
  { label: "beta_g", detail: "Thermal expansion coefficient" },
  { label: "g_x", detail: "Gravity (x)" },
  { label: "g_y", detail: "Gravity (y)" },
  { label: "g_z", detail: "Gravity (z)" }
];
var BOUNDARY_TYPES = [
  { label: "wall", detail: "No-slip wall (bounce-back)" },
  { label: "velocity", detail: "Velocity boundary", insertText: "velocity(ux = ${1:0.0}, uy = ${2:0.0})" },
  { label: "pressure", detail: "Pressure boundary", insertText: "pressure(rho = ${1:1.0})" },
  { label: "periodic", detail: "Periodic boundary" }
];
var MODULES = [
  { label: "thermal", detail: "Thermal module (Boussinesq)" },
  { label: "axisymmetric", detail: "Axisymmetric coordinates" },
  { label: "twophase_vof", detail: "Two-phase VOF module" },
  { label: "advection_only", detail: "Passive scalar advection" }
];
var LATTICES = [
  { label: "D2Q9", detail: "2D lattice, 9 velocities" },
  { label: "D3Q19", detail: "3D lattice, 19 velocities" }
];
var OUTPUT_FORMATS = [
  { label: "vtk", detail: "VTK output format" },
  { label: "png", detail: "PNG image output" },
  { label: "gif", detail: "Animated GIF output" }
];
var PRESETS = [
  { label: "cavity_2d", detail: "Lid-driven cavity" },
  { label: "poiseuille_2d", detail: "Poiseuille channel flow" },
  { label: "couette_2d", detail: "Couette flow" },
  { label: "taylor_green_2d", detail: "Taylor-Green vortex" },
  { label: "rayleigh_benard_2d", detail: "Rayleigh-Benard convection" }
];
var RHEOLOGY_MODELS = [
  { label: "power_law", detail: "Power-law model" },
  { label: "carreau", detail: "Carreau model" },
  { label: "cross", detail: "Cross model" },
  { label: "bingham", detail: "Bingham plastic" },
  { label: "herschel_bulkley", detail: "Herschel-Bulkley model" }
];
var BOUNDARY_FACES = [
  { label: "north", detail: "Top boundary" },
  { label: "south", detail: "Bottom boundary" },
  { label: "east", detail: "Right boundary" },
  { label: "west", detail: "Left boundary" },
  { label: "top", detail: "Top boundary (3D)" },
  { label: "bottom", detail: "Bottom boundary (3D)" }
];
function makeItems(items, kind) {
  return items.map((item) => {
    const ci = new vscode.CompletionItem(item.label, kind);
    ci.detail = item.detail;
    if (item.insertText) {
      ci.insertText = new vscode.SnippetString(item.insertText);
    }
    return ci;
  });
}
var KrkCompletionProvider = class {
  provideCompletionItems(document, position, _token, _context) {
    const lineText = document.lineAt(position).text;
    const textBefore = lineText.substring(0, position.character).trimStart();
    if (textBefore === "" || /^[A-Z]\w*$/.test(textBefore)) {
      return makeItems(KEYWORDS, vscode.CompletionItemKind.Keyword);
    }
    if (/^Simulation\s+\S+\s+/i.test(textBefore)) {
      return makeItems(LATTICES, vscode.CompletionItemKind.EnumMember);
    }
    if (/^Physics\s+/i.test(textBefore)) {
      return makeItems(PHYSICS_PARAMS, vscode.CompletionItemKind.Property);
    }
    if (/^Boundary\s+$/i.test(textBefore)) {
      return makeItems(BOUNDARY_FACES, vscode.CompletionItemKind.EnumMember);
    }
    if (/^Boundary\s+\S+\s+/i.test(textBefore)) {
      return makeItems(BOUNDARY_TYPES, vscode.CompletionItemKind.Function);
    }
    if (/^Module\s+/i.test(textBefore)) {
      return makeItems(MODULES, vscode.CompletionItemKind.Module);
    }
    if (/^Output\s+$/i.test(textBefore)) {
      return makeItems(OUTPUT_FORMATS, vscode.CompletionItemKind.EnumMember);
    }
    if (/^Preset\s+/i.test(textBefore)) {
      return makeItems(PRESETS, vscode.CompletionItemKind.Value);
    }
    if (/^Rheology\s+/i.test(textBefore)) {
      return makeItems(RHEOLOGY_MODELS, vscode.CompletionItemKind.Value);
    }
    return [];
  }
};

// src/hoverProvider.ts
var vscode2 = __toESM(require("vscode"));
var HOVER_DOCS = {
  // Physics parameters
  "nu": "Kinematic viscosity (lattice units).\n\nRelated: \u03C4 = 3\u03BD + 0.5, \u03C9 = 1/\u03C4\n\nLow \u03BD \u2192 high Re (turbulent). Keep \u03C4 \u2208 [0.51, 2.0] for stability.",
  "alpha": "Thermal diffusivity (lattice units).\n\n\u03B1 = \u03BD / Pr. Controls heat conduction rate.\n\nRequires `Module thermal`.",
  "Pr": "Prandtl number.\n\nPr = \u03BD/\u03B1. Ratio of momentum to thermal diffusivity.\n\nRequires `Module thermal`.\n\nTypical: air \u2248 0.71, water \u2248 7.0.",
  "Ra": "Rayleigh number.\n\nRa = \u03B2g\xB7\u0394T\xB7H\xB3/(\u03BD\xB7\u03B1). Controls convection intensity.\n\nRa > Ra_c triggers convection onset. Requires `Module thermal`.",
  "Fx": "Body force in x-direction (lattice units).\n\nUsed for pressure-driven flows (Poiseuille).",
  "Fy": "Body force in y-direction (lattice units).\n\nUsed for gravity-driven flows.",
  "Fz": "Body force in z-direction (lattice units).\n\n3D only.",
  "sigma": "Surface tension coefficient (lattice units).\n\nControls interface curvature pressure. Requires `Module twophase_vof`.",
  "rho_l": "Liquid density.\n\nUsed in two-phase simulations. Requires `Module twophase_vof`.",
  "rho_g": "Gas density.\n\nUsed in two-phase simulations. Requires `Module twophase_vof`.",
  "beta_g": "Thermal expansion coefficient.\n\nUsed in Boussinesq approximation: F = \u03C1\xB7\u03B2\xB7g\xB7(T - T_ref).\n\nRequires `Module thermal`.",
  "g_x": "Gravity component in x-direction.",
  "g_y": "Gravity component in y-direction.\n\nTypically -1.0 for downward gravity.",
  "g_z": "Gravity component in z-direction.",
  // Lattice types
  "D2Q9": "2D lattice with 9 velocities.\n\nStandard lattice for 2D LBM simulations. 1 rest + 4 cardinal + 4 diagonal.",
  "D3Q19": "3D lattice with 19 velocities.\n\nStandard lattice for 3D LBM simulations. 1 rest + 6 face + 12 edge.",
  // Boundary types
  "wall": "No-slip wall boundary (bounce-back).\n\nZero velocity at the wall. Standard for solid boundaries.",
  "velocity": "Velocity inlet/lid boundary.\n\nSyntax: `velocity(ux = ..., uy = ...)`\n\nSets a prescribed velocity at the boundary (Zou-He).",
  "pressure": "Pressure boundary (Zou-He).\n\nSyntax: `pressure(rho = ...)`\n\nSets a prescribed density/pressure at the boundary.",
  "periodic": "Periodic boundary condition.\n\nApplied to an axis (x, y, z). Wraps flow from one side to the other.",
  // Boundary faces
  "north": "Top boundary face (y = L_y).",
  "south": "Bottom boundary face (y = 0).",
  "east": "Right boundary face (x = L_x).",
  "west": "Left boundary face (x = 0).",
  "top": "Top boundary face in 3D (z = L_z).",
  "bottom": "Bottom boundary face in 3D (z = 0).",
  // Keywords
  "Simulation": "Declare a new simulation.\n\nSyntax: `Simulation <name> <lattice>`\n\nExample: `Simulation cavity D2Q9`",
  "Domain": "Define the simulation domain.\n\nSyntax: `Domain L = Lx x Ly  N = Nx x Ny`\n\nL = physical size, N = grid resolution.",
  "Physics": "Physical parameters for the simulation.\n\nList parameters as `key = value` pairs.\n\nExample: `Physics nu = 0.01 Fx = 1e-5`",
  "Boundary": "Set boundary condition on a face.\n\nSyntax: `Boundary <face> <type>`\n\nFaces: north, south, east, west, top, bottom.",
  "Obstacle": "Define an obstacle using an implicit equation.\n\nSyntax: `Obstacle <name> { <equation> }`\n\nExample: `Obstacle cyl { (x-5)^2 + (y-2)^2 < 1^2 }`",
  "Refine": "Define a grid refinement patch.\n\nSyntax: `Refine <name> { level = L x1 = ... y1 = ... x2 = ... y2 = ... }`\n\nNested refinement with factor 2^level.",
  "Initial": "Set initial conditions.\n\nSyntax: `Initial { ux = expr  uy = expr }`\n\nSupports mathematical expressions with x, y, pi.",
  "Module": "Enable a physics module.\n\nModules: `thermal`, `axisymmetric`, `twophase_vof`, `advection_only`.",
  "Run": "Execute the simulation.\n\nSyntax: `Run <N> steps`",
  "Output": "Configure output.\n\nSyntax: `Output <format> every <N> [fields]`\n\nFormats: vtk, png, gif. Fields: rho, ux, uy, uz, T.",
  "Diagnostics": "Enable diagnostic outputs.\n\nMonitor convergence, forces, Nusselt number, etc.",
  "Rheology": "Set rheology model for non-Newtonian fluids.\n\nModels: power_law, carreau, cross, bingham, herschel_bulkley.",
  "Setup": "Helper for parameter computation.\n\nSyntax: `Setup reynolds Re = 100 L_ref = 128 u_ref = 0.1`\n\nComputes nu from Re, L_ref, u_ref.",
  "Preset": "Use a predefined simulation configuration.\n\nPresets: cavity_2d, poiseuille_2d, couette_2d, taylor_green_2d, rayleigh_benard_2d.",
  "Sweep": "Run a parameter sweep.\n\nSyntax: `Sweep <param> [val1, val2, ...]`\n\nRuns multiple simulations with different parameter values.",
  "Define": "Define a reusable variable.\n\nSyntax: `Define <name> = <value>`",
  "Fluid": "Fluid properties definition.",
  "Velocity": "Velocity field specification.",
  // Modules
  "thermal": "Thermal module (Boussinesq approximation).\n\nAdds temperature field and buoyancy force. Requires Pr or alpha in Physics.",
  "axisymmetric": "Axisymmetric coordinate module.\n\nConverts 2D simulation to axisymmetric (cylindrical) coordinates.",
  "twophase_vof": "Two-phase Volume of Fluid module.\n\nInterface tracking with VOF method. Requires sigma, rho_l, rho_g.",
  "advection_only": "Passive scalar advection module.\n\nAdvects a scalar field without feedback on the flow.",
  // Presets
  "cavity_2d": "Lid-driven cavity preset.\n\nClassic benchmark: square domain, top lid moves at constant velocity.",
  "poiseuille_2d": "Poiseuille flow preset.\n\nChannel flow driven by body force between parallel walls.",
  "couette_2d": "Couette flow preset.\n\nFlow between a moving top wall and a stationary bottom wall.",
  "taylor_green_2d": "Taylor-Green vortex preset.\n\nDecaying vortex benchmark for accuracy testing.",
  "rayleigh_benard_2d": "Rayleigh-Benard convection preset.\n\nThermal convection between heated bottom and cooled top plates.",
  // Output formats
  "vtk": "VTK output format.\n\nCompatible with ParaView. Standard for CFD visualization.",
  "png": "PNG image output.\n\nDirect field visualization as images.",
  "gif": "Animated GIF output.\n\nCreates animation from field snapshots.",
  // Rheology models
  "power_law": "Power-law rheology model.\n\n\u03BC = K \xB7 \u03B3\u0307^(n-1)\n\nParameters: K (consistency), n (power-law index).",
  "carreau": "Carreau rheology model.\n\n\u03BC = \u03BC_\u221E + (\u03BC_0 - \u03BC_\u221E)(1 + (\u03BB\u03B3\u0307)\xB2)^((n-1)/2)\n\nShear-thinning with Newtonian plateaus.",
  "cross": "Cross rheology model.\n\n\u03BC = \u03BC_\u221E + (\u03BC_0 - \u03BC_\u221E)/(1 + (\u03BB\u03B3\u0307)^n)\n\nAlternative to Carreau.",
  "bingham": "Bingham plastic model.\n\n\u03C4 = \u03C4_y + \u03BC_p \xB7 \u03B3\u0307 for \u03C4 > \u03C4_y\n\nYield stress fluid (e.g., concrete, toothpaste).",
  "herschel_bulkley": "Herschel-Bulkley model.\n\n\u03C4 = \u03C4_y + K \xB7 \u03B3\u0307^n for \u03C4 > \u03C4_y\n\nGeneralized yield stress + power-law.",
  // Common fields
  "rho": "Density field.\n\nIn LBM: \u03C1 = \u03A3f_i. Incompressible limit: \u03C1 \u2248 1.",
  "ux": "Velocity component in x-direction.",
  "uy": "Velocity component in y-direction.",
  "uz": "Velocity component in z-direction (3D only).",
  "T": "Temperature field.\n\nRequires `Module thermal`.",
  "steps": "Number of time steps to simulate.",
  "every": "Output frequency (in time steps).",
  "level": "Refinement level.\n\nGrid spacing is divided by 2^level."
};
var KrkHoverProvider = class {
  provideHover(document, position, _token) {
    const wordRange = document.getWordRangeAtPosition(position, /[A-Za-z_][A-Za-z0-9_]*/);
    if (!wordRange) {
      return null;
    }
    const word = document.getText(wordRange);
    const doc = HOVER_DOCS[word];
    if (!doc) {
      return null;
    }
    const markdown = new vscode2.MarkdownString();
    markdown.appendMarkdown(`**${word}**

${doc}`);
    return new vscode2.Hover(markdown, wordRange);
  }
};

// src/extension.ts
var KRK_LANGUAGE = "krk";
function activate(context) {
  const diagnosticCollection = vscode3.languages.createDiagnosticCollection(KRK_LANGUAGE);
  context.subscriptions.push(diagnosticCollection);
  const completionProvider = vscode3.languages.registerCompletionItemProvider(
    { language: KRK_LANGUAGE, scheme: "file" },
    new KrkCompletionProvider(),
    " ",
    "\n"
  );
  context.subscriptions.push(completionProvider);
  const hoverProvider = vscode3.languages.registerHoverProvider(
    { language: KRK_LANGUAGE, scheme: "file" },
    new KrkHoverProvider()
  );
  context.subscriptions.push(hoverProvider);
  const updateDiagnostics = (document) => {
    if (document.languageId !== KRK_LANGUAGE) {
      return;
    }
    const krkDiags = validateDocument(document.getText());
    const diagnostics = krkDiags.map((d) => {
      const range = new vscode3.Range(d.line, 0, d.line, d.endCol ?? 1e3);
      const diag = new vscode3.Diagnostic(range, d.message, d.severity);
      if (d.source) {
        diag.source = d.source;
      }
      return diag;
    });
    diagnosticCollection.set(document.uri, diagnostics);
  };
  if (vscode3.window.activeTextEditor) {
    updateDiagnostics(vscode3.window.activeTextEditor.document);
  }
  context.subscriptions.push(
    vscode3.window.onDidChangeActiveTextEditor((editor) => {
      if (editor) {
        updateDiagnostics(editor.document);
      }
    })
  );
  context.subscriptions.push(
    vscode3.workspace.onDidChangeTextDocument((event) => {
      updateDiagnostics(event.document);
    })
  );
  context.subscriptions.push(
    vscode3.workspace.onDidSaveTextDocument((document) => {
      updateDiagnostics(document);
    })
  );
}
function deactivate() {
}
// Annotate the CommonJS export names for ESM import in node:
0 && (module.exports = {
  activate,
  deactivate
});
//# sourceMappingURL=extension.js.map
