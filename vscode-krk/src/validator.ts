import { DiagnosticSeverity } from 'vscode';

export interface KrkDiagnostic {
    line: number;
    endCol?: number;
    message: string;
    severity: DiagnosticSeverity;
    source?: string;
}

const TOP_LEVEL_KEYWORDS = [
    'Simulation', 'Domain', 'Physics', 'Define', 'Obstacle', 'Fluid',
    'Boundary', 'Refine', 'Initial', 'Velocity', 'Module', 'Run',
    'Output', 'Diagnostics', 'Rheology', 'Setup', 'Preset', 'Sweep'
];

function levenshtein(a: string, b: string): number {
    const m = a.length;
    const n = b.length;
    const dp: number[][] = Array.from({ length: m + 1 }, () => Array(n + 1).fill(0));
    for (let i = 0; i <= m; i++) { dp[i][0] = i; }
    for (let j = 0; j <= n; j++) { dp[0][j] = j; }
    for (let i = 1; i <= m; i++) {
        for (let j = 1; j <= n; j++) {
            dp[i][j] = a[i - 1] === b[j - 1]
                ? dp[i - 1][j - 1]
                : 1 + Math.min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]);
        }
    }
    return dp[m][n];
}

function suggestKeyword(word: string): string | null {
    let best = '';
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

export function validateDocument(text: string): KrkDiagnostic[] {
    const diagnostics: KrkDiagnostic[] = [];
    const lines = text.split('\n');

    let hasSimulation = false;
    let hasPreset = false;
    let hasDomain = false;
    let hasRun = false;
    let hasThermalModule = false;
    let hasPr = false;
    let hasAlpha = false;
    let nuValue: number | null = null;
    let nuLine = -1;
    let insideBlock = false;
    let braceDepth = 0;

    for (let i = 0; i < lines.length; i++) {
        const line = lines[i];
        const trimmed = line.trim();

        // Track brace depth
        for (const ch of line) {
            if (ch === '{') { braceDepth++; insideBlock = true; }
            if (ch === '}') { braceDepth--; if (braceDepth <= 0) { insideBlock = false; braceDepth = 0; } }
        }

        // Skip empty lines and comments
        if (trimmed === '' || trimmed.startsWith('#')) {
            continue;
        }

        // Skip lines inside blocks
        if (insideBlock && !trimmed.match(/^\s*(Simulation|Domain|Physics|Define|Obstacle|Fluid|Boundary|Refine|Initial|Velocity|Module|Run|Output|Diagnostics|Rheology|Setup|Preset|Sweep)\b/)) {
            continue;
        }

        // Check first word
        const firstWordMatch = trimmed.match(/^([A-Za-z_]\w*)\b/);
        if (!firstWordMatch) {
            continue;
        }
        const firstWord = firstWordMatch[1];

        // Check if it's a known keyword
        if (!TOP_LEVEL_KEYWORDS.includes(firstWord)) {
            // Only flag capitalized words that look like they should be keywords
            if (firstWord[0] === firstWord[0].toUpperCase() && firstWord.length > 2) {
                const suggestion = suggestKeyword(firstWord);
                const msg = suggestion
                    ? `Unknown keyword "${firstWord}". Did you mean "${suggestion}"?`
                    : `Unknown keyword "${firstWord}".`;
                diagnostics.push({
                    line: i,
                    endCol: firstWord.length,
                    message: msg,
                    severity: DiagnosticSeverity.Error,
                    source: 'krk'
                });
            }
            continue;
        }

        // Track presence
        if (firstWord === 'Simulation') { hasSimulation = true; }
        if (firstWord === 'Preset') { hasPreset = true; }
        if (firstWord === 'Domain') { hasDomain = true; }
        if (firstWord === 'Run') { hasRun = true; }

        // Module thermal check
        if (firstWord === 'Module' && /\bthermal\b/.test(trimmed)) {
            hasThermalModule = true;
        }

        // Physics line: extract nu, check for Pr/alpha
        if (firstWord === 'Physics') {
            const nuMatch = trimmed.match(/\bnu\s*=\s*([0-9eE.+-]+)/);
            if (nuMatch) {
                nuValue = parseFloat(nuMatch[1]);
                nuLine = i;
                if (isNaN(nuValue) || nuValue <= 0) {
                    diagnostics.push({
                        line: i,
                        message: `nu must be positive (got ${nuMatch[1]}).`,
                        severity: DiagnosticSeverity.Error,
                        source: 'krk'
                    });
                }
            }
            if (/\bPr\s*=/.test(trimmed)) { hasPr = true; }
            if (/\balpha\s*=/.test(trimmed)) { hasAlpha = true; }
        }
    }

    // Required blocks
    if (!hasSimulation && !hasPreset) {
        diagnostics.push({
            line: 0,
            message: 'Missing "Simulation" or "Preset" declaration.',
            severity: DiagnosticSeverity.Error,
            source: 'krk'
        });
    }

    if (hasSimulation && !hasPreset && !hasDomain) {
        diagnostics.push({
            line: 0,
            message: 'Missing "Domain" definition. Required when using Simulation.',
            severity: DiagnosticSeverity.Error,
            source: 'krk'
        });
    }

    if (!hasRun) {
        diagnostics.push({
            line: lines.length - 1,
            message: 'Missing "Run" command.',
            severity: DiagnosticSeverity.Warning,
            source: 'krk'
        });
    }

    // Stability check: tau = 3*nu + 0.5
    if (nuValue !== null && nuValue > 0 && nuLine >= 0) {
        const tau = 3 * nuValue + 0.5;
        if (tau < 0.51) {
            diagnostics.push({
                line: nuLine,
                message: `Stability warning: τ = ${tau.toFixed(4)} is very close to 0.5. Simulation may be unstable.`,
                severity: DiagnosticSeverity.Warning,
                source: 'krk-stability'
            });
        }
        if (tau > 2.0) {
            diagnostics.push({
                line: nuLine,
                message: `Stability warning: τ = ${tau.toFixed(4)} > 2.0. High viscosity may cause inaccuracies.`,
                severity: DiagnosticSeverity.Warning,
                source: 'krk-stability'
            });
        }
    }

    // Thermal module requires Pr or alpha
    if (hasThermalModule && !hasPr && !hasAlpha) {
        diagnostics.push({
            line: 0,
            message: 'Module thermal requires "Pr" or "alpha" in Physics.',
            severity: DiagnosticSeverity.Warning,
            source: 'krk'
        });
    }

    return diagnostics;
}
