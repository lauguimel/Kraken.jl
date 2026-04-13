import * as vscode from 'vscode';
import { validateDocument, KrkDiagnostic } from './validator';
import { KrkCompletionProvider } from './completionProvider';
import { KrkHoverProvider } from './hoverProvider';

const KRK_LANGUAGE = 'krk';

export function activate(context: vscode.ExtensionContext): void {
    const diagnosticCollection = vscode.languages.createDiagnosticCollection(KRK_LANGUAGE);
    context.subscriptions.push(diagnosticCollection);

    // Register completion provider
    const completionProvider = vscode.languages.registerCompletionItemProvider(
        { language: KRK_LANGUAGE, scheme: 'file' },
        new KrkCompletionProvider(),
        ' ', '\n'
    );
    context.subscriptions.push(completionProvider);

    // Register hover provider
    const hoverProvider = vscode.languages.registerHoverProvider(
        { language: KRK_LANGUAGE, scheme: 'file' },
        new KrkHoverProvider()
    );
    context.subscriptions.push(hoverProvider);

    // Run diagnostics on open and change
    const updateDiagnostics = (document: vscode.TextDocument): void => {
        if (document.languageId !== KRK_LANGUAGE) {
            return;
        }
        const krkDiags = validateDocument(document.getText());
        const diagnostics = krkDiags.map((d: KrkDiagnostic) => {
            const range = new vscode.Range(d.line, 0, d.line, d.endCol ?? 1000);
            const diag = new vscode.Diagnostic(range, d.message, d.severity);
            if (d.source) {
                diag.source = d.source;
            }
            return diag;
        });
        diagnosticCollection.set(document.uri, diagnostics);
    };

    // Validate on open
    if (vscode.window.activeTextEditor) {
        updateDiagnostics(vscode.window.activeTextEditor.document);
    }

    context.subscriptions.push(
        vscode.window.onDidChangeActiveTextEditor((editor) => {
            if (editor) {
                updateDiagnostics(editor.document);
            }
        })
    );

    context.subscriptions.push(
        vscode.workspace.onDidChangeTextDocument((event) => {
            updateDiagnostics(event.document);
        })
    );

    context.subscriptions.push(
        vscode.workspace.onDidSaveTextDocument((document) => {
            updateDiagnostics(document);
        })
    );
}

export function deactivate(): void {
    // nothing to clean up
}
