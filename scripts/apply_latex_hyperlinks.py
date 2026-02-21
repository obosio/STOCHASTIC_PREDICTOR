#!/usr/bin/env python3
"""
Apply hyperlinks to LaTeX document references.

This script reads the analysis JSON and converts plain text file/doc references
into clickeable hyperlinks using the custom LaTeX commands defined in preambles.
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

# Load analysis
ANALYSIS_FILE = Path(__file__).parent.parent / "latex_references_analysis.json"
DOC_ROOT = Path(__file__).parent.parent / "doc"

# Common file patterns to replace
PATTERNS = [
    # Pattern 1: Python files in texttt: texttt{Python/...}
    (
        r'\\texttt\{(Python/[^}]+\.py)\}',
        lambda m: f'\\\\filehref{{{m.group(1)}}}',
        'texttt_python'
    ),
    # Pattern 2: Config files in texttt
    (
        r'\\texttt\{(config\.toml|requirements\.txt|\.github[^}]*|\.gitignore|\.env)\}',
        lambda m: f'\\\\filehref{{{m.group(1)}}}',
        'texttt_config'
    ),
    # Pattern 3: Test scripts in texttt
    (
        r'\\texttt\{(tests/[^}]+\.py)\}',
        lambda m: f'\\\\filehref{{{m.group(1)}}}',
        'texttt_test'
    ),
    # Pattern 4: Doc paths in texttt
    (
        r'\\texttt\{(doc/[^}]+)\}',
        lambda m: f'\\\\filehref{{{m.group(1)}}}',
        'texttt_doc'
    ),
    # Pattern 5: LaTeX doc references in texttt
    (
        r'\\texttt\{([^}]*Stochastic_Predictor_[^}]+\.tex)\}',
        lambda m: f'\\\\dochref{{{Path(m.group(1)).stem}}}{{\\\\texttt{{{m.group(1)}}}}}',
        'latex_spec'
    ),
    # Pattern 6: Implementation doc references
    (
        r'\\texttt\{([^}]*Implementation_v[^}]+\.tex)\}',
        lambda m: f'\\\\dochref{{{Path(m.group(1)).stem}}}{{\\\\texttt{{{m.group(1)}}}}}',
        'latex_impl'
    ),
    # Pattern 7: Testing doc references
    (
        r'\\texttt\{(Testing_Infrastructure_[^}]+\.tex)\}',
        lambda m: f'\\\\dochref{{{Path(m.group(1)).stem}}}{{\\\\texttt{{{m.group(1)}}}}}',
        'latex_test'
    ),
    # Pattern 8: Code references with "\.py" in italic/emphasis
    (
        r'\\emph\{(Python/[^}]+\.py)\}',
        lambda m: f'\\\\filehref{{{m.group(1)}}}',
        'emph_python'
    ),
]

def load_analysis() -> Dict:
    """Load reference analysis JSON."""
    if not ANALYSIS_FILE.exists():
        print(f"Analysis file not found: {ANALYSIS_FILE}")
        return {}
    
    with open(ANALYSIS_FILE) as f:
        return json.load(f)

def get_latex_files() -> List[Path]:
    """Get all LaTeX files."""
    return sorted(DOC_ROOT.glob("latex/**/*.tex"))

def apply_hyperlinks_to_file(filepath: Path, dry_run: bool = False) -> Tuple[int, int]:
    """
    Apply hyperlinks to a single LaTeX file.
    
    Returns: (matches_applied, lines_modified)
    """
    if not filepath.exists():
        print(f"File not found: {filepath}")
        return 0, 0
    
    content = filepath.read_text(encoding='utf-8')
    original = content
    
    applied = 0
    for pattern, replacement, ref_type in PATTERNS:
        matches = list(re.finditer(pattern, content))
        if matches:
            content = re.sub(pattern, replacement, content)
            applied += len(matches)
            print(f"  [{ref_type:15}] {len(matches):2} →  {filepath.name}")
    
    if not dry_run and content != original:
        filepath.write_text(content, encoding='utf-8')
        return applied, 1  # 1 file modified
    
    return applied, 0

def main():
    """Main entry point."""
    print("\n" + "="*80)
    print("LaTeX Hyperlink Application")
    print("="*80)
    
    latex_files = get_latex_files()
    print(f"\nFound {len(latex_files)} LaTeX files\n")
    
    total_applied = 0
    files_modified = 0
    
    for filepath in latex_files:
        applied, modified = apply_hyperlinks_to_file(filepath, dry_run=False)
        total_applied += applied
        files_modified += modified
    
    print(f"\n{'='*80}")
    print(f"Summary:")
    print(f"  - Files processed: {len(latex_files)}")
    print(f"  - Files modified: {files_modified}")
    print(f"  - Hyperlinks applied: {total_applied}")
    print(f"{'='*80}\n")

if __name__ == '__main__':
    main()
