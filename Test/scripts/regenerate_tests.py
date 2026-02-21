#!/usr/bin/env python3
"""
Regenerate auto-generated tests from source code discovery.

This utility regenerates Test/tests/ by:
1. Discovering all Python modules in the source directory
2. Inspecting each module's callables (functions/classes)
3. Generating pytest test files

Usage:
    python Test/scripts/regenerate_tests.py              # Defaults: Python/ → Test/tests/
    python Test/scripts/regenerate_tests.py --dry-run    # Show what would be generated

Environment:
    Automatically detects project root from config.toml presence
"""

import sys
from argparse import ArgumentParser
from pathlib import Path


def main():
    parser = ArgumentParser(description="Regenerate auto-generated test files")
    parser.add_argument(
        "--source-dir", default="Python", help="Source directory name (default: Python)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: Test/tests)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be generated without writing files",
    )

    args = parser.parse_args()

    # Setup paths
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent
    test_dir = script_dir.parent

    # Add project and Test to path
    sys.path.insert(0, str(project_root))
    sys.path.insert(0, str(test_dir))

    # Import framework
    try:
        from framework.generator import TestGenerator
    except ImportError:
        from Test.framework.generator import TestGenerator

    # Generator
    output_dir = args.output_dir or project_root / "Test" / "tests"

    print("🔧 Test Generator v2.1.0")
    print("\n📂 Configuration:")
    print(f"   Project root:  {project_root}")
    print(f"   Source dir:    {project_root / args.source_dir}")
    print(f"   Output dir:    {output_dir}")
    print(f"   Dry run:       {args.dry_run}")
    print("\n🔍 Discovering modules...")

    try:
        generator = TestGenerator(project_root, args.source_dir)
        print(f"✅ Found {len(generator.python_files)} Python modules")

        if not args.dry_run:
            print("⚙️  Generating tests with full synchronization...")
            count = generator.generate_all_tests(output_dir)
            print(f"✅ Synchronized tests")
            print(f"   - Auto-generated: {count} test files")
            print(f"   - Cleaned up obsolete tests (modules deleted)")
            print(f"   - Updated existing tests (module structure changed)")
            print("📋 Next steps:")
            print(
                f"   1. Run tests:  pytest {output_dir.relative_to(project_root)} -v"
            )
            print(f"   2. View docs:  cat {project_root / 'Test' / 'README.md'}")
        else:
            print(f"📋 Would synchronize {len(generator.python_files)} modules")
            print("   (Run without --dry-run to actually sync tests)")

        return 0

    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
