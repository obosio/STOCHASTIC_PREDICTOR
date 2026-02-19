#!/usr/bin/env python3
"""
Configuration Migration Script

COMPLIANCE: GAP-2 - Migrate legacy config.toml to new schema with locked parameters

This script migrates old config.toml files (v2.0.x) to the new schema (v2.1.0+)
which includes locked parameter annotations for Level 4 autonomy safety guarantees.

Usage:
    python scripts/migrate_config.py config.toml config_migrated.toml
    python scripts/migrate_config.py --in-place config.toml

Features:
    - Adds [security] section with locked parameters
    - Annotates immutable parameters with __locked__ metadata
    - Validates config structure before writing
    - Creates backup before in-place migration

Safety:
    - Original config backed up to config.toml.bak
    - Atomic write via temporary file
    - Schema validation before replacement

References:
    - IO.tex §3.3.3 - Invariant Protection: Locked Configuration Subsections
    - Implementation_v2.1.0_IO.tex §11 - Configuration Mutation Safety
    
Date: 19 February 2026
"""

import argparse
import logging
import shutil
import sys
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any

try:
    import toml
except ImportError:
    print("ERROR: toml package required. Install with: pip install toml")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Locked parameter schema (immutable parameters for safety)
LOCKED_PARAMETERS_SCHEMA = {
    "io": [
        "snapshot_path",           # Changing path orphans existing snapshots
        "credentials_vault_path",  # Changing path breaks credential lookups
    ],
    "core": [
        "float_precision",         # Switching to 32-bit breaks Malliavin calculus
        "jax_platform",            # Platform changes require full recompilation
    ],
    "security": [
        "telemetry_hash_interval_steps",  # Security-critical parameter
    ],
}


def migrate_config(
    input_path: Path,
    output_path: Path,
    create_backup: bool = True
) -> None:
    """
    Migrate config.toml to new schema with locked parameter annotations.
    
    Args:
        input_path: Path to existing config.toml
        output_path: Path to write migrated config.toml
        create_backup: Create .bak backup before migration (default: True)
    
    Raises:
        FileNotFoundError: If input config doesn't exist
        ValueError: If config validation fails
    """
    if not input_path.exists():
        raise FileNotFoundError(f"Config file not found: {input_path}")
    
    logger.info(f"Loading config from: {input_path}")
    config = toml.load(input_path)
    
    # Add [security] section if missing
    if "security" not in config:
        logger.info("Adding [security] section")
        config["security"] = {
            "telemetry_hash_interval_steps": 1000,
        }
    
    # Annotate locked parameters in each section
    for section, locked_params in LOCKED_PARAMETERS_SCHEMA.items():
        if section not in config:
            logger.warning(f"Section [{section}] not found in config, skipping")
            continue
        
        if "__locked__" not in config[section]:
            config[section]["__locked__"] = locked_params
            logger.info(f"Annotated [{section}] with locked params: {locked_params}")
        else:
            # Merge with existing locked params
            existing_locked = config[section]["__locked__"]
            if not isinstance(existing_locked, list):
                logger.warning(f"Invalid __locked__ format in [{section}], replacing")
                config[section]["__locked__"] = locked_params
            else:
                merged_locked = list(set(existing_locked + locked_params))
                config[section]["__locked__"] = merged_locked
                logger.info(f"Merged locked params in [{section}]: {merged_locked}")
    
    # Add migration metadata
    if "metadata" not in config:
        config["metadata"] = {}
    
    config["metadata"]["migrated_at"] = datetime.now(timezone.utc).isoformat()
    config["metadata"]["migration_script_version"] = "1.0.0"
    config["metadata"]["schema_version"] = "v2.1.0"
    
    # Validate config structure
    validate_config(config)
    
    # Create backup if requested
    if create_backup and output_path.exists():
        backup_path = output_path.with_suffix('.toml.bak')
        logger.info(f"Creating backup: {backup_path}")
        shutil.copy2(output_path, backup_path)
    
    # Atomic write via temporary file
    tmp_path = output_path.with_suffix('.toml.tmp')
    
    logger.info(f"Writing migrated config to: {output_path}")
    with open(tmp_path, 'w') as f:
        toml.dump(config, f)
    
    # Atomic replacement
    tmp_path.replace(output_path)
    
    logger.info("✅ Migration complete")
    logger.info(f"Migrated config saved to: {output_path}")


def validate_config(config: Dict[str, Any]) -> None:
    """
    Validate migrated config structure.
    
    Args:
        config: Migrated config dictionary
    
    Raises:
        ValueError: If validation fails
    """
    logger.info("Validating migrated config...")
    
    # Check required sections
    required_sections = ["io", "core", "security", "metadata"]
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Required section [{section}] missing")
    
    # Check locked parameters exist in their sections
    for section, locked_params in LOCKED_PARAMETERS_SCHEMA.items():
        if section not in config:
            continue
        
        for param in locked_params:
            if param not in config[section] and param != "__locked__":
                logger.warning(
                    f"Locked parameter '{param}' not found in [{section}]. "
                    f"You may need to add it manually."
                )
    
    # Check float_precision is 64
    if config.get("core", {}).get("float_precision") != 64:
        logger.warning(
            "float_precision is not 64. Level 4 autonomy requires 64-bit precision "
            "for Malliavin calculus and signature computations."
        )
    
    logger.info("✅ Validation passed")


def main():
    parser = argparse.ArgumentParser(
        description="Migrate config.toml to v2.1.0 schema with locked parameters"
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Path to input config.toml"
    )
    parser.add_argument(
        "output",
        type=Path,
        nargs='?',
        help="Path to output config.toml (default: config_migrated.toml)"
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Modify input file in-place (creates .bak backup)"
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Skip backup creation (not recommended)"
    )
    
    args = parser.parse_args()
    
    # Determine output path
    if args.in_place:
        output_path = args.input
    elif args.output:
        output_path = args.output
    else:
        output_path = args.input.with_name("config_migrated.toml")
    
    logger.info("=" * 80)
    logger.info("Config Migration Script - v2.1.0")
    logger.info("=" * 80)
    logger.info(f"Input:  {args.input}")
    logger.info(f"Output: {output_path}")
    logger.info(f"Backup: {not args.no_backup}")
    logger.info("=" * 80)
    
    try:
        migrate_config(
            input_path=args.input,
            output_path=output_path,
            create_backup=not args.no_backup
        )
        
        logger.info("")
        logger.info("Next steps:")
        logger.info("1. Review migrated config for correctness")
        logger.info("2. Update config.toml values as needed")
        logger.info("3. Restart predictor to load new configuration")
        
    except Exception as e:
        logger.error(f"❌ Migration failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
