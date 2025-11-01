"""Security scanning script using Bandit and pip-audit.

This script runs security scans to identify vulnerabilities.
Note: Requires bandit and pip-audit to be installed.
"""

import json
import subprocess
import sys
from pathlib import Path
from typing import Optional


def run_bandit(output_dir: Path) -> bool:
    """
    Run Bandit security scanner.
    
    Args:
        output_dir: Directory to save results
        
    Returns:
        True if scan completed successfully
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    json_output = output_dir / "bandit_report.json"
    txt_output = output_dir / "bandit_report.txt"
    
    print("Running Bandit security scanner...")
    print("=" * 80)
    
    try:
        # Run Bandit with JSON and text output
        result = subprocess.run(
            [
                sys.executable, "-m", "bandit",
                "-r", "llmds",
                "-f", "json",
                "-o", str(json_output),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        
        # Also generate text report
        subprocess.run(
            [
                sys.executable, "-m", "bandit",
                "-r", "llmds",
                "-f", "txt",
                "-o", str(txt_output),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        
        # Parse results
        if json_output.exists():
            with open(json_output) as f:
                bandit_data = json.load(f)
            
            # Count issues by severity
            metrics = bandit_data.get("metrics", {})
            total = metrics.get("_totals", {})
            
            print(f"\nBandit Results:")
            print(f"  HIGH:   {total.get('SEVERITY.HIGH', 0)} issues")
            print(f"  MEDIUM: {total.get('SEVERITY.MEDIUM', 0)} issues")
            print(f"  LOW:    {total.get('SEVERITY.LOW', 0)} issues")
            print(f"  Total:  {total.get('CONFIDENCE.HIGH', 0)} high confidence issues")
            
            # List high severity issues
            high_severity = [
                issue for issue in bandit_data.get("results", [])
                if issue.get("issue_severity") == "HIGH"
            ]
            
            if high_severity:
                print(f"\n  HIGH Severity Issues ({len(high_severity)}):")
                for issue in high_severity[:10]:  # Show first 10
                    print(f"    - {issue.get('test_id')}: {issue.get('test_name')}")
                    print(f"      File: {issue.get('filename')}:{issue.get('line_number')}")
            
            print(f"\n  Full report: {txt_output}")
            print(f"  JSON report: {json_output}")
            
            return total.get("SEVERITY.HIGH", 0) == 0
        else:
            print("  Warning: Bandit JSON output not found")
            return False
            
    except FileNotFoundError:
        print("  Error: Bandit not installed. Install with: pip install bandit[toml]")
        return False
    except Exception as e:
        print(f"  Error running Bandit: {e}")
        return False


def run_pip_audit(output_dir: Path) -> bool:
    """
    Run pip-audit to check for known vulnerabilities in dependencies.
    
    Args:
        output_dir: Directory to save results
        
    Returns:
        True if no HIGH/CRITICAL vulnerabilities found
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    json_output = output_dir / "pip_audit_report.json"
    txt_output = output_dir / "pip_audit_report.txt"
    
    print("\nRunning pip-audit security scanner...")
    print("=" * 80)
    
    try:
        # Run pip-audit
        result = subprocess.run(
            [
                sys.executable, "-m", "pip_audit",
                "--format", "json",
                "--output", str(json_output),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        
        # Also generate text output
        subprocess.run(
            [
                sys.executable, "-m", "pip_audit",
                "--format", "text",
                "--output", str(txt_output),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        
        # Parse results
        if json_output.exists():
            with open(json_output) as f:
                audit_data = json.load(f)
            
            vulnerabilities = audit_data.get("vulnerabilities", [])
            high_critical = [
                v for v in vulnerabilities
                if v.get("aliases", [{}])[0].get("severity", "").upper() in ["HIGH", "CRITICAL"]
            ]
            
            print(f"\npip-audit Results:")
            print(f"  Total vulnerabilities: {len(vulnerabilities)}")
            print(f"  HIGH/CRITICAL: {len(high_critical)}")
            
            if high_critical:
                print(f"\n  HIGH/CRITICAL Vulnerabilities:")
                for vuln in high_critical[:10]:  # Show first 10
                    package = vuln.get("name", "unknown")
                    severity = vuln.get("aliases", [{}])[0].get("severity", "UNKNOWN")
                    print(f"    - {package}: {severity}")
                    if "versions" in vuln:
                        print(f"      Affected versions: {vuln['versions']}")
            
            print(f"\n  Full report: {txt_output}")
            print(f"  JSON report: {json_output}")
            
            return len(high_critical) == 0
        else:
            print("  Warning: pip-audit JSON output not found")
            # Check if there were errors
            if result.stderr:
                print(f"  Error output: {result.stderr}")
            return False
            
    except FileNotFoundError:
        print("  Error: pip-audit not installed. Install with: pip install pip-audit")
        return False
    except Exception as e:
        print(f"  Error running pip-audit: {e}")
        if result.stderr:
            print(f"  Error output: {result.stderr}")
        return False


def generate_sbom(output_dir: Path) -> bool:
    """
    Generate Software Bill of Materials (SBOM) using pip-audit.
    
    Args:
        output_dir: Directory to save SBOM
        
    Returns:
        True if SBOM generated successfully
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    sbom_output = output_dir / "sbom.json"
    
    print("\nGenerating SBOM (Software Bill of Materials)...")
    print("=" * 80)
    
    try:
        # Try to generate SBOM using pip-audit (if supported)
        # Note: pip-audit may need additional flags for SBOM generation
        result = subprocess.run(
            [
                sys.executable, "-m", "pip_audit",
                "--format", "json",
                "--output", str(sbom_output),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        
        if sbom_output.exists():
            print(f"  SBOM generated: {sbom_output}")
            print("  Note: For CycloneDX format, consider using cyclonedx-bom or pip-tools")
            return True
        else:
            print("  Warning: SBOM generation may require additional tools")
            print("  Consider using: cyclonedx-py or pip-tools for full SBOM")
            return False
            
    except Exception as e:
        print(f"  Error generating SBOM: {e}")
        return False


def main():
    """Run all security scans."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run security scans")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("audit/security"),
        help="Directory for security scan results (default: audit/security)",
    )
    parser.add_argument(
        "--skip-bandit",
        action="store_true",
        help="Skip Bandit scan",
    )
    parser.add_argument(
        "--skip-pip-audit",
        action="store_true",
        help="Skip pip-audit scan",
    )
    parser.add_argument(
        "--skip-sbom",
        action="store_true",
        help="Skip SBOM generation",
    )
    args = parser.parse_args()
    
    print("Security Scanning")
    print("=" * 80)
    print(f"Output directory: {args.output_dir}")
    print()
    
    results = {}
    
    # Run Bandit
    if not args.skip_bandit:
        results["bandit"] = run_bandit(args.output_dir)
    else:
        print("Skipping Bandit scan")
    
    # Run pip-audit
    if not args.skip_pip_audit:
        results["pip_audit"] = run_pip_audit(args.output_dir)
    else:
        print("Skipping pip-audit scan")
    
    # Generate SBOM
    if not args.skip_sbom:
        results["sbom"] = generate_sbom(args.output_dir)
    else:
        print("Skipping SBOM generation")
    
    # Summary
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    
    all_passed = all(results.values())
    
    for tool, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {tool}: {status}")
    
    if all_passed:
        print("\n✓ All security scans passed!")
        return 0
    else:
        print("\n✗ Some security issues found. Please review reports.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

