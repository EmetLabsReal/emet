"""Emet command-line interface.

Entry points:
    emet tui          Launch the terminal user interface (requires textual)
    emet decide FILE  Decide a JSON problem file
    emet certify FILE Certify a JSON problem file and print the sealed hash
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _cmd_tui(_args: argparse.Namespace) -> None:
    try:
        from emet.tui import launch_tui
    except ImportError:
        print(
            "textual is required for the TUI.\n"
            "Install it with: uv pip install textual\n"
            "Or: pip install emet[tui]",
            file=sys.stderr,
        )
        sys.exit(1)
    launch_tui()


def _cmd_decide(args: argparse.Namespace) -> None:
    import emet

    raw = Path(args.file).read_text()
    problem = json.loads(raw)

    if "entries" in problem:
        report = emet.decide(problem)
    else:
        import numpy as np
        H = np.array(problem["matrix"], dtype=float)
        report = emet.decide_dense_matrix(
            H,
            retained=problem["retained"],
            omitted=problem["omitted"],
        )

    if args.pretty:
        print(json.dumps(report, indent=2))
    elif args.json:
        print(json.dumps(report))
    else:
        _print_human(report)


def _cmd_certify(args: argparse.Namespace) -> None:
    import numpy as np
    import emet
    from emet.certificate import certify, to_json

    raw = Path(args.file).read_text()
    problem = json.loads(raw)

    if "entries" in problem:
        dim = problem["dimension"]
        H = np.zeros((dim, dim))
        for e in problem["entries"]:
            H[e["row"], e["col"]] = e["value"]
        retained = problem["retained"]
        omitted = problem["omitted"]
    else:
        H = np.array(problem["matrix"], dtype=float)
        retained = problem["retained"]
        omitted = problem["omitted"]

    report = emet.decide_dense_matrix(H, retained=retained, omitted=omitted)
    cert = certify(H, retained, omitted, report)

    if args.json:
        print(to_json(cert))
    elif args.pretty:
        print(to_json(cert, pretty=True))
    else:
        print(f"seal:      {cert.seal}")
        print(f"regime:    {cert.regime}")
        print(f"licensed:  {cert.licensed}")
        print(f"chi:       {cert.chi}")
        print(f"kahan:     {'certified' if cert.kahan_certified else 'not certified'}")
        if cert.kahan_margin is not None:
            print(f"margin:    {cert.kahan_margin:.6e}")
        print(f"input:     {cert.input_hash[:16]}...")
        if cert.reduced_matrix_hash:
            print(f"H_eff:     {cert.reduced_matrix_hash[:16]}...")


def _print_human(report: dict) -> None:
    regime = report.get("regime", "unknown").upper()
    valid = report.get("valid", False)
    reason = report.get("reason_code", "")
    pp = report.get("partition_profile", {})

    lines = [
        f"Regime: {regime}",
        f"Licensed: {'YES' if valid else 'NO'}",
        f"Reason: {reason}",
        f"Retained / omitted: {pp.get('retained_dimension', '?')} / {pp.get('omitted_dimension', '?')}",
    ]

    m = report.get("advanced_metrics")
    if m:
        lines.append(f"lambda (coupling):      {m['lambda']:.12f}")
        lines.append(f"gamma (control):        {m['gamma']:.12f}")
        if m.get("chi") is not None:
            lines.append(f"chi = (lambda/gamma)^2: {m['chi']:.12f}")

    print("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="emet",
        description="Spectral reduction certification.",
    )
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("tui", help="Launch the terminal user interface")

    p_decide = sub.add_parser("decide", help="Decide a JSON problem file")
    p_decide.add_argument("file", help="Path to problem JSON")
    p_decide.add_argument("--json", action="store_true", help="Compact JSON output")
    p_decide.add_argument("--pretty", action="store_true", help="Pretty JSON output")

    p_certify = sub.add_parser("certify", help="Certify and seal a reduction")
    p_certify.add_argument("file", help="Path to problem JSON")
    p_certify.add_argument("--json", action="store_true", help="Compact JSON output")
    p_certify.add_argument("--pretty", action="store_true", help="Pretty JSON output")

    args = parser.parse_args()

    if args.command == "tui":
        _cmd_tui(args)
    elif args.command == "decide":
        _cmd_decide(args)
    elif args.command == "certify":
        _cmd_certify(args)
    else:
        parser.print_help()
        sys.exit(0)


if __name__ == "__main__":
    main()
