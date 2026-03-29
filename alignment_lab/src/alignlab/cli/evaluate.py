"""CLI stub for evaluation entrypoints."""

from __future__ import annotations

from ._shared import build_argument_parser, resolve_config, summarize_config


def main() -> None:
    parser = build_argument_parser("Resolve an evaluation config.")
    args = parser.parse_args()
    config = resolve_config(args.config, sample_limit=args.sample_limit, max_steps=args.max_steps)
    print(summarize_config(config))
    if not args.dry_run:
        raise NotImplementedError("TODO: evaluation CLI wiring is incremental.")


if __name__ == "__main__":
    main()
