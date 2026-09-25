"""Command-line entry points; core commands do not import Flask or COMSOL."""

import argparse
import json
import os
from pathlib import Path
from . import Config, generate, list_families, save_model, __version__


def main(argv=None):
    parser = argparse.ArgumentParser(prog="tpmslab", description="Direct TPMS solid volume meshes")
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("families", help="List built-in formulas and aliases")
    init = sub.add_parser("init-config", help="Write a default JSON configuration")
    init.add_argument("path", type=Path)
    make = sub.add_parser("generate", help="Generate and validate a volume mesh")
    make.add_argument("--config", type=Path)
    make.add_argument("--out", type=Path, required=True)
    make.add_argument("--mph", action="store_true")
    make.add_argument("--solve", action="store_true")
    web = sub.add_parser("web", help="Start the optional local browser interface")
    web.add_argument("--output", type=Path, default=Path.cwd() / "tpmslab-output")
    web.add_argument("--port", type=int, default=0)
    web.add_argument("--no-browser", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.command == "families":
            print(json.dumps(list_families(), indent=2))
            return 0
        if args.command == "init-config":
            with args.path.open("x", encoding="utf8") as stream:
                json.dump(Config().to_dict(), stream, indent=2)
            return 0
        if args.command == "web":
            os.environ["TPMSLAB_OUTPUT_DIR"] = str(args.output.resolve())
            try:
                from .web import serve
            except ImportError as exc:
                raise RuntimeError('Install the web extra: pip install "tpmslab[web]"') from exc
            serve(args.port, args.no_browser)
            return 0
        config = (
            Config.from_dict(json.loads(args.config.read_text(encoding="utf-8-sig")))
            if args.config
            else Config()
        )
        if args.out.exists():
            raise FileExistsError(f"Output already exists: {args.out}")
        model = generate(config, progress=print)
        save_model(model, args.out)
        if args.mph or args.solve:
            from .comsol import build_mph

            build_mph(args.out, model["report"], solve=args.solve, progress=print)
        print(str(args.out.resolve()))
        return 0
    except (ValueError, TypeError, OSError, RuntimeError) as exc:
        parser.exit(2, f"tpmslab: {exc}\n")


if __name__ == "__main__":
    main()
