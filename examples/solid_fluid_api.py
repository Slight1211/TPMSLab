"""Generate a graded solid/fluid mesh; optionally create or solve a COMSOL MPH."""
import argparse
from datetime import datetime
from pathlib import Path

from tpmslab import Config, generate, save_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=None, help='New output directory')
    parser.add_argument('--family', default='Gyroid')
    parser.add_argument('--resolution', type=int, default=12)
    parser.add_argument('--comsol', choices=['none', 'create', 'solve'], default='none')
    args = parser.parse_args()
    destination = args.out or Path('results') / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    if destination.exists():
        parser.error(f'Output already exists: {destination}; choose a new directory.')
    config = Config(
        family=args.family, mode='sheet', domain_mode='solid_fluid',
        size=(5.0, 5.0, 5.0), cells=(1, 1, 1),
        density_start=0.25, density_end=0.50, gradient='linear', axis='z',
        resolution=args.resolution, quality_strategy='quality_fan',
    )
    model = generate(config, progress=print)
    folder = save_model(model, destination)
    print(f'Exported {len(model["tetra"]):,} tetrahedra to {folder}')
    if args.comsol != 'none':
        from tpmslab.comsol import build_mph
        verification = build_mph(
            folder, model['report'], solve=(args.comsol == 'solve'), progress=print,
        )
        print(verification)
    return folder


if __name__ == '__main__':
    main()
