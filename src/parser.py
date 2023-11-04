from pathlib import Path
from argparse import ArgumentParser


def existing_file_path(s):
    p = Path(s)
    if not p.exists() or not p.is_file():
        raise TypeError('Path is invalid or is not a file.')
    return p


def parsed_args():
    parser = ArgumentParser('NerFlexGIF', description='A perfect GIF creator.',
                            epilog="Flow: 1. Process video (COLMAP). 2. Train NeRF model."
                                   "3. Crop excess parts. 4. Synthesize middle frames. 5. Generate GIF!")
    subparsers = parser.add_subparsers(description='Creation or Visualization of GIFs.')
    gif_parser = subparsers.add_parser('create_gif', help='Create a GIF via NerFlexGIF.')
    gif_parser.set_defaults(which='create_gif')
    gif_parser.add_argument('-v', '--video-path', type=existing_file_path, required=True,
                            help="Path to video file to process.")
    gif_parser.add_argument('-p', '--project-name', type=str,
                            help='Name of project. Default is video name.')
    gif_parser.add_argument('--skip-train', action='store_true',
                            help="Skip training, model was trained ahead of time.")
    gif_parser.add_argument('--train-from-checkpoint', action='store_true',
                            help="Continue training from checkpoint")
    gif_parser.add_argument('-g', '--gif-output-name', dest='gif_filename', type=str,
                            help='Name of output GIF. Default is video name.')
    gif_parser.add_argument('--keep-gif-dir', default=False, action='store_true',
                            help='Keep pre-GIF created folder with real+synthesized images after GIF creation.')
    vis_parser = subparsers.add_parser('visualization_kit',
                                       help='Visualization utils. Assumes that NerFlexGIF went through a full run of'
                                            ' \'create_gif\'.')
    vis_parser.set_defaults(which='visualization_kit')
    vis_parser.add_argument('-i', '--interactive', action='store_true',
                            help='Create interactive 3d path position visualization. Otherwise, assumed already exist.')
    vis_parser.add_argument('--interactive-name', dest='inter_gif_name', type=str,
                            help='Create a standalone GIF for the interactive path of cameras. '
                                 'Default: GIF not created.')
    vis_parser.add_argument('-g', '--gif-output-path', dest='gif_filepath', type=str,
                            help='Synced Rendered + Positional images created GIF path.')
    args = parser.parse_args()
    if args.which == 'create_gif' and args.skip_train and args.train_from_checkpoint:
        parser.error('Can not use \'--skip-train\' and \'--train-from-checkpoint\' simultaneously!')
    return parser.parse_args()
