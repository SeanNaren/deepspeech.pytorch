def add_data_opts(parser):
    data_opts = parser.add_argument_group("General Data Options")
    data_opts.add_argument('--manifest-dir', default='./', type=str,
                           help='Output directory for manifests')
    data_opts.add_argument('--min-duration', default=1, type=int,
                           help='Prunes training samples shorter than the min duration (given in seconds, default 1)')
    data_opts.add_argument('--max-duration', default=15, type=int,
                           help='Prunes training samples longer than the max duration (given in seconds, default 15)')
    parser.add_argument('--num-workers', default=4, type=int, help='Number of workers for processing data.')
    parser.add_argument('--sample-rate', default=16000, type=int, help='Sample rate')
    return parser
