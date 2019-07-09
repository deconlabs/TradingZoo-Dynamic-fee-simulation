import argparse


def argparser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--n_agent', type=int, default=20,
                        help='에이전트 수')
    parser.add_argument('--no_short', type=bool, default=False,
                        help='Allow Inverse trading?')

    args = parser.parse_args()
    return args
