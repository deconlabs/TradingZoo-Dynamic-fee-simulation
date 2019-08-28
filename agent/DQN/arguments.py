import argparse


def argparser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--agent_num', type=int, default=1,
                        help='which agent to load')

    parser.add_argument('--device_num', type=int, default=0,
                        help='cuda device num')

    parser.add_argument('--save_num', type=int, default=1,
                        help='folder name')

    parser.add_argument('--risk_aversion', type=float, default=1.,
                        help='risk_aversion_level')

    parser.add_argument('--n_episodes', type=int, default=3000,
                        help='risk_aversion_level')

    parser.add_argument('--fee', type=float, default=.001,
                        help='fee percentage')
    
    parser.add_argument('--render', type=bool, default=False,
                        help='want to render?')
    
    parser.add_argument('--environment', type=str, default="default",
                        help='what environment to use')

    args = parser.parse_args()
    return args
