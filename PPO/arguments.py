import argparse


def argparser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--load_weight', type=bool, default=False,
                        help='웨이트 로드 할 것인지?')
    parser.add_argument('--n', type=int, default=100,
                        help='몇번쨰 트레이닝 웨이트 로드할것인지?')
    parser.add_argument('--additional_num_step', type=int, default=0,
                        help='몇번쨰 트레이닝 웨이트 로드할것인지?')
    parser.add_argument('--beta_decay', type=bool, default=False,
                        help='몇번쨰 트레이닝 웨이트 로드할것인지?')



    args = parser.parse_args()
    return args