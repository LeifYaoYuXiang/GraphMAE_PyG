import numpy as np

from argument_parser_v1 import parser_args
from main import main

seed_list = list(range(10))

if __name__ == '__main__':
    args = parser_args()
    import warnings
    warnings.filterwarnings("ignore")
    test_f1_record_list = []
    for each_seed in seed_list:
        args.seeds = each_seed
        test_f1 = main(args)
        test_f1_record_list.append(test_f1)
        print(each_seed, test_f1_record_list)
    print('avg', np.mean(test_f1_record_list))