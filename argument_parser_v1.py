import argparse


def parser_args():
    parser = argparse.ArgumentParser(description="GraphMAE-PyG")
    # dataset argument
    parser.add_argument('--dataset_root', type=str, default='data')
    parser.add_argument('--dataset_name', type=str, default='COLLAB')
    parser.add_argument('--batch_size', type=int, default=32)
    # model argument
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--in_dim', type=int, default=-1)
    parser.add_argument('--num_hidden', type=int, default=512)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--num_out_heads', type=int, default=8)
    parser.add_argument('--activation', type=str, default='prelu')
    parser.add_argument('--feat_drop', type=float, default=0.2)
    parser.add_argument('--attn_drop', type=float, default=0.1)
    parser.add_argument('--negative_slope', type=float, default=0.1)
    parser.add_argument('--encoder', type=str, default='gin')
    parser.add_argument('--decoder', type=str, default='gin')
    parser.add_argument('--replace_rate', type=float, default=0.1)
    parser.add_argument('--mask_rate', type=float, default=0.75)
    parser.add_argument('--drop_edge_rate', type=float, default=0.0)
    parser.add_argument('--optimizer_name', type=str, default='adam')
    parser.add_argument('--loss_fn', type=str, default='sce')
    parser.add_argument('--linear_prob', type=bool, default=True)
    parser.add_argument('--alpha_l', type=int, default=2)
    parser.add_argument('--norm', type=str, default='layernorm')

    parser.add_argument('--scheduler', type=bool, default=True)
    parser.add_argument('--pooling', type=str, default='sum')
    parser.add_argument('--deg4feat', type=bool, default=False)
    parser.add_argument('--residual', type=bool, default=True)
    parser.add_argument('--concat_hidden', type=bool, default=False)

    # train evaluation test argument
    parser.add_argument('--seeds', type=int, default=0)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--max_epoch', type=int, default=100)
    parser.add_argument('--max_epoch_f', type=int, default=0)
    parser.add_argument('--lr', type=float, default=0.00015)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--lr_f', type=float, default=0.005)
    parser.add_argument('--weight_decay_f', type=float, default=0.0)

    parser.add_argument('--load_model', type=bool, default=False)
    parser.add_argument('--save_model', type=bool, default=False)
    parser.add_argument('--logging', type=bool, default=False)
    parser.add_argument('--log_filepath', type=str, default='runs')
    args = parser.parse_args()
    return args
