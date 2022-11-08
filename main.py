from argument_parser_v1 import parser_args
from dataloader_v1 import build_dataloader
from model_building import build_model
from train_eval_test_v1 import train_eval_test_v1
from utils import seed_setting, get_summary_writer, record_configuration


def main(args):
    seed = args.seeds
    seed_setting(seed)
    log_filepath = args.log_filepath
    dataset_config = {
        'dataset_root': args.dataset_root,
        'dataset_name': args.dataset_name,
        'batch_size': args.batch_size,
    }
    model_config = {
        'num_layers': args.num_layers,
        'in_dim': args.in_dim,
        'num_hidden': args.num_hidden,
        'num_heads': args.num_heads,
        'num_out_heads': args.num_out_heads,
        'activation': args.activation,
        'feat_drop': args.feat_drop,
        'attn_drop': args.attn_drop,
        'negative_slope': args.negative_slope,
        'encoder': args.encoder,
        'decoder': args.decoder,
        'replace_rate': args.replace_rate,
        'mask_rate': args.mask_rate,
        'drop_edge_rate': args.drop_edge_rate,
        'optimizer_name': args.optimizer_name,
        'loss_fn': args.loss_fn,
        'linear_prob': args.linear_prob,
        'alpha_l': args.alpha_l,
        'norm': args.norm,
        'scheduler': args.scheduler,
        'pooling': args.pooling,
        'deg4feat': args.deg4feat,
        'residual': args.residual,
        'concat_hidden': args.concat_hidden,
    }
    train_test_config = {
        'seeds': seed,
        'device': args.device,
        'max_epoch': args.max_epoch,
        'max_epoch_f': args.max_epoch_f,
        'lr': args.lr,
        'weight_decay': args.weight_decay,
        'lr_f': args.lr_f,
        'weight_decay_f': args.weight_decay_f,
    }
    summary_writer, log_dir = get_summary_writer(log_filepath)
    train_loader, test_loader, dataset_information = build_dataloader(dataset_config)
    print('dataloader build finish')
    # change the model input dimension according to the dataset information
    model_config['in_dim'] = dataset_information['node_feat']
    model, optimizer, scheduler, pooler = build_model(model_config, train_test_config)
    print('model build finish')
    record_configuration(save_dir=log_dir, configuration_dict={
        'MODEL': model_config,
        'DATASET': dataset_config,
        'TRAIN': train_test_config,
    })
    print(model_config, '\n', dataset_config, '\n', train_test_config, '\n')
    test_f1 = train_eval_test_v1(model, optimizer, scheduler, pooler, train_loader, test_loader, summary_writer, train_test_config)
    return test_f1


if __name__ == '__main__':
    args = parser_args()
    import warnings
    warnings.filterwarnings("ignore")
    main(args)
