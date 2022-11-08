import numpy as np
import torch
from metric import evaluate_graph_embeddings_using_svm


def train_eval_test_v1(model, optimizer, scheduler, pooler, train_loader, eval_loader, summary_writer, train_test_config):
    n_epoch = train_test_config['max_epoch']
    device = train_test_config['device']
    # pretrain
    for each_epoch in range(n_epoch):
        model.train()
        loss_list = []
        for batch in train_loader:
            batch = batch.to(device)
            loss, loss_dict = model(batch, batch.x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())
        scheduler.step()
        print(each_epoch, n_epoch, np.mean(loss_list))
        summary_writer.add_scalar('Pretrain/Loss', np.mean(loss_list), each_epoch)

    # evaluation
    model.eval()
    x_list = []
    y_list = []
    with torch.no_grad():
        for i, batch_g in enumerate(eval_loader):
            labels = batch_g.y
            batch_g = batch_g.to(device)
            out = model.embed(batch_g, batch_g.x)
            out = pooler(x=out, batch=batch_g.batch)
            y_list.append(labels.numpy())
            x_list.append(out.cpu().numpy())
    x = np.concatenate(x_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    test_f1, test_std = evaluate_graph_embeddings_using_svm(x, y)
    print(f"#Test_f1: {test_f1:.4f}±{test_std:.4f}")
    return test_f1


