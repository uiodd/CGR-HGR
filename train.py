import os

import numpy as np, argparse, time, random
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from dataloader import IEMOCAPDataset, MELDDataset
from model import MaskedNLLLoss, MaskedKLDivLoss, Transformer_Based_Model
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, classification_report
import pickle as pk


def _resolve_existing_path(path, fallback_candidates=None):
    candidates = []
    if path:
        candidates.append(path)
    if fallback_candidates:
        candidates.extend(fallback_candidates)
    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(
        "找不到数据文件，已尝试路径: {}".format(
            ", ".join([str(item) for item in candidates if item])
        )
    )


def _project_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def infer_comet_feature_dim(knowledge_path):
    relation_values = pk.load(open(knowledge_path, "rb"), encoding="latin1")
    if not relation_values:
        raise ValueError("知识文件为空，无法推断向量维度: {}".format(knowledge_path))
    first_relation = relation_values[0]
    if not isinstance(first_relation, dict) or not first_relation:
        raise ValueError("知识文件结构异常，无法推断向量维度: {}".format(knowledge_path))
    first_vid = next(iter(first_relation.keys()))
    sentence_features = first_relation[first_vid]
    if not sentence_features:
        raise ValueError("知识文件中存在空句子特征，无法推断向量维度: {}".format(knowledge_path))
    vector = np.asarray(sentence_features[0])
    if vector.ndim == 0:
        raise ValueError("知识向量维度异常，无法推断向量维度: {}".format(knowledge_path))
    return int(vector.shape[-1])


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def validate_runtime_summary(runtime_summary, use_knowledge, use_graph):
    if runtime_summary is None:
        raise ValueError("运行时核验失败：模型未返回任何调试摘要。")
    if use_knowledge:
        if runtime_summary["conflict_forward_calls"] <= 0:
            raise ValueError("运行时核验失败：已启用 --use-knowledge，但 ConflictEstimator 未被调用。")
        if runtime_summary["knowledge_forward_calls"] <= 0:
            raise ValueError("运行时核验失败：已启用 --use-knowledge，但 KnowledgeEnhancer 未被调用。")
        if runtime_summary["knowledge_last_shape"] is None:
            raise ValueError("运行时核验失败：已启用 --use-knowledge，但没有记录到 knowledge 张量形状。")
    if use_graph:
        if runtime_summary["conflict_forward_calls"] <= 0:
            raise ValueError("运行时核验失败：已启用 --use-graph，但 ConflictEstimator 未被调用。")
        if runtime_summary["graph_forward_calls"] <= 0:
            raise ValueError("运行时核验失败：已启用 --use-graph，但 HeterogeneousKnowledgeGraphEnhancer 未被调用。")
        if runtime_summary["graph_conflict_aware_calls"] <= 0:
            raise ValueError("运行时核验失败：已启用 --use-graph，但图传播没有拿到 conflict_score。")
        if runtime_summary["graph_knowledge_node_count"] <= 0:
            raise ValueError("运行时核验失败：已启用 --use-graph，但没有记录到任何 knowledge node 进入图传播。")
        if runtime_summary["graph_total_node_count"] <= runtime_summary["graph_knowledge_node_count"]:
            raise ValueError("运行时核验失败：异构图节点统计异常，请检查 utterance node 与 knowledge node 是否都已构造。")
        if runtime_summary["graph_bridge_edge_count"] <= 0:
            raise ValueError("运行时核验失败：已启用 --use-graph，但没有记录到任何 utterance-knowledge bridge edge。")
    if use_knowledge or use_graph:
        if runtime_summary["fusion_forward_calls"] <= 0:
            raise ValueError("运行时核验失败：已启用增强模块，但最终融合头未被调用。")


def format_runtime_summary(runtime_summary):
    return (
        "[Stage3][RuntimeCheck] "
        "conflict_calls={conflict_forward_calls} "
        "conflict_mean={conflict_mean} "
        "knowledge_calls={knowledge_forward_calls} "
        "knowledge_shape={knowledge_last_shape} "
        "knowledge_attn_mean={knowledge_attn_mean} "
        "knowledge_reliability_mean={knowledge_reliability_mean} "
        "graph_calls={graph_forward_calls} "
        "graph_conflict_calls={graph_conflict_aware_calls} "
        "graph_edge_mean={graph_edge_mean} "
        "graph_bridge_mean={graph_bridge_weight_mean} "
        "graph_utter_edge_mean={graph_utterance_edge_mean} "
        "graph_knowledge_edge_mean={graph_knowledge_edge_mean} "
        "graph_bridge_edges={graph_bridge_edge_count} "
        "graph_knowledge_nodes={graph_knowledge_node_count} "
        "graph_total_nodes={graph_total_node_count} "
        "fusion_calls={fusion_forward_calls} "
        "fusion_gate_mean={fusion_gate_mean}"
    ).format(**runtime_summary)

def get_train_valid_sampler(trainset, valid=0.1, dataset='MELD'):
    size = len(trainset)
    idx = list(range(size))
    split = int(valid*size)
    return SubsetRandomSampler(idx[split:]), SubsetRandomSampler(idx[:split])

def get_MELD_loaders(batch_size=32, valid=0.1, num_workers=0, pin_memory=False):
    trainset = MELDDataset('data/meld_multimodal_features.pkl')
    train_sampler, valid_sampler = get_train_valid_sampler(trainset, valid, 'MELD')
    train_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              sampler=train_sampler,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)
    valid_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              sampler=valid_sampler,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    testset = MELDDataset('data/meld_multimodal_features.pkl', train=False)
    test_loader = DataLoader(testset,
                             batch_size=batch_size,
                             collate_fn=testset.collate_fn,
                             num_workers=num_workers,
                             pin_memory=pin_memory)
    return train_loader, valid_loader, test_loader


def get_IEMOCAP_loaders(batch_size=32, valid=0.1, num_workers=0, pin_memory=False,
                        data_path=None, knowledge_path=None, use_knowledge=False):
    trainset = IEMOCAPDataset(
        data_path=data_path,
        knowledge_path=knowledge_path,
        use_knowledge=use_knowledge,
    )
    train_sampler, valid_sampler = get_train_valid_sampler(trainset, valid)
    train_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              sampler=train_sampler,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)
    valid_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              sampler=valid_sampler,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    testset = IEMOCAPDataset(
        train=False,
        data_path=data_path,
        knowledge_path=knowledge_path,
        use_knowledge=use_knowledge,
    )
    test_loader = DataLoader(testset,
                             batch_size=batch_size,
                             collate_fn=testset.collate_fn,
                             num_workers=num_workers,
                             pin_memory=pin_memory)
    return train_loader, valid_loader, test_loader


def train_or_eval_model(model, loss_function, kl_loss, dataloader, epoch, optimizer=None, train=False, gamma_1=1.0, gamma_2=1.0, gamma_3=1.0):
    losses, preds, labels, masks = [], [], [], []

    assert not train or optimizer!=None
    if train:
        model.train()
    else:
        model.eval()

    for data in dataloader:
        if train:
            optimizer.zero_grad()

        batch_tensors = [d.cuda() for d in data[:-1]] if cuda else data[:-1]
        if len(batch_tensors) == 7:
            textf, visuf, acouf, knowledgef, qmask, umask, label = batch_tensors
        else:
            if args.use_knowledge:
                raise ValueError("已启用 --use-knowledge，但当前 batch 未返回 knowledge 张量，已停止运行以避免静默跳过知识模块。")
            textf, visuf, acouf, qmask, umask, label = batch_tensors
            knowledgef = None
        if (not args.use_knowledge) and knowledgef is not None:
            raise ValueError("当前 batch 返回了 knowledge 张量，但命令行未启用 --use-knowledge，请检查数据流配置。")
        qmask = qmask.permute(1, 0, 2)
        lengths = [(umask[j] == 1).nonzero().tolist()[-1][0] + 1 for j in range(len(umask))]

        log_prob1, log_prob2, log_prob3, all_log_prob, all_prob, \
        kl_log_prob1, kl_log_prob2, kl_log_prob3, kl_all_prob = model(
            textf, visuf, acouf, umask, qmask, lengths, knowledge=knowledgef
        )
        
        lp_1 = log_prob1.view(-1, log_prob1.size()[2])
        lp_2 = log_prob2.view(-1, log_prob2.size()[2])
        lp_3 = log_prob3.view(-1, log_prob3.size()[2])
        lp_all = all_log_prob.view(-1, all_log_prob.size()[2])
        labels_ = label.view(-1)

        kl_lp_1 = kl_log_prob1.view(-1, kl_log_prob1.size()[2])
        kl_lp_2 = kl_log_prob2.view(-1, kl_log_prob2.size()[2])
        kl_lp_3 = kl_log_prob3.view(-1, kl_log_prob3.size()[2])
        kl_p_all = kl_all_prob.view(-1, kl_all_prob.size()[2])
        
        loss = gamma_1 * loss_function(lp_all, labels_, umask) + \
                gamma_2 * (loss_function(lp_1, labels_, umask) + loss_function(lp_2, labels_, umask) + loss_function(lp_3, labels_, umask)) + \
               gamma_3 * (kl_loss(kl_lp_1, kl_p_all, umask) + kl_loss(kl_lp_2, kl_p_all, umask) + kl_loss(kl_lp_3, kl_p_all, umask))

        lp_ = all_prob.view(-1, all_prob.size()[2])

        pred_ = torch.argmax(lp_,1)
        preds.append(pred_.data.cpu().numpy())
        labels.append(labels_.data.cpu().numpy())
        masks.append(umask.view(-1).cpu().numpy())

        losses.append(loss.item()*masks[-1].sum())
        if train:
            loss.backward()
            if args.tensorboard:
                for param in model.named_parameters():
                    writer.add_histogram(param[0], param[1].grad, epoch)
            optimizer.step()

    if preds!=[]:
        preds = np.concatenate(preds)
        labels = np.concatenate(labels)
        masks = np.concatenate(masks)
    else:
        return float('nan'), float('nan'), [], [], [], float('nan')

    valid_eval = (masks > 0) & (labels != -1)
    if not np.any(valid_eval):
        return float('nan'), float('nan'), [], [], [], float('nan')

    preds = preds[valid_eval]
    labels = labels[valid_eval]
    masks = masks[valid_eval]

    avg_loss = round(np.sum(losses)/np.sum(masks), 4)
    avg_accuracy = round(accuracy_score(labels,preds, sample_weight=masks)*100, 2)
    avg_fscore = round(f1_score(labels,preds, sample_weight=masks, average='weighted', zero_division=0)*100, 2)
    return avg_loss, avg_accuracy, labels, preds, masks, avg_fscore


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true', default=False, help='does not use GPU')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR', help='learning rate')
    parser.add_argument('--l2', type=float, default=0.00001, metavar='L2', help='L2 regularization weight')
    parser.add_argument('--dropout', type=float, default=0.5, metavar='dropout', help='dropout rate')
    parser.add_argument('--batch-size', type=int, default=16, metavar='BS', help='batch size')
    parser.add_argument('--hidden_dim', type=int, default=1024, metavar='hidden_dim', help='output hidden size')
    parser.add_argument('--n_head', type=int, default=8, metavar='n_head', help='number of heads')
    parser.add_argument('--epochs', type=int, default=2, metavar='E', help='number of epochs')
    parser.add_argument('--temp', type=int, default=1, metavar='temp', help='temp')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--tensorboard', action='store_true', default=False, help='Enables tensorboard log')
    parser.add_argument('--class-weight', action='store_true', default=True, help='use class weights')
    parser.add_argument('--Dataset', default='IEMOCAP', help='dataset to train and test')
    parser.add_argument('--use-knowledge', action='store_true', default=False, help='enable COMET knowledge enhancement')
    parser.add_argument('--use-graph', action='store_true', default=False, help='enable graph propagation enhancement')
    parser.add_argument('--knowledge-dim', type=int, default=1024, help='dimension of COMET features')
    parser.add_argument('--graph-layers', type=int, default=2, help='number of residual graph layers')
    parser.add_argument('--temporal-window', type=int, default=2, help='temporal window for heterogeneous graph edges')
    parser.add_argument('--iemocap-path', default=None, help='path to iemocap_multimodal_features.pkl')
    parser.add_argument('--knowledge-path', default=None, help='path to iemocap_features_comet.pkl')

    args = parser.parse_args()

    if args.use_knowledge and args.Dataset != 'IEMOCAP':
        raise ValueError("第三阶段 `7/` 目录中的知识增强目前只支持 IEMOCAP。若启用 --use-knowledge，请使用 --Dataset IEMOCAP。")
    if args.use_graph and not args.use_knowledge:
        raise ValueError("`7/` 的异构知识图需要显式 knowledge node，因此启用 --use-graph 时必须同时启用 --use-knowledge。")

    if args.use_knowledge and args.Dataset == 'IEMOCAP':
        root_dir = _project_root()
        resolved_knowledge_path = _resolve_existing_path(
            args.knowledge_path,
            [
                os.path.join(root_dir, "iemocap_features_comet.pkl"),
                os.path.join(root_dir, "data", "iemocap_features_comet.pkl"),
                os.path.join(os.path.dirname(__file__), "data", "iemocap_features_comet.pkl"),
            ],
        )
        inferred_dim = infer_comet_feature_dim(resolved_knowledge_path)
        if args.knowledge_dim != inferred_dim:
            print(
                "检测到知识特征维度为 {}，覆盖参数 --knowledge-dim={}。知识文件: {}".format(
                    inferred_dim, args.knowledge_dim, resolved_knowledge_path
                )
            )
            args.knowledge_dim = inferred_dim

    print(args)
    seed_everything(args.seed)

    args.cuda = torch.cuda.is_available() and not args.no_cuda
    if args.cuda:
        print('Running on GPU')
    else:
        print('Running on CPU')

    if args.tensorboard:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter()

    cuda = args.cuda
    n_epochs = args.epochs
    batch_size = args.batch_size
    feat2dim = {'IS10':1582, 'denseface':342, 'MELD_audio':300}
    D_audio = feat2dim['IS10'] if args.Dataset=='IEMOCAP' else feat2dim['MELD_audio']
    D_visual = feat2dim['denseface']
    D_text = 1024

    D_m = D_audio + D_visual + D_text

    n_speakers = 9 if args.Dataset=='MELD' else 2
    n_classes = 7 if args.Dataset=='MELD' else 6 if args.Dataset=='IEMOCAP' else 1

    print('temp {}'.format(args.temp))
    print('seed {}'.format(args.seed))
    print('knowledge enhancement: {}'.format(args.use_knowledge))
    print('graph enhancement: {}'.format(args.use_graph))
    print('stage: 3 (conflict-guided heterogeneous knowledge graph)')

    model = Transformer_Based_Model(args.Dataset, args.temp, D_text, D_visual, D_audio, args.n_head,
                                        n_classes=n_classes,
                                        hidden_dim=args.hidden_dim,
                                        n_speakers=n_speakers,
                                        dropout=args.dropout,
                                        use_knowledge=args.use_knowledge,
                                        use_graph=args.use_graph,
                                        knowledge_dim=args.knowledge_dim,
                                        graph_layers=args.graph_layers,
                                        temporal_window=args.temporal_window)

    total_params = sum(p.numel() for p in model.parameters())
    print('total parameters: {}'.format(total_params))
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('training parameters: {}'.format(total_trainable_params))

    if cuda:
        model.cuda()
        
    kl_loss = MaskedKLDivLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)

    if args.Dataset == 'MELD':
        loss_function = MaskedNLLLoss()
        train_loader, valid_loader, test_loader = get_MELD_loaders(valid=0.0,
                                                                    batch_size=batch_size,
                                                                    num_workers=0)
    elif args.Dataset == 'IEMOCAP':
        loss_weights = torch.FloatTensor([1/0.086747,
                                        1/0.144406,
                                        1/0.227883,
                                        1/0.160585,
                                        1/0.127711,
                                        1/0.252668])
        loss_function = MaskedNLLLoss(loss_weights.cuda() if cuda else loss_weights)
        train_loader, valid_loader, test_loader = get_IEMOCAP_loaders(valid=0.0,
                                                                      batch_size=batch_size,
                                                                      num_workers=0,
                                                                      data_path=args.iemocap_path,
                                                                      knowledge_path=args.knowledge_path,
                                                                      use_knowledge=args.use_knowledge)
    else:
        print("There is no such dataset")

    best_fscore, best_loss, best_label, best_pred, best_mask = None, None, None, None, None
    all_fscore, all_acc, all_loss = [], [], []
    for e in range(n_epochs):
        if hasattr(model, "reset_runtime_debug_stats"):
            model.reset_runtime_debug_stats()
        start_time = time.time()

        train_loss, train_acc, _, _, _, train_fscore = train_or_eval_model(model, loss_function, kl_loss, train_loader, e, optimizer, True)
        valid_loss, valid_acc, _, _, _, valid_fscore = train_or_eval_model(model, loss_function, kl_loss, valid_loader, e)
        test_loss, test_acc, test_label, test_pred, test_mask, test_fscore = train_or_eval_model(model, loss_function, kl_loss, test_loader, e)
        all_fscore.append(test_fscore)

        if best_fscore == None or best_fscore < test_fscore:
            best_fscore = test_fscore
            best_label, best_pred, best_mask = test_label, test_pred, test_mask

        if args.tensorboard:
            writer.add_scalar('test: accuracy', test_acc, e)
            writer.add_scalar('test: fscore', test_fscore, e)
            writer.add_scalar('train: accuracy', train_acc, e)
            writer.add_scalar('train: fscore', train_fscore, e)

        print('epoch: {}, train_loss: {}, train_acc: {}, train_fscore: {}, valid_loss: {}, valid_acc: {}, valid_fscore: {}, test_loss: {}, test_acc: {}, test_fscore: {}, time: {} sec'.\
                format(e+1, train_loss, train_acc, train_fscore, valid_loss, valid_acc, valid_fscore, test_loss, test_acc, test_fscore, round(time.time()-start_time, 2)))
        if args.use_knowledge or args.use_graph:
            runtime_summary = model.get_runtime_debug_summary()
            validate_runtime_summary(runtime_summary, args.use_knowledge, args.use_graph)
            if (e + 1) % 10 == 0:
                print(format_runtime_summary(runtime_summary))
        if (e+1)%10 == 0:
            print(classification_report(best_label, best_pred, sample_weight=best_mask,digits=4))
            print(confusion_matrix(best_label,best_pred,sample_weight=best_mask))


    if args.tensorboard:
        writer.close()

    print('Test performance..')
    print('F-Score: {}'.format(max(all_fscore)))
    print('F-Score-index: {}'.format(all_fscore.index(max(all_fscore)) + 1))

    print(classification_report(best_label, best_pred, sample_weight=best_mask,digits=4))
    print(confusion_matrix(best_label,best_pred,sample_weight=best_mask))

