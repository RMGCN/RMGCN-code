import argparse
import os

from openea.modules.load.kg import KG
from openea.modules.load.read import read_relation_triples, read_links

from rmgcn import RMGCN
from utils import read_items, MyKGs

parser = argparse.ArgumentParser(description='NullEA')
parser.add_argument('--training_data', type=str, default='DBP2.0/zh_en/')
parser.add_argument('--output', type=str, default='output/results/')
parser.add_argument('--dataset_division', type=str, default='splits')

parser.add_argument('--align_direction', type=str, default='left', choices=['left', 'right'])
parser.add_argument('--detection_mode', type=str, default='margin', choices=['classification', 'margin',
                                                                             'open', 'none'])
parser.add_argument('--alignment_module', type=str, default='mapping', choices=['mapping'])

parser.add_argument('--batch_size', type=int, default=10240)
parser.add_argument('--layer_dims', type=int, default=[512, 384, 256])

parser.add_argument('--min_rel_win', type=int, default=10)
parser.add_argument('--rel_param', type=float, default=0.1)

parser.add_argument('--num_features_nonzero', type=float, default=0.0)
parser.add_argument('--dropout', type=float, default=0.0)

parser.add_argument('--neg_margin', type=float, default=1.4)
parser.add_argument('--neg_margin_balance', type=float, default=0.05)
parser.add_argument('--neg_triple_num', type=int, default=30)
parser.add_argument('--truncated_epsilon', type=float, default=0.995)

parser.add_argument('--num_layers', type=int, default=2)
parser.add_argument('--distance_margin', type=float, default=0.2)

parser.add_argument('--learning_rate', type=float, default=0.001)
parser.add_argument('--optimizer', type=str, default='Adam')
parser.add_argument('--batch_threads_num', type=int, default=1)
parser.add_argument('--test_threads_num', type=int, default=1)
parser.add_argument('--max_epoch', type=int, default=35)
parser.add_argument('--eval_freq', type=int, default=5)

parser.add_argument('--ordered', type=bool, default=True)
parser.add_argument('--top_k', type=list, default=[1, 5, 10])
parser.add_argument('--csls', type=int, default=5)

parser.add_argument('--is_save', type=bool, default=False)
parser.add_argument('--eval_norm', type=bool, default=True)
parser.add_argument('--start_valid', type=int, default=0)
parser.add_argument('--stop_metric', type=str, default='mrr', choices=['hits1', 'mrr'])
parser.add_argument('--eval_metric', type=str, default='inner', choices=['inner', 'cosine', 'euclidean', 'manhattan','softmax','softmax2'])

parser.add_argument('--use_NCA_loss', default=True, action='store_true')
parser.add_argument('--NCA_alpha', type=float, default=15.0) 
parser.add_argument('--NCA_beta', type=float, default=10.0)

parser.add_argument('--test_batch_num', type=int, default=1)

parser.add_argument('--resume', default=False, action='store_true')
parser.add_argument('--resume_path', type=str, default="")

args = parser.parse_args()
print(args)


def read_kgs_from_folder(training_data_folder, division, mode, ordered, direction):
    kg1_relation_triples, _, _ = read_relation_triples(training_data_folder + 'rel_triples_1')
    kg2_relation_triples, _, _ = read_relation_triples(training_data_folder + 'rel_triples_2')

    train_links = read_links(os.path.join(training_data_folder, division, 'train_links'))
    valid_links = read_links(os.path.join(training_data_folder, division, 'valid_links'))
    test_links = read_links(os.path.join(training_data_folder, division, 'test_links'))

    train_unlinked_ent1 = read_items(os.path.join(training_data_folder, division, 'train_unlinked_ent1'))
    valid_unlinked_ent1 = read_items(os.path.join(training_data_folder, division, 'valid_unlinked_ent1'))
    test_unlinked_ent1 = read_items(os.path.join(training_data_folder, division, 'test_unlinked_ent1'))

    train_unlinked_ent2 = read_items(os.path.join(training_data_folder, division, 'train_unlinked_ent2'))
    valid_unlinked_ent2 = read_items(os.path.join(training_data_folder, division, 'valid_unlinked_ent2'))
    test_unlinked_ent2 = read_items(os.path.join(training_data_folder, division, 'test_unlinked_ent2'))

    kg1 = KG(kg1_relation_triples, set())
    kg2 = KG(kg2_relation_triples, set())

    if direction == "left":
        two_kgs = MyKGs(kg1, kg2, train_links, test_links,
                        train_unlinked_ent1, valid_unlinked_ent1, test_unlinked_ent1,
                        train_unlinked_ent2, valid_unlinked_ent2, test_unlinked_ent2,
                        valid_links=valid_links, mode=mode, ordered=ordered)
    else:
        assert direction == "right"
        train_links_rev = [(e2, e1) for e1, e2 in train_links]
        test_links_rev = [(e2, e1) for e1, e2 in test_links]
        valid_links_rev = [(e2, e1) for e1, e2 in valid_links]
        two_kgs = MyKGs(kg2, kg1, train_links_rev, test_links_rev,
                        train_unlinked_ent2, valid_unlinked_ent2, test_unlinked_ent2,
                        train_unlinked_ent1, valid_unlinked_ent1, test_unlinked_ent1,
                        valid_links=valid_links_rev, mode=mode, ordered=ordered)
    return two_kgs


if __name__ == '__main__':
    kgs = read_kgs_from_folder(args.training_data, args.dataset_division, args.alignment_module,
                               args.ordered, args.align_direction)
    model = RMGCN()
    model.set_args(args)
    model.set_kgs(kgs)
    model.init()
    model.run()
    model.test()
