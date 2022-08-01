import torch

import os
import time
import json
import random
import numpy as np
from collections import defaultdict
import hydra

from models.nav_decoder.utils import timeSince, read_img_features, print_progress
import utils
from models.nav_decoder.env import R2RBatch
from models.nav_decoder.agent import GVUENavAgent
from models.nav_decoder.eval import R2REvaluation

import warnings
warnings.filterwarnings("ignore")
from torch.utils.tensorboard import SummaryWriter

from models.nav_decoder.vlnbert.vlnbert_init import get_tokenizer

is_ResNet = None

@hydra.main(config_path="../configs", config_name='r2r')
def main(cfg):
    log_dir = cfg.train.log.dir
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    if cfg.model.v_sizes[-1] == 2048:
        is_ResNet = True
    elif cfg.model.v_sizes[-1] == 768:
        is_ResNet = False
    else:
        assert False, "Only supports ResNet features (2048) or ViT features (768), to add more, change VLNBert's visual MLP encoder"
    
    if cfg.train.type in ['listener', 'validlistener']:
        train_val(cfg)
    elif cfg.train.type == 'auglistener':
        train_val_augment(cfg)
    else:
        assert False


def train_val(cfg):
    ''' Train on the training set, and validate on seen and unseen splits. '''
    setup()
    tok = get_tokenizer(cfg)

    feat_dict = read_img_features(cfg.train.data.v_feature, test_only=cfg.test_only)

    if cfg.test_only:
        featurized_scans = None
        val_env_names = ['val_train_seen']
    else:
        featurized_scans = set([key.split("_")[0] for key in list(feat_dict.keys())])
        val_env_names = ['val_train_seen', 'val_seen', 'val_unseen']

    train_env = R2RBatch(cfg, feat_dict, batch_size=cfg.train.setting.batch_size, splits=['train'], tokenizer=tok)
    from collections import OrderedDict

    if cfg.submit:
        val_env_names.append('test')
    else:
        pass

    val_envs = OrderedDict(
        ((split,
          (R2RBatch(cfg, feat_dict, batch_size=cfg.train.setting.batch_size, splits=[split], tokenizer=tok),
           R2REvaluation(cfg, [split], featurized_scans, tok))
          )
         for split in val_env_names
         )
    )

    if cfg.train.type == 'listener':
        train(cfg, train_env, tok, cfg.train.setting.iters, val_envs=val_envs)
    elif cfg.train.type == 'validlistener':
        valid(cfg, train_env, tok, val_envs=val_envs)
    else:
        assert False

def train_val_augment(cfg):
    """
    Train the listener with the augmented data
    """
    setup()

    # Create a batch training environment that will also preprocess text
    tok_bert = get_tokenizer(cfg)

    # Load the env img features
    feat_dict = read_img_features(cfg.train.v_feature, test_only=cfg.test_only)

    if cfg.test_only:
        featurized_scans = None
        val_env_names = ['val_train_seen']
    else:
        featurized_scans = set([key.split("_")[0] for key in list(feat_dict.keys())])
        val_env_names = ['val_train_seen', 'val_seen', 'val_unseen']

    # Create the training environment
    train_env = R2RBatch(cfg, feat_dict, batch_size=cfg.train.setting.batch_size, splits=['train'], tokenizer=tok_bert)
    aug_env   = R2RBatch(cfg, feat_dict, batch_size=cfg.train.setting.batch_size, splits=[cfg.train.aug_path], tokenizer=tok_bert, name='aug')

    # Setup the validation data
    val_envs = {split: (R2RBatch(cfg, feat_dict, batch_size=cfg.train.setting.batch_size, splits=[split], tokenizer=tok_bert),
                R2REvaluation(cfg, [split], featurized_scans, tok_bert))
                for split in val_env_names}

    # Start training
    train(cfg, train_env, tok_bert, cfg.train.setting.iters, val_envs=val_envs, aug_env=aug_env)

def train(cfg, train_env, tok, n_iters, log_every=2000, val_envs={}, aug_env=None):
    writer = SummaryWriter(log_dir=cfg.train.log.dir)
    listner = GVUENavAgent(cfg, train_env, "", tok, cfg.train.setting.max_action)

    record_file = open(cfg.train.log.record_file, 'a')
    record_file.write(str(cfg) + '\n\n')
    record_file.close()

    start_iter = 0
    if cfg.train.continue_training.key:
        if cfg.train.data.aug_path is None:
            start_iter = listner.load(os.path.join(cfg.train.continue_training.path))
            print("\nLOAD the model from {}, iteration ".format(cfg.train.continue_training.path, start_iter))
        else:
            load_iter = listner.load(os.path.join(cfg.train.continue_training.path))
            print("\nLOAD the model from {}, iteration ".format(cfg.train.continue_training.path, load_iter))

    start = time.time()
    print('\nListener training starts, start iteration: %s' % str(start_iter))

    best_val = {'val_unseen': {"spl": 0., "sr": 0., "state":"", 'update':False}}

    for idx in range(start_iter, start_iter+n_iters, log_every):
        listner.logs = defaultdict(list)
        interval = min(log_every, n_iters-idx)
        iter = idx + interval

        # Train for log_every interval
        if aug_env is None:
            listner.env = train_env
            listner.train(interval, feedback=cfg.train.setting.feedback)  # Train interval iters
        else:
            jdx_length = len(range(interval // 2))
            for jdx in range(interval // 2):
                # Train with GT data
                listner.env = train_env
                listner.train(1, feedback=cfg.train.setting.feedback)

                # Train with Augmented data
                listner.env = aug_env
                listner.train(1, feedback=cfg.train.setting.feedback)

                print_progress(jdx, jdx_length, prefix='Progress:', suffix='Complete', bar_length=50)

        # Log the training stats to tensorboard
        total = max(sum(listner.logs['total']), 1)
        length = max(len(listner.logs['critic_loss']), 1)
        critic_loss = sum(listner.logs['critic_loss']) / total
        RL_loss = sum(listner.logs['RL_loss']) / max(len(listner.logs['RL_loss']), 1)
        IL_loss = sum(listner.logs['IL_loss']) / max(len(listner.logs['IL_loss']), 1)
        entropy = sum(listner.logs['entropy']) / total
        writer.add_scalar("loss/critic", critic_loss, idx)
        writer.add_scalar("policy_entropy", entropy, idx)
        writer.add_scalar("loss/RL_loss", RL_loss, idx)
        writer.add_scalar("loss/IL_loss", IL_loss, idx)
        writer.add_scalar("total_actions", total, idx)
        writer.add_scalar("max_length", length, idx)
        # print("total_actions", total, ", max_length", length)

        # Run validation
        loss_str = "iter {}".format(iter)
        for env_name, (env, evaluator) in val_envs.items():
            listner.env = env

            # Get validation distance from goal under test evaluation conditions
            listner.test(use_dropout=False, feedback='argmax', iters=None)
            result = listner.get_results()
            score_summary, _ = evaluator.score(result)
            loss_str += ", %s " % env_name
            for metric, val in score_summary.items():
                if metric in ['spl']:
                    writer.add_scalar("spl/%s" % env_name, val, idx)
                    if env_name in best_val:
                        if val > best_val[env_name]['spl']:
                            best_val[env_name]['spl'] = val
                            best_val[env_name]['update'] = True
                        elif (val == best_val[env_name]['spl']) and (score_summary['success_rate'] > best_val[env_name]['sr']):
                            best_val[env_name]['spl'] = val
                            best_val[env_name]['update'] = True
                loss_str += ', %s: %.4f' % (metric, val)

        record_file = open(os.path.join(cfg.train.log.dir, cfg.name + '.txt'), 'a')
        record_file.write(loss_str + '\n')
        record_file.close()

        for env_name in best_val:
            if best_val[env_name]['update']:
                best_val[env_name]['state'] = 'Iter %d %s' % (iter, loss_str)
                best_val[env_name]['update'] = False
                listner.save(idx, os.path.join(cfg.train.log.dir, "snap", cfg.name, "state_dict", "best_%s" % (env_name)))
            else:
                listner.save(idx, os.path.join(cfg.train.log.dir, "snap", cfg.name, "state_dict", "latest_dict"))

        print(('%s (%d %d%%) %s' % (timeSince(start, float(iter)/n_iters),
                                             iter, float(iter)/n_iters*100, loss_str)))

        if iter % 1000 == 0:
            print("BEST RESULT TILL NOW")
            for env_name in best_val:
                print(env_name, best_val[env_name]['state'])

                record_file = open(os.path.join(cfg.train.log.dir, cfg.name + '.txt'), 'a')
                record_file.write('BEST RESULT TILL NOW: ' + env_name + ' | ' + best_val[env_name]['state'] + '\n')
                record_file.close()

    listner.save(idx, os.path.join(cfg.train.log.dir, "snap", cfg.name, "state_dict", "LAST_iter%d" % (idx)))


def valid(cfg, train_env, tok, val_envs={}):
    agent = GVUENavAgent(cfg, train_env, "", tok, cfg.train.setting.max_action)

    print("Loaded the listener model at iter %d from %s" % (agent.load(cfg.eval.path), cfg.eval.path))

    for env_name, (env, evaluator) in val_envs.items():
        agent.logs = defaultdict(list)
        agent.env = env

        iters = None
        agent.test(use_dropout=False, feedback='argmax', iters=iters)
        result = agent.get_results()

        if env_name != '':
            score_summary, _ = evaluator.score(result)
            loss_str = "Env name: %s" % env_name
            for metric,val in score_summary.items():
                loss_str += ', %s: %.4f' % (metric, val)
            print(loss_str)

        if cfg.submit:
            json.dump(
                result,
                open(os.path.join(cfg.train.log.dir, "submit_%s.json" % env_name), 'w'),
                sort_keys=True, indent=4, separators=(',', ': ')
            )

def setup():
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    random.seed(0)
    np.random.seed(0)


if __name__ == "__main__":
    main()
