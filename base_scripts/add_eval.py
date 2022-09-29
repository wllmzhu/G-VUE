""" evaluation of Unified-IO and OFA, as additional baselines of large general-purpose models """

import hydra
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.nn.functional import interpolate
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fvcore.common.registry import Registry
from datasets.base import create_dataset
from models.uio import runner
from transformers import OFATokenizer, OFAModel
from sentence_transformers import SentenceTransformer, util
from utils.misc import collate_fn
from base_scripts.task_evaluation_lib import box_iou
from tqdm import tqdm
from einops import rearrange
from absl import logging
logging.set_verbosity(logging.INFO)

norm_means = torch.as_tensor([0.485, 0.456, 0.406])
norm_stds = torch.as_tensor([0.229, 0.224, 0.225])
rgb2grey_coef = np.array([0.299, 0.587, 0.114])

EVAL_TASK = Registry('eval_task')
BATCH_SIZE = 32


def undo_img_transform(img):
    # used for Unified-IO
    # transform a single image tensor [3, H, W] returned from __getitem__ to original array [H, W, 3]
    # as well as revert value normalization
    if img.ndim == 3:
        img = rearrange(img, 'C H W -> H W C')
    else:
        img = rearrange(img, 'B C H W -> B H W C')
    img = img.mul_(norm_stds).add_(norm_means)
    return img.cpu().numpy()


def ofa_img_transform(imgs, refexp=False):
    imgs = imgs.mul_(norm_stds.reshape(1, -1, 1, 1)).add_(norm_means.reshape(1, -1, 1, 1))
    imgs = imgs.sub_(0.5).div_(0.5)
    resolution = 512 if refexp else 480
    imgs = interpolate(imgs, size=(resolution, resolution), mode='bicubic')
    return imgs


def uio_bin2bbox(str_list, token_per_label=4):
    bboxes = []
    for txt in str_list:
        txt = txt.strip().split()
        cur = 0
        locations = None
        while cur < len(txt):
            if all(['extra_id' in str for str in txt[cur: cur+token_per_label]]):
                try:
                    locations = [int(str[10:-1])-100 for str in txt[cur: cur+token_per_label]]
                    locations = np.array(locations)
                    locations = locations.reshape((-1, 2))[:, ::-1].reshape((-1, token_per_label))   # [yx to xy]
                    locations = locations / 1000
                    bboxes.append(locations)
                except:
                    logging.info(f'parse fail: {txt[cur: cur+token_per_label]}')
                    bboxes.append(np.zeros((1, 4)))
                break
            cur += 1
        if locations is None:
            bboxes.append(np.zeros((1, 4)))
    
    return np.concatenate(bboxes, axis=0)


def ofa_bin2bbox(str_list, token_per_label=4):
    bboxes = []
    for txt in str_list:
        txt = txt.strip().split()
        try:
            locations = np.array([int(str[5:-1]) for str in txt]).reshape(-1, 4)
            locations = locations / 1000
            bboxes.append(locations)
        except:
            bboxes.append(np.zeros((1, 4)))
    return np.concatenate(bboxes, axis=0)


def compute_depth_errors(pred, gt):
    """ here pred and gt are both numpy.ndarray """
    delta = np.maximum((pred / gt), (gt / pred))
    d1 = (delta < 1.25).astype(np.float).mean()

    abs_rel = (np.abs(pred - gt) / gt).mean()

    rms = (pred - gt) ** 2
    rms = np.sqrt(rms.mean())

    return np.array([d1, abs_rel, rms])


@EVAL_TASK.register()
def eval_depth(cfg, model, uio=True, tokenizer=None, sen_encoder=None):
    assert uio is True
    dataset = create_dataset(cfg, 'test')
    dataloader = DataLoader(dataset, batch_size=1, collate_fn=collate_fn)

    min_depth, max_depth = 1e-3, 10
    metrics = []
    logging.info(f'evaluating {"Unified-IO" if uio else "OFA"} on depth')
    for batch in tqdm(dataloader):
        # batch: (imgs, txts, targets)
        img, target = batch[0][0], batch[-1][0]
        if uio:
            img = undo_img_transform(img)
            out = model.depth(img)
        else:
            pass

        pred = (rgb2grey_coef * out['rescaled_image']).sum(-1)
        pred = (pred * max_depth).clip(min_depth, max_depth)
        
        target = target.cpu().squeeze().numpy()
        valid_mask = np.logical_and(target>min_depth, target<max_depth)
        metrics.append(compute_depth_errors(pred[valid_mask], target[valid_mask]).reshape(1, -1))
    
    metrics = np.concatenate(metrics, axis=0).mean(0)
    for i, k in enumerate(['d1', 'abs_rel', 'rms']):
        logging.info(f'{k}: {metrics[i]:.4f}')

    err2per = np.exp(-1.386*metrics[1:])
    task_score = (metrics[0]+err2per.sum()) / 3 * 100
    logging.info(f'task score: {task_score:.2f}\n')
    return task_score


@EVAL_TASK.register()
def eval_vl_retrieval(cfg, model, uio=True, tokenizer=None, sen_encoder=None):
    dataset = create_dataset(cfg, 'test')
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn)
    txts = dataset.texts
    img2txt_id = dataset.img2txt
    top1 = 0
    top5 = 0
    top10 = 0
    total = 0
    logging.info(f'evaluating {"Unified-IO" if uio else "OFA"} on vl_retrieval')
    if uio:
        if sen_encoder is None:
            # answer_options mode, ignored, turn to use tokenizer
            for batch in tqdm(dataloader):
                # batch: (imgs, txts, targets)
                img, _, target = batch
                txt_gt_ids = np.array(img2txt_id[target])
            
                img = undo_img_transform(img)
                out = model.run(
                    input_images=[img],
                    input_texts=['What does the image describe ?'],
                    generate_image=False, num_decodes=None,
                    answer_options=txts, top_k=10
                )
                is_gt = np.in1d(out['topk_indices'][0], txt_gt_ids, assume_unique=True)
                gt_indices = is_gt.nonzero()[0]
                if gt_indices.size > 0:
                    highest_rank = gt_indices[0]
                    if highest_rank == 0:
                        top1 += 1
                    if highest_rank < 5:
                        top5 += 1
                    if highest_rank < 10:
                        top10 += 1
                total += 1
                # periodical check
                if total % 50 == 0:
                    logging.info(f'total: {total} | top1: {top1} | top5: {top5} | top10: {top10}')
        else:
            pred_embs = []
            # use sentence similarity
            for batch in tqdm(dataloader):
                # batch: (imgs, txts, targets)
                imgs, _, targets = batch
                imgs = undo_img_transform(imgs)
                out = model.run(
                    input_images=imgs,
                    input_texts=['What does the image describe ?']*len(targets),
                    generate_image=False, num_decodes=None,
                )
                pred_embs.append(sen_encoder.encode(out['text']))
            
            pred_embs = np.concatenate(pred_embs, axis=0)
            cand_embs = sen_encoder.encode(txts)
            cos_sim = util.cos_sim(pred_embs, cand_embs)   # [1000, 5000]
            pred_ranks = torch.argsort(cos_sim, dim=-1, descending=True)

            highest_ranks = []
            total = pred_ranks.shape[0]
            for i in range(total):
                txt_gt_ids = img2txt_id[i]   # 5 ground truth candidates
                # get highest rank from 5 candidates
                highest_ranks.append((pred_ranks[i].apply_(lambda x: x in txt_gt_ids)).nonzero()[:, 0].numpy().min())
            
            highest_ranks = np.array(highest_ranks)
            top1 = (highest_ranks < 1).sum()
            top5 = (highest_ranks < 5).sum()
            top10 = (highest_ranks < 10).sum()
    else:
        pred_embs = []
        for batch in tqdm(dataloader):
            # batch: (imgs, txts, targets)
            imgs, _, targets = batch
            imgs = ofa_img_transform(imgs, refexp=False)
            txts = ['what does the image describe?'] * len(targets)
            out = model.generate(
                tokenizer(txts, return_tensors='pt', padding=True).input_ids.to(model.device),
                patch_images=imgs.to(model.device), num_beams=5, no_repeat_ngram_size=3
            )
            pred_cap = tokenizer.batch_decode(out, skip_special_tokens=True)
            pred_embs.append(sen_encoder.encode(pred_cap))
        
        pred_embs = np.concatenate(pred_embs, axis=0)
        cand_embs = sen_encoder.encode(txts)
        cos_sim = util.cos_sim(pred_embs, cand_embs)   # [1000, 5000]
        pred_ranks = torch.argsort(cos_sim, dim=-1, descending=True)

        highest_ranks = []
        total = pred_ranks.shape[0]
        for i in range(total):
            txt_gt_ids = img2txt_id[i]   # 5 ground truth candidates
            # get highest rank from 5 candidates
            highest_ranks.append((pred_ranks[i].apply_(lambda x: x in txt_gt_ids)).nonzero()[:, 0].numpy().min())
        
        highest_ranks = np.array(highest_ranks)
        top1 = (highest_ranks < 1).sum()
        top5 = (highest_ranks < 5).sum()
        top10 = (highest_ranks < 10).sum()
    
    logging.info(f'total: {total} | top1: {top1} | top5: {top5} | top10: {top10}')
    logging.info(f'subset: test | rec@1: {top1/total:.4f} | rec@5: {top5/total:.4f} | rec@10: {top10/total:.4f}')

    task_score = np.mean([top1, top5, top10]) / total * 100
    logging.info(f'task score: {task_score:.2f}\n')
    return task_score


@EVAL_TASK.register()
def eval_phrase_grounding(cfg, model, uio=True, tokenizer=None, sen_encoder=None):
    subsets = ['val', 'testA', 'testB']
    acc_all = []
    for subset in subsets:
        dataset = create_dataset(cfg, subset)
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn)
        correct = 0
        # parse_fail = 0
        total = 0
        logging.info(f'evaluating {"Unified-IO" if uio else "OFA"} on refexp')
        for batch in tqdm(dataloader):
            # batch: (imgs, txts, targets)
            imgs, txts, targets = batch
            txts = [f'Which region does the text " {txt} " describe ?' for txt in txts]
            if isinstance(targets, torch.Tensor):
                targets = targets.cpu().numpy()
            else:
                targets = np.array(targets)
            if uio:
                imgs = undo_img_transform(imgs)
                out = model.run(
                    imgs, txts, output_text_len=32, generate_image=False, num_decodes=None
                )
                pred_bboxes = uio_bin2bbox(out['text'])
            else:
                assert tokenizer is not None
                imgs = ofa_img_transform(imgs)
                out = model.generate(
                    tokenizer(txts, return_tensors='pt', padding=True).input_ids.to(model.device),
                    patch_images=imgs.to(model.device), num_beams=5, no_repeat_ngram_size=3
                )
                pred_bboxes = ofa_bin2bbox(tokenizer.batch_decode(out, skip_special_tokens=True))
                print(pred_bboxes)
            
            ious = box_iou(pred_bboxes, targets)
            correct += (ious >= 0.5).sum()
            total += len(targets)
            # periodical check
            if total % 128 == 0:
                logging.info(f'total: {total} | correct: {correct}')
        
        logging.info(f'total: {total} | correct: {correct}')
        acc = correct / total
        logging.info(f'subset: {subset} | acc@0.5: {acc:.4f}')
        acc_all.append(acc)

    task_score = np.mean(acc_all) * 100
    logging.info(f'task score: {task_score:.2f}\n')
    return task_score


@EVAL_TASK.register()
def eval_vqa(cfg, model, uio=True, tokenizer=None, sen_encoder=None):
    dataset = create_dataset(cfg, 'testdev')
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn)
    answers = list(dataset.answer_to_idx.keys())
    correct = 0
    total = 0
    logging.info(f'evaluating {"Unified-IO" if uio else "OFA"} on vqa')
    if uio:
        if sen_encoder is None:
            # ignored, turn to use sentence encoder
            for batch in tqdm(dataloader):
                # batch: (imgs, txts, targets)
                img, txt, target = batch[0][0], batch[1][0], batch[2][0]
                img = undo_img_transform(img)
                out = model.run(
                    input_images=[img],
                    input_texts=[txt],
                    output_text_len=32,
                    generate_image=False, num_decodes=None,
                    answer_options=answers, top_k=1
                )
                if out['topk_indices'][0, 0] == target:
                    correct += 1
                total += 1
                # periodical check
                if total % 100 == 0:
                    logging.info(f'total: {total} | correct: {correct}')
        else:
            cand_embs = sen_encoder.encode(answers)
            # use sentence similarity
            for batch in tqdm(dataloader):
                # batch: (imgs, txts, targets)
                imgs, txts, targets = batch
                imgs = undo_img_transform(imgs)
                out = model.run(
                    input_images=imgs,
                    input_texts=txts,
                    output_text_len=32,
                    generate_image=False, num_decodes=None,
                )
                pred_embs = sen_encoder.encode(out['text'])
                cos_sim = util.cos_sim(pred_embs, cand_embs)   # [num_samples, num_candidates]
                pred_ranks = torch.argsort(cos_sim, dim=-1, descending=True)
                pred_options = pred_ranks[:, 0]

                correct += (pred_options==torch.as_tensor(targets)).sum()
                total += len(targets)

                # periodical check
                if total % 128 == 0:
                    logging.info(f'total: {total} | correct: {correct}')
    else:
        pred_embs = []
        all_targets = []
        for batch in tqdm(dataloader):
            # batch: (imgs, txts, targets)
            imgs, txts, targets = batch
            imgs = ofa_img_transform(imgs, refexp=False)
            out = model.generate(
                tokenizer(txts, return_tensors='pt', padding=True).input_ids.to(model.device),
                patch_images=imgs.to(model.device), num_beams=5, no_repeat_ngram_size=3
            )
            pred_answer = tokenizer.batch_decode(out, skip_special_tokens=True)
            pred_embs.append(sen_encoder.encode(pred_answer))
            all_targets.append(targets)
        
        pred_embs = np.concatenate(pred_embs, axis=0)
        cand_embs = sen_encoder.encode(answers)
        cos_sim = util.cos_sim(pred_embs, cand_embs)   # [num_samples, num_candidates]
        pred_ranks = torch.argsort(cos_sim, dim=-1, descending=True)
        pred_options = pred_ranks[:, 0]
        
        total = len(all_targets)
        correct = (pred_options==torch.as_tensor(all_targets)).sum()
    
    logging.info(f'total: {total} | correct: {correct}')
    acc = correct / total
    logging.info(f'subset: testdev | acc: {acc:.4f}')

    task_score = acc * 100
    logging.info(f'task score: {task_score:.2f}\n')
    return task_score


@EVAL_TASK.register()
def eval_common_sense(cfg, model, uio=True, tokenizer=None, sen_encoder=None):
    dataset = create_dataset(cfg, 'val')
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn)
    correct = 0
    total = 0
    logging.info(f'evaluating {"Unified-IO" if uio else "OFA"} on common_sense')
    if uio:
        if sen_encoder is None:
            # ignored, turn to use sentence encoder
            for batch in tqdm(dataloader):
                # batch: (imgs, txts, targets)
                img, txts, target = batch[0][0], batch[1][0], batch[2][0]
                img = undo_img_transform(img)
                out = model.run(
                    input_images=[img],
                    input_texts=txts[:1],
                    generate_image=False, num_decodes=None,
                    answer_options=txts[1:], top_k=1
                )
                if out['topk_indices'][0, 0] == target:
                    correct += 1
                total += 1
                # periodical check
                if total % 100 == 0:
                    logging.info(f'total: {total} | correct: {correct}')
        else:
            for batch in tqdm(dataloader):
                # batch: (imgs, txts, targets)
                imgs, txts, targets = batch
                imgs = undo_img_transform(imgs)
                out = model.run(
                    input_images=imgs,
                    input_texts=[txt[0] for txt in txts],
                    generate_image=False, num_decodes=None
                )
                pred_embs = sen_encoder.encode(out['text'])
                for i in range(pred_embs.shape[0]):
                    pred_emb = pred_embs[i]
                    cand_embs = sen_encoder.encode(txts[i][1:])
                    cos_sim = util.cos_sim(pred_emb, cand_embs)   # [1, 4]
                    pred_ranks = torch.argsort(cos_sim, dim=-1, descending=True)
                    pred_options = pred_ranks[0, 0]
                    if pred_options == targets[i]:
                        correct += 1
                    total += 1
                # periodical check
                if total % 128 == 0:
                    logging.info(f'total: {total} | correct: {correct}')
    else:
        for batch in tqdm(dataloader):
            # batch: (imgs, txts, targets)
            imgs, txts, targets = batch
            imgs = ofa_img_transform(imgs, refexp=False)
            out = model.generate(
                tokenizer([txt[0] for txt in txts], return_tensors='pt', padding=True).input_ids.to(model.device),
                patch_images=imgs.to(model.device), num_beams=5, no_repeat_ngram_size=3
            )
            pred_answer = tokenizer.batch_decode(out, skip_special_tokens=True)
            pred_embs = sen_encoder.encode(pred_answer)
            for i in range(pred_embs.shape[0]):
                pred_emb = pred_embs[i]
                cand_embs = sen_encoder.encode(txts[i][1:])
                cos_sim = util.cos_sim(pred_emb, cand_embs)   # [1, 4]
                pred_ranks = torch.argsort(cos_sim, dim=-1, descending=True)
                pred_options = pred_ranks[0, 0]
                if pred_options == targets[i]:
                    correct += 1
                total += 1
            # periodical check
            if total % 128 == 0:
                logging.info(f'total: {total} | correct: {correct}')
    
    logging.info(f'total: {total} | correct: {correct}')
    acc = correct / total
    logging.info(f'subset: val | acc: {acc:.4f}')

    task_score = acc * 100
    logging.info(f'task score: {task_score:.2f}\n')
    return task_score


@hydra.main(config_path='../configs', config_name='base')
def main(cfg):
    fun_name = f'eval_{cfg.task.key}'
    sen_encoder = SentenceTransformer('all-MiniLM-L6-v2')
    if cfg.exp_name == 'uio':
        model = runner.ModelRunner('xl', '/scratch/huangjiangyong/data/UIO/xl_1000k.bin', max_options=100)
        # model.model.eval()   # ModelRunner.model is nn.Module rather than runner itself, flax.linen.Module has no eval()
        EVAL_TASK.get(fun_name)(cfg, model=model, uio=True, sen_encoder=sen_encoder)
    elif cfg.exp_name == 'ofa':
        tokenizer = OFATokenizer.from_pretrained('/scratch/huangjiangyong/data/OFA')
        model = OFAModel.from_pretrained('/scratch/huangjiangyong/data/OFA', use_cache=False)
        model.eval()
        model.cuda()
        EVAL_TASK.get(fun_name)(cfg, model=model, uio=False, tokenizer=tokenizer, sen_encoder=sen_encoder)
    else:
        raise ValueError('model not exist or prepared')


"""
take depth for example, run:
    python base_scripts/eval_uio.py task=depth
"""
if __name__ == '__main__':
    main()
