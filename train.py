
import argparse
import datetime
import os
import shutil
import torch
import copy
import numpy as np
from tensorboardX import SummaryWriter

from utils import data_manager
from utils.transforms import build_transforms
from utils.data_manager import ImageDatasetHpMask, ImageDataset
from utils.samplers import RandomIdentitySamplerHp
from utils.collate_batch import train_collate, test_collate, origin_test_collate

from utils import model_manager
from utils import model_manager_ft
from utils import model_manager_face
from utils.losses import make_loss, make_loss_face
from utils.inference import inference_prcc_global, inference_prcc_visual_rank, inference_prcc_visual_rank_easy
from utils.optimizer import make_optimizer_with_triplet, WarmupMultiStepLR
from torch.utils.data import DataLoader



def main(cfg):


    # loading dataset
    print('loading dataset')

    dataset = data_manager.init_dataset(name=cfg.dataset)
    num_classes = dataset.train_data_ids
    # num_test = len(dataset.query_data)

    train_transforms = build_transforms(cfg, is_train=True)
    test_transforms = build_transforms(cfg, is_train=False)

    train_loader = DataLoader(
        ImageDatasetHpMask(dataset.train_data, cfg.height, cfg.width, train_transforms),
        batch_size=cfg.batch_size, sampler=RandomIdentitySamplerHp(dataset.train_data, cfg.batch_size, 4),
        num_workers=8, collate_fn=train_collate)

    test_dataset = data_manager.init_test_dataset(name=cfg.dataset)
    num_test = len(test_dataset.query_data)
    test_loader = DataLoader(ImageDataset(test_dataset.test_data, cfg.height, cfg.width, test_transforms),
                                 batch_size=1, shuffle=False, num_workers=8, collate_fn=origin_test_collate)
    # test_loader = DataLoader(ImageDatasetHpMask(dataset.test_data, cfg.height, cfg.width, test_transforms),
    # batch_size=1, shuffle=False, num_workers=8, collate_fn=test_collate)

    # loading model
    print('loading model')

    # model = model_manager.init_model(name=cfg.model, num_classes=num_classes)
    model = model_manager_ft.init_model(name=cfg.model, class_num = num_classes)
    model = torch.nn.DataParallel(model).cuda()

    # model_face = model_manager_face.MobileFaceNet(embedding_size=num_classes)
    model_face = model_manager_ft.init_model(name='resnet50', class_num=num_classes)
    model_face = torch.nn.DataParallel(model_face).cuda()

    loss = make_loss(cfg, num_classes)
    loss_face = make_loss_face(cfg, num_classes)

    trainer = make_optimizer_with_triplet(model)
    scheduler = WarmupMultiStepLR(trainer, milestones=[40, 80], gamma=0.1, warmup_factor=1.0 / 3, warmup_iters=500,
                                  warmup_method="linear", last_epoch=-1,)

    working_dir = os.path.dirname(os.path.abspath(__file__))
    log_dir_name = str('logs/' + cfg.dataset)
    log_dir = os.path.join(working_dir, log_dir_name)
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    if cfg.test:
        # Test
        # model_wts = torch.load(os.path.join(log_dir, 'checkpoint_best.pth'))
        model_wts = torch.load('models/prccHrnetoriginal57.9.pth')
        model.load_state_dict(model_wts['state_dict'])

        mAP, cmc1 = inference_prcc_global(model, test_loader, num_test)
        start_time = datetime.datetime.now()
        start_time = '%4d:%d:%d-%2d:%2d:%2d' % (
        start_time.year, start_time.month, start_time.day, start_time.hour, start_time.minute, start_time.second)
        line = '{} - Test: cmc1: {:.1%}, mAP: {:.1%}\n'.format(start_time, cmc1, mAP)
        print(line)

    if cfg.visualize:
        # Visualize
        model_wts = torch.load(os.path.join(log_dir, 'checkpoint_best.pth'))
        # model_wts = torch.load('models/vcclothes_0.pth')
        model.load_state_dict(model_wts['state_dict'])

        home = os.path.join('logs', 'visualize', os.path.basename(log_dir))
        if cfg.easy:
            inference_prcc_visual_rank_easy(model, test_loader, num_test, home=home, show_rank=10, use_flip=True)
        else:
            inference_prcc_visual_rank(model, test_loader, num_test, home=home, show_rank=10, use_flip=True)

    if cfg.test == False and cfg.visualize == False:
        # training
        print('start training')
        train(model, model_face, train_loader, test_loader, loss, loss_face, trainer, scheduler, cfg.epochs, num_query=num_test, log_dir=log_dir)

    print('finished')

def train(model, model_face, train_loader, test_loader, loss_fn, loss_face_fn, optimizer, scheduler, num_epochs, num_query, log_dir):

    writer = SummaryWriter(log_dir=log_dir)
    acc_best = 0.0
    last_acc_val = acc_best

    use_cuda = torch.cuda.is_available()

    for epoch in range(num_epochs):
        model.train()
        model_face.train()

        for ii, (img, target, path, mask, face) in enumerate(
                train_loader):  # [64, 3, 256, 128],  [64,],  [64,], [64, 6, 256,b   128]
            img = img.cuda() if use_cuda else img  # [64, 3, 256, 128]
            target = target.cuda() if use_cuda else target  # [64,]
            face = face.cuda() if use_cuda else face
            b, c, h, w = img.shape

            # mask = mask.cuda() if use_cuda else mask  # [64, 6, 256, 128]
            # mask_i = mask.unsqueeze(dim=1)
            # mask_i = mask_i.expand_as(img)
            # img_a = copy.deepcopy(img)  # [40, 3, 256, 128]
            '''
            index = np.random.permutation(b)
            img_r = img[index]  # [64, 3, 256, 128]
            msk_r = mask_i[index]  # [64, 6, 256, 128]
            img_a[mask_i == 5] = img_r[msk_r == 5]

            index = np.random.permutation(b)
            img_r = img[index]  # [64, 3, 256, 128]
            msk_r = mask_i[index]  # [64, 6, 256, 128]
            img_a[mask_i == 6] = img_r[msk_r == 6]

            index = np.random.permutation(b)
            img_r = img[index]  # [64, 3, 256, 128]
            msk_r = mask_i[index]  # [64, 6, 256, 128]
            img_a[mask_i == 7] = img_r[msk_r == 7]

            index = np.random.permutation(b)
            img_r = img[index]  # [64, 3, 256, 128]
            msk_r = mask_i[index]  # [64, 6, 256, 128]
            img_a[mask_i == 9] = img_r[msk_r == 9]

            index = np.random.permutation(b)
            img_r = img[index]  # [64, 3, 256, 128]
            msk_r = mask_i[index]  # [64, 6, 256, 128]
            img_a[mask_i == 10] = img_r[msk_r == 10]

            index = np.random.permutation(b)
            img_r = img[index]  # [64, 3, 256, 128]
            msk_r = mask_i[index]  # [64, 6, 256, 128]
            img_a[mask_i == 12] = img_r[msk_r == 12]
            '''

            '''
            img_a[mask_i == 0] = img_r[msk_r == 0]  # background
            img_a[mask_i == 1] = img_r[msk_r == 1]  # hat
            img_a[mask_i == 2] = img_r[msk_r == 2]  # hair
            img_a[mask_i == 3] = img_r[msk_r == 3]  # glove
            img_a[mask_i == 4] = img_r[msk_r == 4]  # sunglasses
            img_a[mask_i == 5] = img_r[msk_r == 5]  # upper cloth
            img_a[mask_i == 6] = img_r[msk_r == 6]  # dress
            img_a[mask_i == 7] = img_r[msk_r == 7]  # coat
            img_a[mask_i == 8] = img_r[msk_r == 8]  # sock
            img_a[mask_i == 9] = img_r[msk_r == 9]  # pants
            img_a[mask_i == 10] = img_r[msk_r == 10]  # jumpsuit
            img_a[mask_i == 11] = img_r[msk_r == 11]  # scarf
            img_a[mask_i == 12] = img_r[msk_r == 12]  # skirt
            img_a[mask_i == 13] = img_r[msk_r == 13]  # face
            img_a[mask_i == 14] = img_r[msk_r == 14]  # left leg
            img_a[mask_i == 15] = img_r[msk_r == 15]  # right leg
            img_a[mask_i == 16] = img_r[msk_r == 16]  # left arm
            img_a[mask_i == 17] = img_r[msk_r == 17]  # right arm
            img_a[mask_i == 18] = img_r[msk_r == 18]  # left shoe
            img_a[mask_i == 19] = img_r[msk_r == 19]  # right shoe

            # img_a[mask_i == 0] = 0
            '''

            # img_c = torch.cat([img, img_a], dim=0)  # [80, 3, 256, 128]
            # target_c = torch.cat([target, target], dim=0)
            # score, feat = model(img_c)  # [64, 150], [64, 2018]
            score, feat = model(img)
            # score_face, feat_face = model_face(face)
            # loss_x, loss_t, loss_f, loss_x_face = loss_fn(score, feat, score_face, feat_face, target_c, target)
            # loss = loss_x + loss_t + loss_f + 0.5 * loss_x_face
            loss = loss_fn(score, feat, target)

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # compute acc
        # acc = (score.max(1)[1] == target_c).float().mean()
        acc = (score.max(1)[1] == target).float().mean()
        loss = float(loss)
        # loss_x = float(loss_x)
        # loss_t = float(loss_t)
        # loss_f = float(loss_f)
        # loss_x_face = float(loss_x_face)
        acc = float(acc)

        start_time = datetime.datetime.now()
        start_time = '%d/%d-%2d:%2d' % (start_time.month, start_time.day, start_time.hour, start_time.minute)
        # print('{} - epoch: {}  Loss: {:.04f}  Loss_x: {:.04f}  Loss_t: {:.04f}  Loss_f: {:.04f}  Loss_x_face: {:.04f}  Acc: {:.1%}  '.format(start_time, epoch, loss, loss_x, loss_t, loss_f, loss_x_face, acc))
        print('lr:', optimizer.state_dict()['param_groups'][0]['lr'])
        print('{} - epoch: {}  Loss: {:.04f}  Acc: {:.1%}  '.format(start_time, epoch, loss, acc))

        if epoch % 5 == 0:
            # test & save model
            mAP, cmc1 = inference_prcc_global(model, test_loader, num_query)
            start_time = datetime.datetime.now()
            start_time = '%d/%d-%2d:%2d' % (start_time.month, start_time.day, start_time.hour, start_time.minute)
            line = '{} - cmc1: {:.1%} mAP: {:.1%}\n'.format(start_time, cmc1, mAP)
            print(line)
            f = open(os.path.join(log_dir, 'logs.txt'), 'a')
            f.write(line)
            f.close()

            acc_test = 0.5 * (cmc1 + mAP)
            is_best = acc_test >= last_acc_val
            save_checkpoint({
                'state_dict': model.state_dict(),
                'epoch': epoch + 1,
                'best_acc': acc_test,
            }, is_best, fpath=log_dir)
            if is_best:
                last_acc_val = acc_test

            writer.add_scalar('train_loss', float(loss), epoch + 1)
            writer.add_scalar('test_rank1', float(cmc1), epoch + 1)
            writer.add_scalar('test_mAP', float(mAP), epoch + 1)

        scheduler.step()

    # Test
    last_model_wts = torch.load(os.path.join(log_dir, 'checkpoint_best.pth'))
    model.load_state_dict(last_model_wts['state_dict'])
    mAP, cmc1 = inference_prcc_global(model, test_loader, num_query)

    start_time = datetime.datetime.now()
    start_time = '%4d:%d:%d-%2d:%2d:%2d' % (start_time.year, start_time.month, start_time.day, start_time.hour, start_time.minute, start_time.second)
    line = '{} - Final: cmc1: {:.1%}, mAP: {:.1%}\n'.format(start_time, cmc1, mAP)
    print(line)
    f = open(os.path.join(log_dir, 'logs.txt'), 'a')
    f.write(line)
    f.close()


def save_checkpoint(state, is_best, fpath):
    if not os.path.exists(fpath):
        os.makedirs(fpath)

    fpath = os.path.join(fpath, 'checkpoint.pth')
    torch.save(state, fpath)
    if is_best:
        shutil.copy(fpath, os.path.join(os.path.dirname(fpath), 'checkpoint_best.pth')) 

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Train")

    parser.add_argument('--dataset', type=str, default='vcclothes')
    parser.add_argument('--model', type=str, default='hr')
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--height', type=int, default=256)
    parser.add_argument('--width', type=int, default=128)
    parser.add_argument('--test', action='store_true', help='test')
    parser.add_argument('--visualize', action='store_true', help='visualize')
    parser.add_argument('--easy', action='store_true', help='easy visualize')

    cfg = parser.parse_args()

    main(cfg)
