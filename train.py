import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import utils
from dataset.PH2 import PH2
from model import CoPA
from dataset.concept_dataset import PH2_dict
from optparse import OptionParser


def get_parser():
    parser = OptionParser()
    parser.add_option('-e', '--epochs', dest='epochs', default=150,
                      type='int', help='number of epochs')
    parser.add_option('-b', '--batch_size', dest='batch_size', default=128,
                      type='int', help='batch size')
    parser.add_option('-l', '--lr', dest='lr', default=1e-4,
                      type='float', help='learning rate')
    parser.add_option('--warmup_epoch', dest='warmup_epoch', default=5, type='int')
    parser.add_option('--optimizer', dest='optimizer', default='adamw', type='str')
    parser.add_option('--load', type='str', dest='load', default=False,
                      help='load pretrained model')
    parser.add_option('--workdir', type='str', dest='work_dir',
                      default='./work_dir', help='work dir')
    parser.add_option('--model', type='str', dest='model',
                      default='CoPA', help='name of model ')
    parser.add_option('--dataset', type='str', dest='dataset',
                      default='PH2', help='name of dataset')
    parser.add_option('--datapath', type='str', dest='data_path',
                      default='./dataset/PH2/', help='path of the dataset')
    parser.add_option('--unique_name', type='str', dest='unique_name',
                      default='train', help='name prefix')
    parser.add_option('--save_epoch', type='int', dest='save_epoch', default=50)
    parser.add_option('--seed', type='int', dest='seed', default=304)

    (config, args) = parser.parse_args()
    config.log_path = os.path.join(config.work_dir, config.dataset, config.unique_name, 'log')
    config.cp_path = os.path.join(config.work_dir, config.dataset, config.unique_name, 'checkpoint')

    return config


def load_transforms(config, split='train'):
    if split == 'train':
        img_transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.RandomResizedCrop(size=(224, 224),
                                         scale=(0.75, 1.0),
                                         ratio=(0.75, 1.33),
                                         interpolation=utils.get_interpolation_mode('bicubic')
                                         ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
        ])
    else:
        img_transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
        ])
    return img_transforms


def load_dataset(config, train_transforms, val_transforms):
    dataset_dict = {'PH2': PH2}
    train_set = dataset_dict[config.dataset](config.data_path,
                                             mode='train',
                                             transforms=train_transforms,
                                             config=config)
    train_loader = DataLoader(train_set,
                              batch_size=config.batch_size,
                              shuffle=True,
                              num_workers=14,
                              drop_last=False)

    val_set = dataset_dict[config.dataset](config.data_path,
                                           mode='test',
                                           transforms=val_transforms,
                                           config=config)
    val_loader = DataLoader(val_set,
                            batch_size=config.batch_size,
                            shuffle=False,
                            num_workers=14,
                            drop_last=False)

    return train_loader, val_loader


def load_optimizer(config, model):
    if config.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=config.lr, momentum=0.9, weight_decay=0.0005)
    elif config.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=config.lr)
    elif config.optimizer == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=1e-4)
    else:
        raise RuntimeError('Cannot find specified optimizer')
    return optimizer


def validation(model, dataloader, criterion):
    model.eval()

    losses_cls = 0
    losses_concepts = 0

    pred_list = np.zeros(0, dtype=np.uint8)
    gt_list = np.zeros(0, dtype=np.uint8)
    pred_concept_dict, gt_concept_dict = {}, {}

    with torch.no_grad():
        for i, (data, label, concept_label) in enumerate(dataloader):
            data, label, concept_label = data.cuda(), label.long().cuda(), concept_label.long().cuda()
            cls_logits, image_logits_dict = model(data)

            for index, key in enumerate(image_logits_dict.keys()):
                _, pred_concept = torch.max(image_logits_dict[key], dim=1)
                gt_concept = concept_label[:, index]
                if key not in pred_concept_dict:
                    pred_concept_dict[key] = []
                    gt_concept_dict[key] = []
                pred_concept_dict[key] += pred_concept.cpu().tolist()
                gt_concept_dict[key] += gt_concept.cpu().tolist()

            loss_cls = criterion(cls_logits, label)
            losses_cls += loss_cls.item()

            tmp_loss_concepts = 0
            for idx, key in enumerate(model.concept_token_dict.keys()):
                image_concept_loss = F.cross_entropy(image_logits_dict[key], concept_label[:, idx])
                tmp_loss_concepts += image_concept_loss.item()

            losses_concepts += tmp_loss_concepts / len(list(model.concept_token_dict.keys()))

            _, label_pred = torch.max(cls_logits, dim=1)

            pred_list = np.concatenate((pred_list, label_pred.cpu().numpy().astype(np.uint8)), axis=0)
            gt_list = np.concatenate((gt_list, label.cpu().numpy().astype(np.uint8)), axis=0)

    correct = np.sum(gt_list == pred_list)
    acc = 100 * correct / len(pred_list)

    pred_concept_list = []
    gt_concept_list = []
    for key in pred_concept_dict.keys():
        pred_concept_list += pred_concept_dict[key]
        gt_concept_list += gt_concept_dict[key]

    correct_concept = np.sum(np.array(pred_concept_list) == np.array(gt_concept_list))
    acc_concept = 100 * correct_concept / len(pred_concept_list)

    return acc, acc_concept, losses_cls / (i + 1), losses_concepts / (i + 1)


def train(model, config):
    print(config.unique_name)
    if not os.path.isdir(config.cp_path):
        os.mkdir(config.cp_path)
    if not os.path.isdir(config.log_path):
        os.mkdir(config.log_path)

    train_transforms = load_transforms(config, split='train')
    val_transforms = load_transforms(config, split='val')

    trainLoader, valLoader = load_dataset(config, train_transforms, val_transforms)

    writer = SummaryWriter(config.log_path)
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = load_optimizer(config, model)

    val_acc, val_concept_acc, _, _ = validation(model, valLoader, criterion)
    print('Acc: %.5f, Concept Acc: %.5f' % (val_acc, val_concept_acc))

    best_acc, best_concept_acc = 0, 0
    best_acc_epoch, best_concept_acc_epoch = -1, -1

    for epoch in range(config.epochs):
        print('Starting epoch {}/{}'.format(epoch + 1, config.epochs))
        epoch_loss_cls, epoch_loss_concept = 0, 0

        model.train()
        exp_scheduler = utils.exp_lr_scheduler_with_warmup(optimizer,
                                                           init_lr=config.lr,
                                                           epoch=epoch,
                                                           warmup_epoch=config.warmup_epoch,
                                                           max_epoch=config.epochs)

        for i, (data, label, concept_label) in enumerate(trainLoader, 0):
            x, target, concept_label = data.float().cuda(), label.long().cuda(), concept_label.long().cuda()

            optimizer.zero_grad()

            cls_logits, image_logits_dict = model(x)

            loss_cls = criterion(cls_logits, target)
            loss_concepts = 0
            for idx, key in enumerate(model.concept_token_dict.keys()):
                image_concept_loss = F.cross_entropy(image_logits_dict[key], concept_label[:, idx])
                loss_concepts += image_concept_loss

            loss = loss_cls + 0.1 * (loss_concepts / len(model.concept_token_dict.keys()))

            loss.backward()
            optimizer.step()

            epoch_loss_cls += loss_cls.item()
            epoch_loss_concept += loss_concepts.item()
            print(i, 'loss_cls: %.5f, loss_concept: %.5f' % (loss_cls.item(), loss_concepts.item()))

        print('[epoch %d] epoch loss_cls: %.5f, epoch_loss_concept: %.5f' % (epoch + 1,
                                                                             epoch_loss_cls / (i + 1),
                                                                             epoch_loss_concept / (i + 1)))
        writer.add_scalar('Train/Loss_cls', epoch_loss_cls / (i + 1), epoch + 1)
        writer.add_scalar('Train/Loss_concept', epoch_loss_concept / (i + 1), epoch + 1)
        if (epoch + 1) % config.save_epoch == 0:
            torch.save(model.state_dict(), '%s/CP%d.pth' % (config.cp_path, epoch + 1))

        val_acc, val_concept_acc, val_loss_cls, val_loss_concept = validation(model, valLoader, criterion)
        writer.add_scalar('Val/Acc', val_acc, epoch + 1)
        writer.add_scalar('Val/concept_acc', val_concept_acc, epoch + 1)
        writer.add_scalar('Val/val_loss_cls', val_loss_cls, epoch + 1)
        writer.add_scalar('Val/val_loss_concept', val_loss_concept, epoch + 1)

        lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('LR/lr', lr, epoch + 1)

        if val_acc > best_acc:
            best_acc = val_acc
            best_acc_epoch = epoch + 1
            torch.save(model.state_dict(), '%s/best_acc.pth' % config.cp_path)
            print('save new best checkpoint')
        if val_concept_acc > best_concept_acc:
            best_concept_acc = val_concept_acc
            best_concept_acc_epoch = epoch + 1
            torch.save(model.state_dict(), '%s/best_con.pth' % config.cp_path)
            print('save new concept best checkpoint')

        print('Acc: %.5f/best ACC: %.5f, best acc epoch: %d' % (val_acc, best_acc, best_acc_epoch))
        print('Concept Acc: %.5f/best Concept Acc: %.5f, best concept acc epoch: %d' % (val_concept_acc, best_concept_acc, best_concept_acc_epoch))
        print()


def main():
    config = get_parser()
    unique_path = os.path.join(config.work_dir, config.dataset, config.unique_name)
    if not os.path.exists(unique_path):
        os.mkdir(unique_path)
    seed = config.seed
    print(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    num_class_dict = {'PH2': 2}
    config.num_class = num_class_dict[config.dataset]
    concept_dict = {'PH2': PH2_dict}
    concept_list = concept_dict[config.dataset]

    model = CoPA(concept_list=concept_list, model_name='biomedclip', config=config)
    if config.load:
        model.load_state_dict(torch.load(config.load), strict=False)
        print('Model loaded from {}'.format(config.load))
    model.cuda()

    train(model, config)
    print(f'{config.unique_name} is done')


if __name__ == '__main__':
    main()
