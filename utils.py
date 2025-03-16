import torchvision.transforms as transforms


def get_interpolation_mode(interpolation_str):
    interpolation_mapping = {
        'nearest': transforms.InterpolationMode.NEAREST,
        'lanczos': transforms.InterpolationMode.LANCZOS,
        'bilinear': transforms.InterpolationMode.BILINEAR,
        'bicubic': transforms.InterpolationMode.BICUBIC,
        'box': transforms.InterpolationMode.BOX,
        'hamming': transforms.InterpolationMode.HAMMING
    }
    return interpolation_mapping.get(interpolation_str)


def exp_lr_scheduler_with_warmup(optimizer, init_lr, epoch, warmup_epoch, max_epoch):
    if 0 <= epoch <= warmup_epoch:
        lr = init_lr * 2.718 ** (10 * (float(epoch) / float(warmup_epoch) - 1.))
        if epoch == warmup_epoch:
            lr = init_lr
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        return lr

    else:
        lr = init_lr * (1 - epoch / max_epoch) ** 0.9
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    return lr

