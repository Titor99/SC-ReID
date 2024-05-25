
import torchvision.transforms as T

def build_transforms(cfg, is_train=True):

    normalize_transform = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    if is_train:
        transform = T.Compose([
            T.Resize([cfg.height, cfg.width]),
            T.Pad(10),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomCrop([cfg.height, cfg.width]),
            T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            T.ToTensor(),
            normalize_transform,
        ])
    else:
        transform = T.Compose([
            T.Resize([cfg.height, cfg.width]),
            T.ToTensor(),
            normalize_transform,
        ])

    return transform



