import torch

def train_collate(batch):
    imgs, masks, face_imgs, pids, paths = zip(*batch)
    pids = torch.tensor(pids, dtype=torch.int64)
    return torch.stack(imgs), pids, paths, torch.cat(masks), torch.stack(face_imgs)

def test_collate(batch):
    imgs, masks, face_imgs, pids, paths = zip(*batch)
    return torch.stack(imgs, dim=0), pids, paths, torch.cat(masks, dim=0), torch.stack(face_imgs)

def origin_test_collate(batch):
    imgs, pids, paths, camid = zip(*batch)
    # index1 = 1
    # index2 = 1
    return torch.stack(imgs, dim=0), pids, paths, camid  # index1, index2