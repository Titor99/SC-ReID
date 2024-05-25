import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

class ImageDataset(Dataset):
    def __init__(self, dataset, height=256, width=128, transform=None):
        self.dataset = dataset
        self.transform = transform
        self.h = height
        self.w = width

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, pid, camid = self.dataset[index]
        img = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, pid, img_path, camid

transform_face = T.Compose([
            # T.Resize([50, 50]),
            T.Resize([128, 128]),
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

class ImageDatasetHpMask(Dataset):

    def __init__(self, dataset, height=256, width=128, transform=None):
        self.dataset = dataset
        self.transform = transform
        self.h = height
        self.w = width

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, pid, msk_path, face_path = self.dataset[index]
        img = Image.open(img_path).convert('RGB')
        msk = Image.open(msk_path).convert('P')
        msk = msk.resize((self.w, self.h))
        msk = np.array(msk)
        msk = torch.from_numpy(msk)
        msk = msk.unsqueeze(dim=0)
        face_img = Image.open(face_path).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)
            face_img = transform_face(face_img)

        return img, msk, face_img, pid, img_path

class CCPRF(object):

    rgb_dir = 'rgb'
    msk_dir = 'hp'
    face_dir = 'head+'  # face

    def __init__(self, name, root='../datasets/SC-ReID'):

        self.name = name
        self.root = root
        __factory = {
            'prcc': 'PRCC',
            'celeb': 'Celeb',
            'ltcc': 'LTCC',
            'ltcc+prcc': 'LTCC_PRCC',
            'celeb+last': 'Celeb_LaST',
            'vcclothes': 'VC-Clothes',
            'market': 'market1501',
        }
        self.dataset_dir = os.path.join(self.root, __factory[self.name], self.rgb_dir)
        self.mask_dir = os.path.join(self.root, __factory[self.name], self.msk_dir)
        self.face_dir = os.path.join(self.root, __factory[self.name], self.face_dir)
        self.train_dir = os.path.join(self.dataset_dir, 'train')
        # self.query_dir = os.path.join(self.dataset_dir, 'query')
        # self.gallery_dir = os.path.join(self.dataset_dir, 'gallery')

        train_data, train_data_ids = self._process_data(self.train_dir, os.path.join(self.mask_dir, 'train'),
                                                        os.path.join(self.face_dir, 'train'), relabel=True)
        # query_data, query_data_ids = self._process_data(self.query_dir, os.path.join(self.mask_dir, 'query'),
        # os.path.join(self.face_dir, 'query'), relabel=False)
        # gallery_data, gallery_data_ids = self._process_data(self.gallery_dir, os.path.join(self.mask_dir, 'gallery'),
        # os.path.join(self.face_dir, 'gallery'), relabel=False)

        self.train_data = train_data
        self.train_data_ids = train_data_ids
        # self.query_data = query_data
        # self.query_data_ids = query_data_ids
        # self.gallery_data = gallery_data
        # self.gallery_data_ids = gallery_data_ids
        # self.test_data = self.query_data + self.gallery_data

        print("SC-ReID {} loaded".format(__factory[self.name]))
        print("dataset  |   ids |     imgs")
        print("train    | {:5d} | {:8d}".format(self.train_data_ids,len(self.train_data)))
        # print("query    | {:5d} | {:8d}".format(self.query_data_ids, len(self.query_data)))
        # print("gallery  | {:5d} | {:8d}".format(self.gallery_data_ids, len(self.gallery_data)))
        print("----------------------------------------")

    def _process_data(self, dir_path, msk_dir_path, face_dir_path, relabel=False):

        if self.name == 'prcc':
            persons = os.listdir(dir_path)
            dataset = []
            pid2label = {pid: label for label, pid in enumerate(persons)}

            for person in persons:
                person_path = os.path.join(dir_path, person)
                pics = os.listdir(person_path)
                for pic in pics:
                    if relabel:
                        pid = pid2label[person]
                    else:
                        pid = int(person)
                    img_path = os.path.join(dir_path, person, pic)
                    name = pic.split('.')[0] + '.png'
                    msk_path = os.path.join(msk_dir_path, person, name)
                    face_path = os.path.join(face_dir_path, person, pic)
                    if os.path.exists(msk_path) and os.path.exists(face_path):
                        dataset.append((img_path, pid, msk_path, face_path))

        if self.name == 'ltcc':
            persons = os.listdir(dir_path)
            dataset = []
            # i=0
            # pid_container = {}
            for person in persons:
                pid = int(person.split('_')[0])
                # if pid not in pid_container:
                    # pid_container[pid] = i
                    # i += 1
                img_path = os.path.join(dir_path, person)
                name = person.split('.')[0] + '.png'
                msk_path = os.path.join(msk_dir_path, name)
                face_path = os.path.join(face_dir_path, person)
                if os.path.exists(msk_path) and os.path.exists(face_path):
                    dataset.append((img_path, pid, msk_path, face_path))

        if self.name == 'vcclothes':
            persons = os.listdir(dir_path)
            dataset = []
            # i=0
            # pid_container = {}
            for person in persons:
                pid = int(person.split('-')[0])
                # if pid not in pid_container:
                    # pid_container[pid] = i
                    # i += 1
                img_path = os.path.join(dir_path, person)
                name = person.split('.')[0] + '.png'
                msk_path = os.path.join(msk_dir_path, name)
                face_path = os.path.join(face_dir_path, person)
                if os.path.exists(msk_path) and os.path.exists(face_path):
                    dataset.append((img_path, pid, msk_path, face_path))
                    # dataset.append((img_path, pid_container[pid], msk_path, face_path))

        if self.name == 'market':
            persons = os.listdir(dir_path)
            dataset = []
            # i=0
            # pid_container = {}
            for person in persons:
                pid = int(person.split('_')[0])
                # if pid not in pid_container:
                    # pid_container[pid] = i
                    # i += 1
                img_path = os.path.join(dir_path, person)
                name = person.split('.')[0] + '.png'
                msk_path = os.path.join(msk_dir_path, name)
                face_path = os.path.join(face_dir_path, person)
                if os.path.exists(msk_path) and os.path.exists(face_path):
                    dataset.append((img_path, pid, msk_path, face_path))
                    # dataset.append((img_path, pid_container[pid], msk_path, face_path))

        return dataset, len(persons)  # len(pid_container)



class Origin_dataset(object):

    def __init__(self, name, root='../datasets'):

        self.name = name
        self.root = root
        __testDS = {
            'prcc': 'prcc',
            'celeb': 'celeb',
            'ltcc': 'ltcc',
            'last': 'last',
            'vcclothes': 'vcclothes',
            'vsclothes': 'vsclothes',
            'market': 'market',
        }
        __factory = {
            'prcc': 'prcc/rgb/test',
            'celeb': 'Celeb-reID',
            'ltcc': 'LTCC_ReID',
            'last': 'last',
            'vcclothes': 'VC-Clothes',
            'vsclothes': 'VS-clothes/01',
            'market': 'market1501',
        }

        if __testDS[self.name] == 'prcc':
            self.dataset_dir = os.path.join(self.root, __factory[__testDS[self.name]])
            self.query_dir = os.path.join(self.dataset_dir, 'C')
            self.gallery_dir = os.path.join(self.dataset_dir, 'A')
            query_data, query_data_ids = self._process_prcc(self.query_dir, 'query')
            gallery_data, gallery_data_ids = self._process_prcc(self.gallery_dir, 'gallery')
            self.query_data = query_data
            self.query_data_ids = query_data_ids
            self.gallery_data = gallery_data
            self.gallery_data_ids = gallery_data_ids
            self.test_data = self.query_data + self.gallery_data

        if __testDS[self.name] == 'ltcc':
            self.dataset_dir = os.path.join(self.root, __factory[__testDS[self.name]])
            self.query_dir = os.path.join(self.dataset_dir, 'query')
            self.gallery_dir = os.path.join(self.dataset_dir, 'test')
            query_data, query_data_ids = self._process_ltcc(self.query_dir)
            gallery_data, gallery_data_ids = self._process_ltcc(self.gallery_dir)
            self.query_data = query_data
            self.query_data_ids = query_data_ids
            self.gallery_data = gallery_data
            self.gallery_data_ids = gallery_data_ids
            self.test_data = self.query_data + self.gallery_data

        if __testDS[self.name] == 'celeb':
            self.dataset_dir = os.path.join(self.root, __factory[__testDS[self.name]])
            self.query_dir = os.path.join(self.dataset_dir, 'query')
            self.gallery_dir = os.path.join(self.dataset_dir, 'gallery')
            query_data, query_data_ids = self._process_celeb(self.query_dir)
            gallery_data, gallery_data_ids = self._process_celeb(self.gallery_dir)
            self.query_data = query_data
            self.query_data_ids = query_data_ids
            self.gallery_data = gallery_data
            self.gallery_data_ids = gallery_data_ids
            self.test_data = self.query_data + self.gallery_data

        if __testDS[self.name] == 'last':

            self.dataset_dir = os.path.join(self.root, __factory[__testDS[self.name]])
            self.query_dir = os.path.join(self.dataset_dir, 'test', 'query')
            self.gallery_dir = os.path.join(self.dataset_dir, 'test', 'gallery')
            query_data, query_data_ids = self._process_last(self.query_dir)
            gallery_data, gallery_data_ids = self._process_prcc(self.gallery_dir)
            self.query_data = query_data
            self.query_data_ids = query_data_ids
            self.gallery_data = gallery_data
            self.gallery_data_ids = gallery_data_ids
            self.test_data = self.query_data + self.gallery_data

        if __testDS[self.name] == 'vcclothes':
            self.dataset_dir = os.path.join(self.root, __factory[__testDS[self.name]])
            self.query_dir = os.path.join(self.dataset_dir, 'query')
            self.gallery_dir = os.path.join(self.dataset_dir, 'gallery')
            query_data, query_data_ids = self._process_vcclothes(self.query_dir)
            gallery_data, gallery_data_ids = self._process_vcclothes(self.gallery_dir)
            self.query_data = query_data
            self.query_data_ids = query_data_ids
            self.gallery_data = gallery_data
            self.gallery_data_ids = gallery_data_ids
            self.test_data = self.query_data + self.gallery_data

        if __testDS[self.name] == 'vsclothes':
            self.dataset_dir = os.path.join(self.root, __factory[__testDS[self.name]])
            self.query_dir = os.path.join(self.dataset_dir, 'query')
            self.gallery_dir = os.path.join(self.dataset_dir, 'gallery')
            query_data, query_data_ids = self._process_vsclothes(self.query_dir)
            gallery_data, gallery_data_ids = self._process_vsclothes(self.gallery_dir)
            self.query_data = query_data
            self.query_data_ids = query_data_ids
            self.gallery_data = gallery_data
            self.gallery_data_ids = gallery_data_ids
            self.test_data = self.query_data + self.gallery_data

        if __testDS[self.name] == 'market':
            self.dataset_dir = os.path.join(self.root, __factory[__testDS[self.name]])
            self.query_dir = os.path.join(self.dataset_dir, 'query')
            self.gallery_dir = os.path.join(self.dataset_dir, 'bounding_box_test')
            query_data, query_data_ids = self._process_market(self.query_dir)
            gallery_data, gallery_data_ids = self._process_market(self.gallery_dir)
            self.query_data = query_data
            self.query_data_ids = query_data_ids
            self.gallery_data = gallery_data
            self.gallery_data_ids = gallery_data_ids
            self.test_data = self.query_data + self.gallery_data

        print("original {} loaded".format(self.name))
        print("dataset  |   ids |     imgs")
        print("query    | {:5d} | {:8d}".format(self.query_data_ids, len(self.query_data)))
        print("gallery  | {:5d} | {:8d}".format(self.gallery_data_ids, len(self.gallery_data)))
        print("----------------------------------------")

    def _process_prcc(self, dir_path, set):

        persons = os.listdir(dir_path)
        dataset = []
        for person in persons:
            person_path = os.path.join(dir_path, person)
            pics = os.listdir(person_path)
            for pic in pics:
                pid = int(person)
                if set == 'query':
                    camid = 1
                if set == 'gallery':
                    camid = 2
                img_path = os.path.join(dir_path, person, pic)
                dataset.append((img_path, pid, camid))

        return dataset, len(persons)

    def _process_ltcc(self, dir_path):

        persons = os.listdir(dir_path)
        dataset = []
        id_container = []
        for person in persons:
            img_path = os.path.join(dir_path, person)
            pid = int(person.split('_')[0])
            camids = person.split('_')[2]
            camid = int(camids.replace('c', ''))
            if pid not in id_container:
                id_container.append(pid)
            dataset.append((img_path, pid, camid))

        return dataset, len(id_container)

    def _process_celeb(self, dir_path):

        persons = os.listdir(dir_path)
        dataset = []
        id_container = []
        for person in persons:
            img_path = os.path.join(dir_path, person)
            pid = int(person.split('-')[0])
            if pid not in id_container:
                id_container.append(pid)
            dataset.append((img_path, pid))

        return dataset, len(id_container)

    def _process_last(self, dir_path):

        persons = os.listdir(dir_path)
        persons = persons[:(len(persons)//4)]
        dataset = []
        id_container = []
        for person in persons:
            img_path = os.path.join(dir_path, person)
            pid = int(person.split('_')[0])
            if pid not in id_container:
                id_container.append(pid)
            dataset.append((img_path, pid))

        return dataset, len(id_container)

    def _process_vcclothes(self, dir_path):

        persons = os.listdir(dir_path)
        dataset = []
        id_container = []
        for person in persons:
            img_path = os.path.join(dir_path, person)
            pid = int(person.split('-')[0])
            camid = int(person.split('-')[1])
            if pid not in id_container:
                id_container.append(pid)
            dataset.append((img_path, pid, camid))

        return dataset, len(id_container)

    def _process_vsclothes(self, dir_path):

        persons = os.listdir(dir_path)
        dataset = []
        id_container = []
        for person in persons:
            img_path = os.path.join(dir_path, person)
            pid = int(person.split('_')[1])
            camid = int(person.split('_')[2])
            if pid not in id_container:
                id_container.append(pid)
            dataset.append((img_path, pid, camid))

        return dataset, len(id_container)

    def _process_market(self, dir_path):

        persons = os.listdir(dir_path)
        dataset = []
        id_container = []
        for person in persons:
            img_path = os.path.join(dir_path, person)
            pid = int(person.split('_')[0])
            cs = person.split('_')[1]
            camid = int(cs[1])
            if pid not in id_container:
                id_container.append(pid)
            dataset.append((img_path, pid, camid))

        return dataset, len(id_container)

datasets_list = ['prcc', 'ltcc', 'celeb', 'ltcc+prcc', 'celeb+last', 'vsclothes', 'vcclothes', 'market']

def init_dataset(name, **kwargs):
    if name not in datasets_list:
        raise KeyError("Invalid dataset, expected to be one of {}".format(datasets_list))
    return CCPRF(name, **kwargs)

def init_test_dataset(name, **kwargs):
    if name not in datasets_list:
        raise KeyError("Invalid dataset, expected to be one of {}".format(datasets_list))
    return Origin_dataset(name, **kwargs)
