import argparse
import os
import torch
from PIL import Image
from torchvision import transforms
from torchvision.models import vgg19
import model_manager_ft

from gradcam import GradCam

def CAM(im_path, save_path):
    im = Image.open(im_path)
    im1 = im.resize((128, 256))
    test_image = (transforms.ToTensor()(im1)).unsqueeze(dim=0)
    # model = vgg19(pretrained=True)
    model = model_manager_ft.init_model(name='hr', class_num=9449)
    model = torch.nn.DataParallel(model).cuda()
    model_wts = torch.load('imagenet/vcclothesHrnetbaseline.pth')
    model.load_state_dict(model_wts['state_dict'])
    if torch.cuda.is_available():
        test_image = test_image.cuda()
        model.cuda()
    grad_cam = GradCam(model)
    feature_image = grad_cam(test_image).squeeze(dim=0)
    feature_image = transforms.ToPILImage()(feature_image)
    feature_image.save(save_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test Grad-CAM')
    parser.add_argument('--image_name', default='both.png', type=str, help='the tested image name')
    parser.add_argument('--save_name', default='grad_cam.png', type=str, help='saved image name')

    opt = parser.parse_args()

    # IMAGE_NAME = opt.image_name
    # SAVE_NAME = opt.save_name
    # CAM(IMAGE_NAME, SAVE_NAME)

    idlist = os.listdir('03')
    for id in idlist:
        id_path = os.path.join('03', id)
        imglist = os.listdir(id_path)
        for img in imglist:
            IMAGE_NAME = os.path.join(id_path, img)
            SAVE_NAME = os.path.join('result1', img)
            CAM(IMAGE_NAME, SAVE_NAME)