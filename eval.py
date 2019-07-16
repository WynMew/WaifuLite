from utils.prepare_images import *
from Models import *
from torchvision.utils import save_image
import cv2
import numpy as np
#putorch1.0

'''
model_cran_v2 = CARN_V2(color_channels=3, mid_channels=64, conv=nn.Conv2d,
                        single_conv_size=3, single_conv_group=1,
                        scale=2, activation=nn.LeakyReLU(0.1),
                        SEBlock=True, repeat_blocks=3, atrous=(1, 1, 1))

model_cran_v2 = network_to_half(model_cran_v2)
checkpoint = "model_check_points/CRAN_V2/CARN_model_checkpoint.pt"
#model_cran_v2.load_state_dict(torch.load(checkpoint, 'cpu'))
# if use GPU, then comment out the next line so it can use fp16.
model_cran_v2 = model_cran_v2.float()

demo_img = "0651-3037651-0.jpg"
img = Image.open(demo_img).convert("RGB")
#img.resize((256,256))

# origin
img_t = to_tensor(img).unsqueeze(0)
#print(img_t.shape)

# used to compare the origin
img = img.resize((img.size[0] // 2, img.size[1] // 2), Image.BICUBIC)

img_splitter = ImageSplitter(seg_size=64, scale_factor=2, boarder_pad_size=3)
img_patches = img_splitter.split_img_tensor(img, scale_method=None, img_pad=0)
with torch.no_grad():
    out = [model_cran_v2(i) for i in img_patches]

img_upscale = img_splitter.merge_img_tensor(out)
print(img_upscale.data.cpu()[0].shape)
output_image = (img_upscale.data.cpu()[0].permute(1, 2, 0).float() * 0.5 + 0.5).numpy()
normImg = cv2.normalize(output_image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX,
                                dtype=cv2.CV_32F).astype(np.uint8)

print(normImg)
outimg = Image.fromarray(normImg)
outimg.save('out.jpg')
'''

'''
model_cran_v2 = CARN_V2(color_channels=3, mid_channels=64, conv=nn.Conv2d,
                        single_conv_size=3, single_conv_group=1,
                        scale=2, activation=nn.LeakyReLU(0.1),
                        SEBlock=True, repeat_blocks=3, atrous=(1, 1, 1))

model_cran_v2 = network_to_half(model_cran_v2)
checkpoint = "model_check_points/CRAN_V2/CARN_model_checkpoint.pt"
model_cran_v2.load_state_dict(torch.load(checkpoint, 'cpu'))
# if use GPU, then comment out the next line so it can use fp16.
model_cran_v2 = model_cran_v2.float()

demo_img = "0651-3037651-0.jpg"
img = Image.open(demo_img).convert("RGB")

img_t = to_tensor(img).unsqueeze(0)

img_splitter = ImageSplitter(seg_size=64, scale_factor=2, boarder_pad_size=3)
img_patches = img_splitter.split_img_tensor(img, scale_method=None, img_pad=0)
with torch.no_grad():
    out = [model_cran_v2(i) for i in img_patches]

img_upscale = img_splitter.merge_img_tensor(out)[0]
normImg=img_upscale.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()

im=Image.fromarray(normImg)
im.save('out.png')

'''
'''
model_cran_v2 = CARN_V2(color_channels=3, mid_channels=64, conv=nn.Conv2d,
                        single_conv_size=3, single_conv_group=1,
                        scale=2, activation=nn.LeakyReLU(0.1),
                        SEBlock=True, repeat_blocks=3, atrous=(1, 1, 1))

model_cran_v2 = network_to_half(model_cran_v2)
checkpoint = "model_check_points/CRAN_V2/CARN_model_checkpoint.pt"
model_cran_v2.load_state_dict(torch.load(checkpoint, 'cpu'))
# if use GPU, then comment out the next line so it can use fp16.
model_cran_v2 = model_cran_v2.float()

valid_ext = ['.jpg', '.png']
input_dir = '/home/wynmew/data/background/origCropCartoon/'
output_dir = '/home/wynmew/data/background/origCropCartoon2X/'

for files in os.listdir(input_dir):
    ext = os.path.splitext(files)[1]
    if ext not in valid_ext:
        continue
    # load image
    print(files)
    img = Image.open(os.path.join(input_dir, files)).convert("RGB")
    img_t = to_tensor(img).unsqueeze(0)
    img_splitter = ImageSplitter(seg_size=64, scale_factor=2, boarder_pad_size=3)
    img_patches = img_splitter.split_img_tensor(img, scale_method=None, img_pad=0)
    with torch.no_grad():
        out = [model_cran_v2(i) for i in img_patches]

    img_upscale = img_splitter.merge_img_tensor(out)[0]
    normImg=img_upscale.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()

    im=Image.fromarray(normImg)
    im.save(os.path.join(output_dir, files[:-4] + '_' + '2x' + '.png'))

'''

model_cran_v2 = CARN_V2(color_channels=3, mid_channels=64, conv=nn.Conv2d,
                        single_conv_size=3, single_conv_group=1,
                        scale=2, activation=nn.LeakyReLU(0.1),
                        SEBlock=True, repeat_blocks=3, atrous=(1, 1, 1))

model_cran_v2 = network_to_half(model_cran_v2)
checkpoint = "model_check_points/CRAN_V2/CARN_model_checkpoint.pt"
model_cran_v2.load_state_dict(torch.load(checkpoint, 'cpu'))
# if use GPU, then comment out the next line so it can use fp16.
#model_cran_v2 = model_cran_v2.float()
model_cran_v2.cuda()
model_cran_v2.eval()

valid_ext = ['.jpg', '.png']
input_dir = '/home/wynmew/data/background/origCropCartoon/'
output_dir = '/home/wynmew/data/background/origCropCartoon2X/'

for files in os.listdir(input_dir):
    ext = os.path.splitext(files)[1]
    if ext not in valid_ext:
        continue
    # load image
    print(files)
    img = Image.open(os.path.join(input_dir, files)).convert("RGB")
    img_t = to_tensor(img).unsqueeze(0)
    img_splitter = ImageSplitter(seg_size=64, scale_factor=2, boarder_pad_size=3)
    img_patches = img_splitter.split_img_tensor(img, scale_method=None, img_pad=0)
    with torch.no_grad():
        out = [model_cran_v2(i.cuda()) for i in img_patches]

    img_upscale = img_splitter.merge_img_tensor(out)[0]
    normImg=img_upscale.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()

    im=Image.fromarray(normImg)
    im.save(os.path.join(output_dir, files[:-4] + '_' + '2x' + '.png'))

