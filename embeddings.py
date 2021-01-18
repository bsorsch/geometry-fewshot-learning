"""
Extract embeddings in the feature layer of trained CNNs, for objects from the ImageNet21k dataset.

"""

import sys, argparse
import numpy as np
import torch

parser=argparse.ArgumentParser()

parser.add_argument('--model', help='CNN architecture (resnet50, vgg10, etc.).')
parser.add_argument('--gpu', help='Which gpu to evaluate on.', default='')
parser.add_argument('--n_manifolds',
				 help='How many objects to obtain embeddings for.', default=100)

args=parser.parse_args()

if args.model==None:
	print('No model selected, defaulting to resnet50.')
	args.model='resnet50'

# Set up gpu
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)


# Location of ImageNet, and location to save embeddings
impath = '/mnt/fs0/datasets/ImageNet21k/'
emb_path = '/mnt/fs2/bsorsch/manifold/embeddings_new/1k_objects'
wnids_1k = np.load(os.path.join(emb_path,'wnids_1k.npy'))
names_1k = np.load(os.path.join(emb_path,'names_1k.npy'))


## Data
# Image preprocessing
from torchvision import transforms
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

from PIL import Image

def get_batch(i, wnid_dir, imgs, batch_size):
    xbatch = []
    for img in imgs[i*batch_size:(i+1)*batch_size]:
        try:
            x = Image.open(os.path.join(wnid_dir, img)).convert('RGB')
            xbatch.append(preprocess(x))
        except OSError: 
            pass

    return torch.stack(xbatch)


from torchvision.utils import make_grid

def show_grid(im_tensor, nrow=8, title=None):
    im_grid = make_grid(im_tensor, nrow=nrow, padding=2)
    im_grid = im_grid.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    im_grid = std * im_grid + mean
    im_grid = np.clip(im_grid, 0, 1)
    
    plt.figure(figsize=(15,2*len(input_tensor)//nrow))
    plt.imshow(im_grid)
    plt.axis('off')
    plt.title(title)



## Model
model_name = args.model
repo = 'pytorch/vision:v0.6.0'
# repo = 'rwightman/gen-efficientnet-pytorch'
model = torch.hub.load(repo, model_name, pretrained=True)
model_dir = os.path.join(emb_path,model_name)

# Remove readout layer
fc_models = ['resnet','resnext','google','dense','inception','efficient']
if np.any([fcm in model_name for fcm in fc_models if fcm]):
    model.fc = torch.nn.Sequential()
else:
    list(model.children())[-1][-1] = torch.nn.Sequential()
# Move to gpu
model.cuda().eval()

# Test batch 
batch_size = 16
wnid = wnids_1k[0]
wnid_dir = os.path.join(impath, wnid)
imgs = os.listdir(wnid_dir)
input_tensor = get_batch(0, wnid_dir, imgs, batch_size)

with torch.no_grad():
    output = model(input_tensor.cuda())
    
N = output.shape[-1]
if N > 2048:
    random_projection=True
    U = torch.randn(N, 2048) / np.sqrt(2048)
    U = U.cuda()
else:
    random_projection=False


P = 500
batch_size = 16

print('Getting embeddings. Saving to: ' + model_dir)
manifolds = []
for ii,wnid in enumerate(wnids_1k[:int(args.n_manifolds)]):
    wnid_dir = os.path.join(impath, wnid)
    imgs = os.listdir(wnid_dir)
    
    manifold = []
    for i in range(len(imgs)//batch_size):
        input_tensor = get_batch(i, wnid_dir, imgs, batch_size)
        with torch.no_grad():
            output = model(input_tensor.cuda())
        if random_projection:
            manifold.append((output@U).cpu().numpy())
        else:
            manifold.append(output.cpu().numpy())
    manifold = np.concatenate(manifold)
    manifolds.append(manifold)

    print(str(ii) + '. Obtained embedding for ' + names_1k[ii])
    
# Make sure there are no duplicates
for i in range(len(manifolds)):
    norms = np.linalg.norm(manifolds[i],axis=-1)
    _,uniq_idxs = np.unique(norms,return_index=True)
    manifolds[i] = manifolds[i][sorted(uniq_idxs)]


# Save manifolds
if not os.path.isdir(model_dir):
    os.mkdir(model_dir)
np.save(os.path.join(model_dir,'manifolds.npy'),manifolds)


print('Saved to: ' + model_dir)