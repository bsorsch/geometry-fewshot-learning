"""
Compute generalization error of prototype few-shot learning.

"""

from jax import numpy as np
from jax import jit
from jax import random
import os
import sys, argparse
from scipy.spatial.distance import squareform

parser=argparse.ArgumentParser()

parser.add_argument('--model', help='CNN architecture (resnet50, vgg10, etc.).')
parser.add_argument('--gpu', help='Which gpu to evaluate on.', default='')
parser.add_argument('--m', help='Num training examples.', default=1)
parser.add_argument('--n_avg', help='Num episodes per manifold.', default=10000)

args=parser.parse_args()

if args.model==None:
	print('No model selected, defaulting to resnet50.')
	args.model='resnet50'

# Set up gpu
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)

# Manifolds directory
model_name = args.model
emb_path = '/mnt/fs2/bsorsch/manifold/embeddings_new/1k_objects'
model_dir = os.path.join(emb_path,model_name)
save_path = os.path.join(model_dir,'errs_{}shot.npy'.format(m))
print('Computing generalization error for ' + str(model_name))
print('Saving to: ' + save_path)

# Load manifolds
manifolds_load = np.load(os.path.join(model_dir, 'manifolds.npy'),allow_pickle=True)
P = 500
manifolds = []
for manifold in manifolds_load:
    manifolds.append(manifold[:P])
manifolds = np.stack(manifolds)


m = int(args.m)

@jit
def shuffle(key,Xa,Xb):
    ''' Randomly shuffle examples in Xa and Xb along the zeroth axis.
    Args: 
        key: random PRNGkey
        Xa: (P,N) first array to shuffle
        Xb: (P,N) second array to shuffle
    
    Returns:
        Xaperm: (P,N) shuffled copy of Xa
        Xbperm: (P,N) shuffled copy of Xb
    '''
    keya,keyb=random.split(key)
    perma = random.shuffle(keya,np.arange(len(Xa)))
    permb = random.shuffle(keyb,np.arange(len(Xb)))
    
    return Xa[perma],Xb[permb]


@jit
def mshot_err(X):
    ''' Performs an m-shot learning experiment on a pair of shuffled manifolds X=(Xa,Xb).
    Args: 
        m: # training examples
        X: X=(Xa,Xb), a pair of (P,N) object manifolds, pre-shuffled along the zeroth axis.
    
    Returns:
        erra: m-shot learning error evaluated on manifold a
        errb: m-shot learning error evaluated on manifold b
    '''
    Xa,Xb = X
    xatrain, xatest = np.split(Xa, (m,))
    xa = xatrain.mean(0)
    xbtrain, xbtest = np.split(Xb, (m,))
    xb = xbtrain.mean(0)
    x = np.vstack([xa,xb])

    distsa = ((x[:,None] - xatest[None])**2).sum(-1)
    ya = distsa.argmin(0)

    distsb = ((x[:,None] - xbtest[None])**2).sum(-1)
    yb = distsb.argmin(0)

    erra = (ya!=0).mean()
    errb = (yb!=1).mean()

    return erra, errb


@jit
def mshot_err_fast(key,Xa,Xb):
    ''' Performs a quick heuristic m-shot learning experiment on a pair of manifolds X=(Xa,Xb),
    allowing overlap between training and test examples.
    
    Args: 
        X: X=(Xa,Xb), a pair of (P,N) object manifolds, pre-shuffled along the zeroth axis.
    
    Returns:
        erra: m-shot learning error evaluated on manifold a
        errb: m-shot learning error evaluated on manifold b
    '''
    keya, keyb = random.split(key)
    idxs_a = random.randint(keya, (m,int(args.n_avg)), 0,P)
    idxs_b = random.randint(keyb, (m,int(args.n_avg)), 0,P)
    
    # Prototypes
    xabar = Xa[idxs_a].mean(0)
    xbbar = Xb[idxs_b].mean(0)

    # Distances to prototypes
    daa = ((Xa[:,None] - xabar[None])**2).sum(-1)
    dab = ((Xa[:,None] - xbbar[None])**2).sum(-1)
    dba = ((Xb[:,None] - xabar[None])**2).sum(-1)
    dbb = ((Xb[:,None] - xbbar[None])**2).sum(-1)
    ha = -daa + dab
    hb = -dbb + dba

    erra = (ha<0).mean()
    errb = (hb<0).mean()

    return erra, errb


key = random.PRNGKey(0)


K = len(manifolds)
N = manifolds.shape[-1]

# # Compute generalization error, slow but precise
# errs_a = []
# errs_std_a = []
# errs_b = []
# errs_std_b = []
# for ii,a in enumerate(range(K)):
#     Xa = np.array(manifolds[a])
#     for b in range(a+1,K):
#         Xb = np.array(manifolds[b])
#         erra = []
#         errb = []
#         for _ in range(args.n_avg):
#             key,_ = random.split(key)
#             erratmp,errbtmp = mshot_err(shuffle(key,Xa,Xb))
#             erra.append(erratmp)
#             errb.append(errbtmp)
#         errs_a.append(np.stack(erra).mean())
#         errs_std_a.append(np.stack(erra).std())
#         errs_b.append(np.stack(errb).mean())
#         errs_std_b.append(np.stack(errb).std())

# Compute generalization error, rapid but heuristic
errs_a = []
errs_b = []
for ii,a in enumerate(range(K)):
    Xa = np.array(manifolds[a])
    for b in range(a+1,K):
        Xb = np.array(manifolds[b])
        erra = []
        errb = []

        key,_ = random.split(key)
        erra,errb = mshot_err_fast(key,Xa,Xb)

        errs_a.append(erra)
        errs_b.append(errb)

    print('Manifold {} of {}. Avg. acc: {}'.format(ii,K,1-errs_a[-1].mean()))

# Combine errs_a and errs_b into K x K matrix
errs_full = np.triu(squareform(errs_a)) + np.tril(squareform(errs_b))

# Save
np.save(save_path,errs_full)
print('Finished with acc. ' + str(1 - np.mean(errs_full)) + '. Saved.')




