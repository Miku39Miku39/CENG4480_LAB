from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR, MultiStepLR
from tqdm import tqdm
import warnings
#from matplotlib import pyplot as plt

warnings.filterwarnings('ignore')
import cv2
from PIL import Image

'''
1. initialize the camera as 'cap' here
'''

# set the width and height, and fps.
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 224)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 224)
cap.set(cv2.CAP_PROP_FPS, 10)
# frame width
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# frame height
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# where to put the text
word_x = int(frame_width / 10)
word_y = int(frame_height / 10)


'''
2. define the neural network model
'''


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--no-mps', action='store_true', default=False,
                        help='disables macOS GPU training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    
    # set the seed
    # torch.manual_seed(args.seed)

    if use_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    '''
    3. Set transform
    '''

    '''
    4. build a model
    '''

    '''
    5. load the model
    '''
    
    while (cap.isOpened()):
        '''
        6. read frames
        '''
        
        '''
        7. process the frames
        '''
        
        # use the given transform to process the binary image
        data = transform(binary)
        data = data.to(device)
        data = data.unsqueeze(0)

        '''
        8. feed the image to the model and get the prediction, and print the prediction
        '''

        # for the putText function:
        # 1. you need to replace the original_frame with the frame you get from the camera
        # 2. the second argument needs a string-type input, so you need to convert your prediction to a string
        cv2.putText(original_frame, str("the above prediction"), (word_x,word_y), cv2.FONT_HERSHEY_SIMPLEX,1,(55,255,155),2)

        '''
        9. show the video
        '''

    cap.release()
    cv2.destroyAllWindows()
	
if __name__ == '__main__':
    main()

