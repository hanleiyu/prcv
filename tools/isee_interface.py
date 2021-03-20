"""
Brief    : The interface class for the modules of vision algorithm used in ISEE.
VersIon  : 0.1
Date     : 2020.01.19
Copyright: ISEE, CRIPAC
"""

import abc
import argparse
import os
import sys
from os import mkdir
import os.path as osp
import cv2
from PIL import Image
import torch
from torch.backends import cudnn
import pdb

import numpy
sys.path.append('.')
from lib.data import make_data_loader
from lib.engine.inference import inference,extract_features
from lib.modeling import build_model
from lib.utils.logger import setup_logger
from lib.data.transforms.build import build_transforms
from lib.config import cfg  # import default parameters

class ISEEVisAlgIntf(metaclass=abc.ABCMeta):

    # Error Code
    _isee_errors = {
        'success': 0,         # Success
        'no_such_file': -1,   # File is not existed
        'null_data': -2,      # Null data
        'null_predictor': -3, # Null predictor
        'bad_device_id': -4,  # Bad device index
        'not_support': -5     # Not supported operation
        }

    @classmethod
    def verson(self):
        return "Verson 0.1"

    @classmethod
    def getErrType(self, err_no):
        for err_type, err_val in self._isee_errors.items():
            if err_no == err_val:
                return err_type
        # No such error nomuber.
        print("ERROR: Bad error code!")
        return None

    @abc.abstractmethod
    def init(self, config_file, params_dict = None):
        """
        Load model.
        params:
          config_file: the path of the cofiguration file containing the 
          necessary parameters (e.g., XML, json, YAML etc.).
          params_dict: the necessary parameters to initialize the project.
          It is in the type of dictionary as follows:
          {
            gpu_id: [-1], # the gpu id (a list of Integers), -1 means using CPU.
            model_path: ['/home/yourmodelpath', ..., ''], # a list of strings.
            reserved: {}  # other necessary parameters.
          }
        note:
          If overlapping prameters are existed in the configuration file and
          the variable of params_dict, the parameters in the variable of 
          params_dict will be used.
        return:
          error code: 0 for success; a negative number for the ERROR type.
        """
        pass

    @abc.abstractmethod
    def process(self, imgs_data, **kwargs):
        """
        Inference through loaded model.
        params:
          imgs_data: a list images data to process.
          **kwargs : the necessary parameters to implement inference combining
                     the results of other tasks.
        return:
          error code: 0 for success; a negative number for the ERROR type.
        """
        pass

    @abc.abstractmethod
    def getResults(self, img_path):
        """
        Get the processing results.
        params:
            img_path: the directory  which contains a list of images
        return:
            a list of l2-normlized feature vectors
        """
        pass

    @abc.abstractmethod
    def release(self):
        """
        Release the resources.
        """
        pass


class PRCV2020REID(ISEEVisAlgIntf):

    def init(self, config_file, params_dict = None ):
        """
        Load model.
        params:
          config_file: the path of the cofiguration file containing the
          necessary parameters (e.g., XML, json, YAML etc.).
          params_dict: the necessary parameters to initialize the project.
          It is in the type of dictionary as follows:
          {
            gpu_id: [-1], # the gpu id (a list of Integers), -1 means using CPU.
            model_path: ['/home/yourmodelpath', ..., ''], # a list of strings.
            reserved: {}  # other necessary parameters.
          }
        note:
          If overlapping prameters are existed in the configuration file and
          the variable of params_dict, the parameters in the variable of
          params_dict will be used.
        return:
          error code: 0 for success; a negative number for the ERROR type.
        """

        # read config file, yml format
        cfg.merge_from_file(config_file)
        cfg.merge_from_list(params_dict)
        cfg.freeze()

        # setting selected cuda devices
        if cfg.MODEL.DEVICE == "cuda":
            os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID
        cudnn.benchmark = True

        # build model and loading trained model
        self.model = build_model(cfg, 1295)
        self.model.load_param(cfg.TEST.WEIGHT)

        # move the model to cuda and set model to eval mode
        self.model.to(cfg.MODEL.DEVICE)
        self.model.eval()

        # build image transform [ Resize, ToTensor, Normalize]
        self.transform = build_transforms(cfg, is_train=False)

        return 0

    def process(self, imgs_data, **kwargs):
        """
        Inference through loaded model.
        params:
          imgs_data: a list of images data (OpenCV format BGR) [img1,img2,..imgN]
        return:
          error code: 0 for success; a negative number for the ERROR type.
        """

        feats = []
        with torch.no_grad():
            for i, img in enumerate(imgs_data):
                # convert cv2 format to PIL format
                img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

                # transform PIL image [ Resize, ToTensor, Normalize]
                img = self.transform(img).unsqueeze(0)

                # inference through the model
                img = img.cuda()
                feat = self.model(img)

                # whether to open flip augmentation
                if cfg.TEST.FLIP_TEST:

                    img_flip = img.flip(dims=[3])  # NCHW
                    feat_flip = self.model(img_flip)
                    feat = (feat + feat_flip) / 2

                feats.append(feat)

        feats = torch.cat(feats, dim=0)
        # l2-normalizing feature vectors
        feats = torch.nn.functional.normalize(feats, dim=1, p=2) # N*8192
        feats = feats.cpu().numpy()

        # A list of output feature vectors (numpy format) [feature1,feature2,...featureN]
        self.features  = [ feat for feat in feats]

        return 0

    def getResults(self, img_dir):
        """
        Get the processing results.
        params:
            img_dir: the directory which contains a list of images
        return:
            a list of l2-normlized feature vectors [feature1,feature2,...featureN]
        """
        # a list of image_names [img_name1,img_name2,..img_nameN]
        img_names = sorted(os.listdir(img_dir))
        # a list of opencv format (BGR) images data [img1,img2,..imgN]
        imgs_data = []

        # read the image by opencv
        for img_name in img_names:
            img = cv2.imread(os.path.join(img_dir,img_name)) # cv2 format: BGR
            imgs_data.append(img)

        # Calling the function process to obtain the feature vectors
        self.process(imgs_data)

        # return the l2-normalized feature vectors [feature1,feature2,...featureN] 8192-dim
        return self.features

    def release(self):
        print("release")

def main():
    parser = argparse.ArgumentParser(description="Re-ID model Inference Demo")
    parser.add_argument(
        "--config_file", default="./configs/prcv_ensemble.yml", help="path to config file", type=str
    )
    parser.add_argument(
        "--img_dir", default="imgs", help="path to the directory which contains a list of images", type=str
    )
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)
    args = parser.parse_args()

    # Create an demo object which belongs the class PRCV2020REID (the children-class of ISEEVisAlgIntf class)
    demo = PRCV2020REID()

    # Initialize the demo object by calling the function init (passing in the config-file and command parameters)
    demo.init(config_file=args.config_file, params_dict=args.opts)

    # Calling the getResults function to obtain the feature vectors [feature1,feature2,...featureN] (passing in the img_dir)
    features = demo.getResults(args.img_dir) # shape: [N*,8192]

    # print the features vectors [feature1,feature2,...,featureN] 8192-dim
    print(features)


if __name__ == '__main__':
    main()

