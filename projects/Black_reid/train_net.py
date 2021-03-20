import os
import sys

from torch import nn
import time
import pdb
sys.path.append('../../')
from fastreid.config import get_cfg
from fastreid.engine import DefaultTrainer, default_argument_parser, default_setup
from fastreid.utils.checkpoint import Checkpointer
from fastreid.evaluation import ReidEvaluator
import pdb

try:
    from apex.parallel import DistributedDataParallel as DDP
    from apex.fp16_utils import *
    from apex import amp, optimizers
    from apex.multi_tensor_apply import multi_tensor_applier
except ImportError:
    raise ImportError("Please install apex from https://www.github.com/nvidia/apex to run this example.")

class Trainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, num_query, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return ReidEvaluator(cfg, num_query)

    @classmethod
    def build_train_loader(cls, cfg):
        return build_reid_train_loader(cfg)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        return build_reid_test_loader(cfg, dataset_name)
    
    def run_step(self):
        """
        Implement the standard training logic described above.
        """
        assert self.model.training, "[SimpleTrainer] model was changed to eval mode!"
        start = time.perf_counter()
        """
        If your want to do something with the data, you can wrap the dataloader.
        """

        data = self.data_loader.next()
        data_time = time.perf_counter() - start
        """
        If your want to do something with the heads, you can wrap the model.
        """
        outputs = self.model(data)
        loss_dict = self.model.module.losses(outputs)
        losses = sum(loss for loss in loss_dict.values())
        self._detect_anomaly(losses, loss_dict)

        metrics_dict = loss_dict
        metrics_dict["data_time"] = data_time
        self._write_metrics(metrics_dict)

        """
        If you need accumulate gradients or something similar, you can
        wrap the optimizer with your custom `zero_grad()` method.
        """
        self.optimizer.zero_grad()
        losses.backward()

        """
        If you need gradient clipping/scaling or other processing, you can
        wrap the optimizer with your custom `step()` method.
        """
        self.optimizer.step()

def setup():
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file("projects/Black_reid/configs/HAA_mgn_blackreid.yml")
    cfg.freeze()
    #default_setup(cfg, args)
    return cfg

def main(args):
    cfg = setup(args)
    if args.eval_only:
        cfg.defrost()
        cfg.MODEL.BACKBONE.PRETRAIN = False
        model = Trainer.build_model(cfg)
        model = nn.DataParallel(model)
        model = model.cuda()

        Checkpointer(model, save_dir=cfg.OUTPUT_DIR).load(cfg.MODEL.WEIGHTS)
        res = Trainer.test(cfg, model)
        return res

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()

def create():
    cfg = setup()
    cfg.defrost()
    cfg.MODEL.BACKBONE.PRETRAIN = False
    model = Trainer.build_model(cfg)
    return model


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    #pdb.set_trace()
    print("Command Line Args:", args)
    main(args)