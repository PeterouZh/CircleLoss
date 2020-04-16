import collections
import os

import torch
from torch import nn, Tensor
from torch.optim import SGD
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from tqdm import tqdm

from circle_loss import convert_label_to_similarity, CircleLoss

from template_lib.trainer.base_trainer import summary_defaultdict2txtfig, summary_dict2txtfig


def get_loader(datadir, is_train: bool, batch_size: int) -> DataLoader:
    datadir = os.path.expanduser(datadir)
    return DataLoader(
        dataset=MNIST(root=datadir, train=is_train, transform=ToTensor(), download=True),
        batch_size=batch_size,
        shuffle=is_train,
    )


class Model(nn.Module):
    def __init__(self) -> None:
        super(Model, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=5),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=5),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(),
        )

    def forward(self, inp: Tensor) -> Tensor:
        feature = self.feature_extractor(inp).mean(dim=[2, 3])
        return nn.functional.normalize(feature)


def main(myargs, resume: bool = True) -> None:
    args = myargs.config

    saved_model = os.path.join(myargs.args.outdir, "resume.state")

    model = Model().cuda()
    optimizer = SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-5)
    train_loader = get_loader(datadir=args.datadir, is_train=True, batch_size=64)
    val_loader = get_loader(datadir=args.datadir, is_train=False, batch_size=2)
    criterion = CircleLoss(m=0.25, gamma=80)

    if resume and os.path.exists("resume.state"):
        model.load_state_dict(torch.load("resume.state"))
    else:
        counter = 0
        for epoch in range(100000):
            if args.dummy_train:
                continue
            myargs.logger.info(f'Epoch {epoch}')
            pbar = tqdm(train_loader, desc=f'{myargs.args.time_str_suffix}', file=myargs.stdout)
            for step, (img, label) in enumerate(pbar):
                img = img.cuda()
                label = label.cuda()
                model.zero_grad()
                pred = model(img)
                sp, sn = convert_label_to_similarity(pred, label)
                loss = criterion(sp, sn)
                loss.backward()
                optimizer.step()
                if counter % 10 == 0:
                    summary_dicts = collections.defaultdict(dict)
                    summary_dicts['sp_sn']['sp_mean'] = sp.mean().item()
                    summary_dicts['sp_sn']['sn_mean'] = sn.mean().item()
                    summary_dicts['loss']['loss'] = loss.item()
                    summary_defaultdict2txtfig(default_dict=summary_dicts, prefix='train',
                                               step=counter, textlogger=myargs.textlogger, save_fig_sec=90)
                counter += 1
            recal, pre, (tp, fp, fn, tn) = validate(val_loader=val_loader, model=model, stdout=myargs.stdout)
            summary_dicts = collections.defaultdict(dict)
            summary_dicts['recal']['recal'] = recal
            summary_dicts['pre']['pre'] = pre
            summary_dicts['tp_fp_fn_tn']['tp'] = tp
            summary_dicts['tp_fp_fn_tn']['fp'] = fp
            summary_dicts['tp_fp_fn_tn']['fn'] = fn
            summary_dicts['tp_fp_fn_tn']['tn'] = tn
            summary_defaultdict2txtfig(default_dict=summary_dicts, prefix='val',
                                       step=epoch, textlogger=myargs.textlogger, save_fig_sec=90)
        torch.save(model.state_dict(), saved_model)


def validate(val_loader, model, stdout):
    tp = 0
    fn = 0
    fp = 0
    tn = 0
    thresh = 0.75
    for img, label in tqdm(val_loader, file=stdout):
        img = img.cuda()
        label = label.cuda()
        pred = model(img)
        gt_label = label[0] == label[1]
        pred_label = torch.sum(pred[0] * pred[1]) > thresh
        if gt_label and pred_label:
            tp += 1
        elif gt_label and not pred_label:
            fn += 1
        elif not gt_label and pred_label:
            fp += 1
        else:
            tn += 1

    recal = tp / (tp + fn)
    print("Recall: {:.4f}".format(recal))
    pre = tp / (tp + fp)
    print("Precision: {:.4f}".format(pre))
    return recal, pre, (tp, fp, fn, tn)


def run(argv_str=None):
  from template_lib.utils.config import parse_args_and_setup_myargs, config2args
  from template_lib.utils.modelarts_utils import prepare_dataset
  run_script = os.path.relpath(__file__, os.getcwd())
  args1, myargs, _ = parse_args_and_setup_myargs(argv_str, run_script=run_script, start_tb=False)
  myargs.args = args1
  myargs.config = getattr(myargs.config, args1.command)

  if hasattr(myargs.config, 'datasets'):
    prepare_dataset(myargs.config.datasets, cfg=myargs.config)

  main(myargs)

if __name__ == '__main__':
  run()


