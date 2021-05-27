from trainer.trainer import Trainer
from dataset.voc import VOCDataset
from model.centernet import CenterNet
from config.voc import Config
from loss.loss import Loss
from torch.utils.data import DataLoader


def train(cfg):
    train_ds = VOCDataset(cfg.root, mode=cfg.split, resize_size=cfg.resize_size)
    train_dl = DataLoader(train_ds, batch_size=1, shuffle=True,
                          num_workers=cfg.num_workers, collate_fn=train_ds.collate_fn, pin_memory=True)

    model = CenterNet(cfg)
    if cfg.gpu:
        model = model.cuda()
    loss_func = Loss(cfg)

    epoch = 100
    cfg.max_iter = len(train_dl) * epoch
    cfg.steps = (int(cfg.max_iter * 0.6), int(cfg.max_iter * 0.8))

    trainer = Trainer(cfg, model, loss_func, train_dl, None)
    trainer.train()


if __name__ == '__main__':
    cfg = Config
    train(cfg)
