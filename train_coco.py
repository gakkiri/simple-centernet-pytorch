from trainer.trainer import Trainer
from dataset.coco import COCODataset
from model.centernet import CenterNet
from config.coco_cfg import Config
from loss.loss import Loss
from torch.utils.data import DataLoader


def train(cfg):
    train_ds = COCODataset(cfg.train_imgs_path, cfg.train_anno_path, resize_size=cfg.resize_size)
    train_dl = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=False,
                          num_workers=cfg.num_workers, collate_fn=train_ds.collate_fn)

    if cfg.eval:
        eval_ds = COCODataset(cfg.eval_imgs_path, cfg.eval_anno_path, resize_size=cfg.resize_size)
        eval_dl = DataLoader(eval_ds, batch_size=max(cfg.batch_size // 2, 1),
                             num_workers=cfg.num_workers, collate_fn=eval_ds.collate_fn)
    else:
        eval_dl = None

    model = CenterNet(cfg)
    if cfg.gpu:
        model = model.cuda()

    loss_func = Loss(cfg)

    trainer = Trainer(cfg, model, loss_func, train_dl, eval_dl)
    trainer.train()


if __name__ == '__main__':
    cfg = Config()
    train(cfg)
