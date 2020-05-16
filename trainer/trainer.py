import os, logging
import torch
from trainer.lr_scheduler import *

try:
    import apex
    APEX = True
except:
    APEX = False


class Trainer(object):
    def __init__(self, config, model, loss_func, train_loader, val_loader=None):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.train_net = model
        self.losser = loss_func

        if config.resume:
            self.resume_model()
        else:
            self.optimizer = torch.optim.AdamW(self.train_net.parameters(),
                                               lr=config.lr,
                                               weight_decay=1e-4,
                                               amsgrad=config.AMSGRAD)
            self.lr_schedule = WarmupMultiStepLR(self.optimizer, config.steps, config.gamma,
                                                 warmup_iters=config.warmup_iters)
            self.start_step = 1
            self.best_loss = 1e6

        if config.gpu:
            self.train_net = self.train_net.cuda()
            self.losser = self.losser.cuda()

        if config.apex and APEX:
            self.train_net, self.optimizer = apex.amp.initialize(self.train_net, self.optimizer, opt_level="O1",
                                                                 verbosity=0)
            self.lr_schedule = WarmupMultiStepLR(self.optimizer, config.steps, config.gamma,
                                                 warmup_iters=config.warmup_iters)

        if not os.path.exists(config.log_dir):
            os.makedirs(config.log_dir)
        if not os.path.exists(config.checkpoint_dir):
            os.makedirs(config.checkpoint_dir)

        self.config = config
        self.logger = self.init_logger()
        self.logger.info('Trainer OK!')

        self.logger.info(config)

    def init_logger(self):
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(message)s')

        handler = logging.FileHandler(os.path.join(self.config.log_dir, "log.txt"))
        handler.setLevel(logging.INFO)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        console.setFormatter(formatter)
        logger.addHandler(console)
        return logger

    def get_lr(self):
        for param_group in self.optimizer.param_groups:
            return param_group['lr']

    def write_log(self, avg_loss, cls_loss=None, iou_loss=None, mode='TRAIN'):
        log = f'[{mode}]TOTAL_STEP: %6d/{self.config.max_iter}' % (self.start_step)
        if cls_loss is not None: log += f'  cls_loss: %.3f' % (cls_loss)
        if iou_loss is not None: log += f'  box_loss: %.3f' % (iou_loss)
        log += f'  avg_loss: %.3f' % (avg_loss)
        log += f'  lr: %.6f' % (self.get_lr())
        self.logger.info(log)

    def train(self):
        self.logger.info('Start trainning...\n')
        while self.start_step < self.config.max_iter:
            loss = self.train_one_epoch()
            if self.config.eval:
                loss = self.val_one_epoch()
                self.write_log(loss, mode='EVAL')

            self.save_model(loss < self.best_loss)
            self.best_loss = min(self.best_loss, loss)

    def train_one_epoch(self):
        self.train_net.train()
        total_loss = 0.
        for step, (gt) in enumerate(self.train_loader):
            if self.config.gpu:
                gt = [i.cuda() if isinstance(i, torch.Tensor) else i for i in gt]
            self.optimizer.zero_grad()

            pred = self.train_net(gt[0])
            losses = self.losser(pred, gt)
            loss = sum(losses)

            if self.config.apex and APEX:
                with apex.amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            self.optimizer.step()
            self.lr_schedule.step()

            total_loss += loss.item()
            self.start_step += 1
            if step % self.config.log_interval == 0:
                self.write_log(total_loss / (step + 1), losses[0].item(), losses[1].item())

        return total_loss / (step + 1)

    @torch.no_grad()
    def val_one_epoch(self):
        self.train_net.eval()
        total_loss = 0
        with torch.no_grad():
            for step, (gt) in enumerate(self.val_loader):
                if self.config.gpu:
                    gt = [i.cuda() if isinstance(i, torch.Tensor) else i for i in gt]

                pred = self.train_net(gt[0])
                losses = self.losser(pred, gt)
                total_loss += sum(losses).item()
        return total_loss / (step + 1)

    def save_model(self, is_best=False):
        state = {
            'model': self.train_net.state_dict(),
            'step': self.start_step,
            'optimizer': self.optimizer,
            'loss': self.best_loss,
            'config': self.config
        }
        if is_best:
            torch.save(state, os.path.join(self.config.checkpoint_dir, 'best_checkpoint.pth'))
        torch.save(state, os.path.join(self.config.checkpoint_dir, 'checkpoint.pth'))

    def resume_model(self):
        if self.config.resume_from_best:
            path = os.path.join(self.config.checkpoint_dir, 'tmp.pth')
        else:
            path = os.path.join(self.config.checkpoint_dir, 'checkpoint.pth')
        ckp = torch.load(path)
        model_static_dict = ckp['model']
        self.optimizer = ckp['optimizer']
        self.lr_schedule = ckp['lr_schedule']
        self.start_step = ckp['step']
        self.best_loss = ckp['loss']
        self.train_net.load_state_dict(model_static_dict)
