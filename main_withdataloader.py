# main.py

import os
import sys
import config
import traceback
from hal.utils import misc

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.plugins import DDPPlugin

import torch

import control
import hal.datasets as datasets


def main():
    # torch.set_default_dtype(torch.float64)
    # torch.set_default_tensor_type(torch.DoubleTensor)

    # parse the arguments
    args = config.parse_args()

    # try:
    # args.sensitive_attr = [int(x) for x in args.sensitive_attr.split(',')]

    # args.target_attr = [int(x) for x in args.target_attr.split(',')]
    # except Exception as e:
    # print(e)
    # print('Error in converting args.sensative_attr to int')

    if args.ngpu == 0:
        args.device = 'cpu'

    pl.seed_everything(args.manual_seed)

    logger = TensorBoardLogger(
        save_dir=args.out_dir,
        log_graph=True,
        name=args.project_name + '_' + args.exp_name
    )

    dataloader = getattr(datasets, args.dataset)(args)
    model = getattr(control, args.control_type)(args, dataloader)

    # if args.resume is not None:
    #     model = model.load_from_checkpoint(args.resume)

    callbacks = []

    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(args.out_dir, 'checkpoints'),
        filename=args.project_name + '-{epoch:03d}-{val_loss:.3f}',
        # filepath=os.path.join(args.save_dir, args.project_name + '-{epoch:03d}-{val_loss:.3f}'),
        monitor='val_loss',
        save_top_k=1)

    callbacks.append(checkpoint_callback)

    checkpoint_callback2 = ModelCheckpoint(
        dirpath=os.path.join(args.out_dir, 'checkpoints'),
        filename=args.project_name + '-lastEpoch-{epoch:03d}-{val_loss:.3f}',
    )

    callbacks.append(checkpoint_callback2)

    lr_monitor = LearningRateMonitor(logging_interval='step')
    callbacks.append(lr_monitor)

    # import pdb; pdb.set_trace()

    if args.EarlyStopping:
        early_stop_callback = EarlyStopping(**args.earlystop_options)

        callbacks.append(early_stop_callback)

    if args.ngpu == 0:
        accelerator = None
        sync_batchnorm = False
    else:
        accelerator = 'ddp'
        sync_batchnorm = True

    # import pdb; pdb.set_trace()

    trainer = pl.Trainer(
        gpus=args.ngpu,
        accelerator=accelerator,
        sync_batchnorm=sync_batchnorm,
        benchmark=True,
        # checkpoint_callback=checkpoint_callback,
        checkpoint_callback=True,
        callbacks=callbacks,
        logger=logger,
        min_epochs=1,
        max_epochs=args.nepochs,
        precision=args.precision,
        reload_dataloaders_every_epoch=False,
        num_sanity_val_steps=0,
        check_val_every_n_epoch=args.check_val_every_n_epochs
    )

    trainer.fit(model, dataloader)

    if args.test_part.lower() == 'train':
        trainer.test(model, test_dataloaders=dataloader.train_dataloader())

    elif args.test_part.lower() == 'val':
        trainer.test(model, test_dataloaders=dataloader.val_dataloader())

    elif args.test_part.lower() == 'test':
        if hasattr(model, 'test_flag'):
            # model.test_flag = 'train'
            # trainer.test(test_dataloaders=dataloader.train_dataloader())
            model.test_flag = 'entire'
            trainer.test(test_dataloaders=dataloader.entire_dataloader())
            model.test_flag = 'test'
            trainer.test(test_dataloaders=dataloader.test_dataloader())

        else:
            trainer.test(model, test_dataloaders=dataloader.test_dataloader())


if __name__ == "__main__":
    misc.setup_graceful_exit()
    try:
        main()
    except (KeyboardInterrupt, SystemExit):
        # do not print stack trace when ctrl-c is pressed
        pass
    except Exception:
        traceback.print_exc(file=sys.stdout)
    finally:
        traceback.print_exc(file=sys.stdout)
        misc.cleanup()
