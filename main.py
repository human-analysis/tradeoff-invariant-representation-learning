# main.py

import os
import sys
import config
import traceback
from hal.utils import misc

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.plugins import DDPPlugin

import control
import hal.datasets as datasets

# torch.set_default_dtype(torch.float64)
# torch.set_default_tensor_type(torch.DoubleTensor)

def main():
    # parse the arguments
    args = config.parse_args()

    if args.ngpu == 0:
        args.device = 'cpu'

    pl.seed_everything(args.manual_seed)

    logger = TensorBoardLogger(
        save_dir=args.out_dir,
        # log_graph=True,
        name=args.project_name
    )

    dataloader = getattr(datasets, args.dataset)(args)
    # model = getattr(control, args.control_type)(args)
    model = getattr(control, args.control_type)(args, dataloader)

    # if args.resume is not None:
    #     model = model.load_from_checkpoint(args.resume)

    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(args.out_dir, 'checkpoints'),
        filename=args.project_name + '-{epoch:03d}-{val_loss:.3f}',
        # filepath=os.path.join(args.save_dir, args.project_name + '-{epoch:03d}-{val_loss:.3f}'),
        monitor='val_loss',
        save_top_k=3)

    if args.ngpu == 0:
        accelerator = None
        sync_batchnorm = False
    else:
        accelerator = 'ddp'
        sync_batchnorm = True

    trainer = pl.Trainer(
        gpus=args.ngpu,
        accelerator=accelerator,
        sync_batchnorm=sync_batchnorm,
        benchmark=True,
        # checkpoint_callback=True,
        # callbacks=checkpoint_callback,
        # plugins=DDPPlugin(find_unused_parameters=False),
        logger=logger,
        num_sanity_val_steps=0,
        min_epochs=1,
        max_epochs=args.nepochs,
        precision=args.precision,
        # reload_dataloaders_every_epoch=True,
        reload_dataloaders_every_n_epochs=1,
        check_val_every_n_epoch=args.check_val_every_n_epochs
        )

    trainer.fit(model, dataloader)

    model.data_flag = False
    # trainer.test(test_dataloaders=dataloader.test_train_dataloader()) # for 'sepehr' conda env
    trainer.test(dataloaders=dataloader.test_train_dataloader())
    model.data_flag = True
    # trainer.test(test_dataloaders=dataloader.test_dataloader()) # for 'sepehr' conda env
    trainer.test(dataloaders=dataloader.test_dataloader())
    # trainer.test()


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
