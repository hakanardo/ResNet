import argparse,logging,os
from random import random

import mxnet as mx
from symbol_resnet import resnet
import mxnet.metric
import numpy as np

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def multi_factor_scheduler(begin_epoch, epoch_size, step=[60, 75, 90], factor=0.1):
    step_ = [epoch_size * (x-begin_epoch) for x in step if x-begin_epoch > 0]
    return mx.lr_scheduler.MultiFactorScheduler(step=step_, factor=factor) if len(step_) else None


def main():
    if args.data_type == "cifar10":
        args.aug_level = 1
        args.num_classes = 10
        # depth should be one of 110, 164, 1001,...,which is should fit (args.depth-2)%9 == 0
        if((args.depth-2)%9 == 0 and args.depth >= 164):
            per_unit = [(args.depth-2)/9]
            filter_list = [16, 64, 128, 256]
            bottle_neck = True
        elif((args.depth-2)%6 == 0 and args.depth < 164):
            per_unit = [(args.depth-2)/6]
            filter_list = [16, 16, 32, 64]
            bottle_neck = False
        else:
            raise ValueError("no experiments done on detph {}, you can do it youself".format(args.depth))
        units = per_unit*3
        symbol = resnet(units=units, num_stage=3, filter_list=filter_list, num_class=args.num_classes,
                        data_type="cifar10", bottle_neck = bottle_neck, bn_mom=args.bn_mom, workspace=args.workspace,
                        memonger=args.memonger)
    elif args.data_type == "imagenet":
        args.num_classes = 1000
        if args.depth == 9:
            units = [1, 1, 1, 1]
        elif args.depth == 18:
            units = [2, 2, 2, 2]
        elif args.depth == 34:
            units = [3, 4, 6, 3]
        elif args.depth == 50:
            units = [3, 4, 6, 3]
        elif args.depth == 101:
            units = [3, 4, 23, 3]
        elif args.depth == 152:
            units = [3, 8, 36, 3]
        elif args.depth == 200:
            units = [3, 24, 36, 3]
        elif args.depth == 269:
            units = [3, 30, 48, 8]
        else:
            raise ValueError("no experiments done on detph {}, you can do it youself".format(args.depth))
        label = mx.sym.Variable(name='softmax_label')
        ids = mx.sym.Variable(name='softmax_label_ids')
        symbol = resnet(units=units, num_stage=4, filter_list=[64, 256, 512, 1024, 2048] if args.depth >=50
                        else [64, 64, 128, 256, 512], num_class=args.num_classes, data_type="imagenet", bottle_neck = True
                        if args.depth >= 50 else False, bn_mom=args.bn_mom, workspace=args.workspace,
                        memonger=args.memonger, label=label)
        symbol = mx.sym.Group([symbol, ids])
    else:
         raise ValueError("do not support {} yet".format(args.data_type))
    kv = mx.kvstore.create(args.kv_store)
    devs = [mx.cpu()] if args.gpus is None else [mx.gpu(int(i)) for i in args.gpus.split(',')]
    epoch_size = max(int(args.num_examples / args.batch_size / kv.num_workers), 1)
    begin_epoch = args.model_load_epoch if args.model_load_epoch else 0
    if not os.path.exists("./model"):
        os.mkdir("./model")
    model_prefix = "model/resnet-bootstrapped-{}-{}-{}".format(args.data_type, args.depth, kv.rank)
    checkpoint = mx.callback.do_checkpoint(model_prefix)
    arg_params = None
    aux_params = None
    if args.retrain:
        _, arg_params, aux_params = mx.model.load_checkpoint(model_prefix, args.model_load_epoch)
    if args.memonger:
        import memonger
        symbol = memonger.search_plan(symbol, data=(args.batch_size, 3, 32, 32) if args.data_type=="cifar10"
                                                    else (args.batch_size, 3, 224, 224))
    train = mx.io.ImageRecordIter(
        path_imgrec         = os.path.join(args.data_dir, "cifar10_train_ids.rec") if args.data_type == 'cifar10' else
                              os.path.join(args.data_dir, "train_256_q90_ids.rec") if args.aug_level == 1
                              else os.path.join(args.data_dir, "train_480_q90_ids.rec"),
        label_width         = 2,
        data_name           = 'data',
        label_name          = 'softmax_label',
        data_shape          = (3, 32, 32) if args.data_type=="cifar10" else (3, 224, 224),
        batch_size          = 1,
        pad                 = 4 if args.data_type == "cifar10" else 0,
        fill_value          = 127,  # only used when pad is valid
        rand_crop           = True,
        max_random_scale    = 1.0,  # 480 with imagnet, 32 with cifar10
        min_random_scale    = 1.0 if args.data_type == "cifar10" else 1.0 if args.aug_level == 1 else 0.533,  # 256.0/480.0
        max_aspect_ratio    = 0 if args.data_type == "cifar10" else 0 if args.aug_level == 1 else 0.25,
        random_h            = 0 if args.data_type == "cifar10" else 0 if args.aug_level == 1 else 36,  # 0.4*90
        random_s            = 0 if args.data_type == "cifar10" else 0 if args.aug_level == 1 else 50,  # 0.4*127
        random_l            = 0 if args.data_type == "cifar10" else 0 if args.aug_level == 1 else 50,  # 0.4*127
        max_rotate_angle    = 0 if args.aug_level <= 2 else 10,
        max_shear_ratio     = 0 if args.aug_level <= 2 else 0.1,
        rand_mirror         = True,
        shuffle             = True,
        num_parts           = kv.num_workers,
        preprocess_threads  = len(devs) * 5,
        part_index          = kv.rank)
    val = mx.io.ImageRecordIter(
        path_imgrec         = os.path.join(args.data_dir, "cifar10_val.rec") if args.data_type == 'cifar10' else
                              os.path.join(args.data_dir, "val_256_q90.rec"),
        label_width         = 1,
        data_name           = 'data',
        label_name          = 'softmax_label',
        batch_size          = args.batch_size,
        data_shape          = (3, 32, 32) if args.data_type=="cifar10" else (3, 224, 224),
        rand_crop           = False,
        rand_mirror         = False,
        num_parts           = kv.num_workers,
        part_index          = kv.rank)
    model = mx.model.FeedForward(
        epoch_size          = 15,
        ctx                 = devs,
        symbol              = symbol,
        arg_params          = arg_params,
        aux_params          = aux_params,
        num_epoch           = args.epochs,
        begin_epoch         = begin_epoch,
        learning_rate       = args.lr,
        momentum            = args.mom,
        wd                  = args.wd,
        optimizer           = 'nag',
        # optimizer          = 'sgd',
        initializer         = mx.init.Xavier(rnd_type='gaussian', factor_type="in", magnitude=2),
        lr_scheduler        = multi_factor_scheduler(begin_epoch, epoch_size, step=[120, 160], factor=0.1)
                             if args.data_type=='cifar10' else
                             multi_factor_scheduler(begin_epoch, epoch_size, step=[30, 60, 90], factor=0.1),
        )
    bootstrap_metric = BootstrapMetric()
    model.fit(
        X                  = BootstrapIter(train, args.batch_size, bootstrap_metric),
        eval_data          = val,
        eval_metric        = ['acc', 'ce'] if args.data_type=='cifar10' else
                            [BAccuracy(name='acc', label_names=['softmax_label'], output_names=['data']),
                             BTopKAccuracy(name='top_k_accuracy', top_k=5, label_names=['softmax_label'], output_names=['data']),
                             bootstrap_metric],
        kvstore            = kv,
        batch_end_callback = mx.callback.Speedometer(args.batch_size, args.frequent),
        epoch_end_callback = checkpoint)
    # logging.info("top-1 and top-5 acc is {}".format(model.score(X = val,
    #               eval_metric = ['acc', mx.metric.create('top_k_accuracy', top_k = 5)])))

class BootstrapIter(mx.io.DataIter):
    def __init__(self, image_iter, batch_size, bootstrap_metric):
        self.image_iter = image_iter
        self.batch_size = image_iter.batch_size
        self.bootstrap_metric = bootstrap_metric
        self.batch_size = batch_size
        self.data_shape = (batch_size,) + self.image_iter.provide_data[0].shape[1:]

    def next(self):
        pok = self.bootstrap_metric.pok
        data = mx.nd.empty(self.data_shape)
        labels = mx.nd.empty(self.batch_size)
        ids = mx.nd.empty(self.batch_size, dtype=int)
        i = 0
        while i < self.batch_size:
            b = self.image_iter.next()
            sample_id = int(b.label[0][:,1].asscalar())
            if random() < pok[sample_id] * 0.5:
                continue
            ids[i] = sample_id
            labels[i] = b.label[0][:,0]
            data[i] = b.data[0][0]
            i += 1
        return mxnet.io.DataBatch([data], [labels, ids], self.batch_size - i)

    def reset(self):
        self.image_iter.reset()

    @property
    def provide_data(self):
        d, =  self.image_iter.provide_data
        return [mx.io.DataDesc(d.name, self.data_shape, d.dtype, d.layout),]

    @property
    def provide_label(self):
        d, = self.image_iter.provide_label
        return [mx.io.DataDesc(d.name, (self.batch_size,), d.dtype, d.layout),
                mx.io.DataDesc(d.name+'_ids', (self.batch_size,), d.dtype, d.layout)]


class BAccuracy(mx.metric.Accuracy):
    def update(self, labels, preds):
        return mx.metric.Accuracy.update(self, [labels[0]], [preds[0]])

class BTopKAccuracy(mx.metric.TopKAccuracy):
    def update(self, labels, preds):
        return mx.metric.TopKAccuracy.update(self, [labels[0]], [preds[0]])

class BootstrapMetric(mx.metric.EvalMetric):
    def __init__(self, maxid=100):
        mx.metric.EvalMetric.__init__(self, "bs")
        self.pok = np.zeros(maxid, np.float32)
        self.alfa = 0.5

    def update(self, labels, preds):
        if len(labels) == 2: # If we are training and not validating
            ids = labels[1].asnumpy()
            labels = labels[0].asnumpy()
            preds = preds[0].asnumpy()
            self.pok[ids] = self.alfa * self.pok[ids] + (1-self.alfa) * (np.argmax(preds, axis=1) == labels)

    def get(self):
        return ["bs_0.1", "bs_0.5", "bs_0.9"], map(int, [np.sum(self.pok > 0.1), np.sum(self.pok > 0.5), np.sum(self.pok > 0.9)])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="command for training resnet-v2")
    parser.add_argument('--gpus', type=str, default=None, help='the gpus will be used, e.g "0,1,2,3"')
    parser.add_argument('--data-dir', type=str, default='./data/imagenet/', help='the input data directory')
    parser.add_argument('--data-type', type=str, default='imagenet', help='the dataset type')
    parser.add_argument('--list-dir', type=str, default='./',
                        help='the directory which contain the training list file')
    parser.add_argument('--lr', type=float, default=0.1, help='initialization learning reate')
    parser.add_argument('--mom', type=float, default=0.9, help='momentum for sgd')
    parser.add_argument('--bn-mom', type=float, default=0.9, help='momentum for batch normlization')
    parser.add_argument('--wd', type=float, default=0.0001, help='weight decay for sgd')
    parser.add_argument('--batch-size', type=int, default=256, help='the batch size')
    parser.add_argument('--workspace', type=int, default=512, help='memory space size(MB) used in convolution, if xpu '
                        ' memory is oom, then you can try smaller vale, such as --workspace 256')
    parser.add_argument('--depth', type=int, default=50, help='the depth of resnet')
    parser.add_argument('--num-classes', type=int, default=1000, help='the class number of your task')
    parser.add_argument('--aug-level', type=int, default=2, choices=[1, 2, 3],
                        help='level 1: use only random crop and random mirror\n'
                             'level 2: add scale/aspect/hsv augmentation based on level 1\n'
                             'level 3: add rotation/shear augmentation based on level 2')
    parser.add_argument('--num-examples', type=int, default=1281167, help='the number of training examples')
    parser.add_argument('--kv-store', type=str, default='device', help='the kvstore type')
    parser.add_argument('--model-load-epoch', type=int, default=0,
                        help='load the model on an epoch using the model-load-prefix')
    parser.add_argument('--frequent', type=int, default=50, help='frequency of logging')
    parser.add_argument('--epochs', type=int, default=200, help='number of epochs')
    parser.add_argument('--memonger', action='store_true', default=False,
                        help='true means using memonger to save momory, https://github.com/dmlc/mxnet-memonger')
    parser.add_argument('--retrain', action='store_true', default=False, help='true means continue training')
    args = parser.parse_args()
    logging.info(args)

    main()
