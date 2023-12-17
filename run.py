import os, argparse, time, glob, pickle, subprocess, shlex, io, pprint

import numpy as np
import pandas
import tqdm
import fire

import torch
import torch.nn as nn
import torch.utils.model_zoo
import torchvision

import cornet

from PIL import Image
Image.warnings.simplefilter('ignore')

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
np.random.seed(0)
torch.manual_seed(0)

torch.backends.cudnn.benchmark = True
normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])

parser = argparse.ArgumentParser(description='ImageNet Training')
parser.add_argument('--data_path', required=True,
                    help='path to ImageNet folder that contains train and val folders')
parser.add_argument('-o', '--output_path', default=None,
                    help='path for storing ')
parser.add_argument('--model', choices=['Z', 'R', 'RT', 'S', 'S100'], default='Z',
                    help='which model to train')
parser.add_argument('--times', default=5, type=int,
                    help='number of time steps to run the model (only R model)')
parser.add_argument('--outer_times', default=1, type=int,
                    help='number of outer time steps to run the model (only S100 model)')
parser.add_argument('--ngpus', default=0, type=int,
                    help='number of GPUs to use; 0 if you want to run on CPU')
parser.add_argument('-j', '--workers', default=4, type=int,
                    help='number of data loading workers')
parser.add_argument('--epochs', default=20, type=int,
                    help='number of total epochs to run')
parser.add_argument('--batch_size', default=256, type=int,
                    help='mini-batch size')
parser.add_argument('--lr', '--learning_rate', default=.1, type=float,
                    help='initial learning rate')
parser.add_argument('--step_size', default=10, type=int,
                    help='after how many epochs learning rate should be decreased 10x')
parser.add_argument('--momentum', default=.9, type=float, help='momentum')
parser.add_argument('--weight_decay', default=1e-4, type=float,
                    help='weight decay ')
parser.add_argument('--ckpt_path', default='map_clsloc.txt', type=str,
                    help='path to model ckpt file')
parser.add_argument('--classes_path', default='map_clsloc.txt', type=str,
                    help='path to class mappings file')


FLAGS, FIRE_FLAGS = parser.parse_known_args()


def set_gpus(n=1):
    """
    Finds all GPUs on the system and restricts to n of them that have the most
    free memory.
    """
    gpus = subprocess.run(shlex.split(
        'nvidia-smi --query-gpu=index,memory.free,memory.total --format=csv,nounits'), check=True, stdout=subprocess.PIPE).stdout
    gpus = pandas.read_csv(io.BytesIO(gpus), sep=', ', engine='python')
    gpus = gpus[gpus['memory.total [MiB]'] > 5000]  # only above 5 GB
    if os.environ.get('CUDA_VISIBLE_DEVICES') is not None:
        visible = [int(i)
                   for i in os.environ['CUDA_VISIBLE_DEVICES'].split(',')]
        gpus = gpus[gpus['index'].isin(visible)]
    gpus = gpus.sort_values(by='memory.free [MiB]', ascending=False)
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'  # making sure GPUs are numbered the same way as in nvidia_smi
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(
        [str(i) for i in gpus['index'].iloc[:n]])


if FLAGS.ngpus > 0:
    set_gpus(FLAGS.ngpus)


def get_model(pretrained=False):
    map_location = None if FLAGS.ngpus > 0 else 'cpu'
    model = getattr(cornet, f'cornet_{FLAGS.model.lower()}')
    if FLAGS.model.lower() == 'r':
        model = model(pretrained=pretrained, map_location=map_location, times=FLAGS.times)
    elif FLAGS.model.lower() == 's100':
        model = model(pretrained=pretrained, map_location=map_location, V4_to_V1_times=FLAGS.outer_times)
    else:
        model = model(pretrained=pretrained, map_location=map_location)

    if FLAGS.ngpus == 0:
        model = model.module  # remove DataParallel
    if FLAGS.ngpus > 0:
        model = model.cuda()
    return model


def train(restore_path=None,  # useful when you want to restart training
          save_train_epochs=.5,  # how often save output during training
          save_val_epochs=.5,  # how often save output during validation
          save_model_epochs=1,  # how often save model weigths
          save_model_secs=None  # how often save model (in sec)
          ):
    print("Start training")
    model = get_model()
    print(model)
    trainer = ImageNetTrain(model)
    validator = ImageNetVal(model)
    
    if FLAGS.output_path is not None:
        os.makedirs(FLAGS.output_path, exist_ok=True)
    
    start_epoch = 0
    if restore_path is not None:
        print("Restoring from", restore_path)
        ckpt_data = torch.load(restore_path)
        start_epoch = ckpt_data['epoch']
        model.load_state_dict(ckpt_data['state_dict'])
        trainer.optimizer.load_state_dict(ckpt_data['optimizer'])

    records = []
    if FLAGS.output_path is not None:
        pickle_path = os.path.join(FLAGS.output_path, 'results.pkl')
        if os.path.exists(pickle_path):
            with open(pickle_path, 'rb') as f:
                records = pickle.load(f)
            
    recent_time = time.time()

    nsteps = len(trainer.data_loader)
    if save_train_epochs is not None:
        save_train_steps = (np.arange(0, FLAGS.epochs + 1,
                                      save_train_epochs) * nsteps).astype(int)
    if save_val_epochs is not None:
        save_val_steps = (np.arange(0, FLAGS.epochs + 1,
                                    save_val_epochs) * nsteps).astype(int)
    if save_model_epochs is not None:
        save_model_steps = (np.arange(0, FLAGS.epochs + 1,
                                      save_model_epochs) * nsteps).astype(int)

    results = {'meta': {'step_in_epoch': 0,
                        'epoch': start_epoch,
                        'wall_time': time.time()}
               }
    for epoch in tqdm.trange(0, FLAGS.epochs + 1, initial=start_epoch, desc='epoch'):
        data_load_start = np.nan
        for step, data in enumerate(tqdm.tqdm(trainer.data_loader, desc=trainer.name)):
            data_load_time = time.time() - data_load_start
            global_step = epoch * len(trainer.data_loader) + step

            if epoch < FLAGS.epochs:
                frac_epoch = (global_step + 1) / len(trainer.data_loader)
                record = trainer(frac_epoch, *data)
                record['data_load_dur'] = data_load_time
                results = {'meta': {'step_in_epoch': step + 1,
                                    'epoch': frac_epoch,
                                    'wall_time': time.time()}
                           }
                if save_train_steps is not None:
                    if step in save_train_steps:
                        results[trainer.name] = record
                        print("\ntrain:", results[trainer.name])
                        
                if save_val_steps is not None:
                    if global_step in save_val_steps:
                        results[validator.name] = validator()
                        print("val:", results[validator.name])
                        trainer.model.train()

                ckpt_data = {}
                ckpt_data['flags'] = FLAGS.__dict__.copy()
                ckpt_data['epoch'] = epoch
                ckpt_data['state_dict'] = model.state_dict()
                ckpt_data['optimizer'] = trainer.optimizer.state_dict()

                if save_model_secs is not None:
                    if time.time() - recent_time > save_model_secs:
                        torch.save(ckpt_data, os.path.join(FLAGS.output_path,
                                                        'latest_checkpoint.pth.tar'))
                        recent_time = time.time()

                if save_model_steps is not None:
                    if global_step in save_model_steps:
                        torch.save(ckpt_data, os.path.join(FLAGS.output_path,
                                                        f'epoch_{epoch:02d}.pth.tar'))
                        
                        if FLAGS.output_path is not None:
                            records.append(results)
                            if len(results) > 1:
                                pickle.dump(records, open(os.path.join(FLAGS.output_path, 'results.pkl'), 'wb'))

                else:
                    if len(results) > 1:
                        pprint.pprint(results)

            data_load_start = time.time()


def val(restore_path=None):
    print("Start validation")
    if FLAGS.ckpt_path is None:
        print("Load default pretrained model")
        model = get_model(pretrained=True).cpu()
    else:
        print("Restore from", FLAGS.ckpt_path)
        model = get_model(pretrained=False).cpu()
        ckpt_data = torch.load(FLAGS.ckpt_path)
        ckpt_data['state_dict'] = {k.replace('module.', ''): v for k, v in ckpt_data['state_dict'].items()}
        model.load_state_dict(ckpt_data['state_dict'])
        
    validator = ImageNetVal(model)

    record = validator()
    print("val:", record)


def load_class_labels(file_path):
    class_labels = {}
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split(' ')
            index = int(parts[1]) - 1  # subtract 1 for 0-based indexing
            label = parts[2]
            class_labels[index] = label
    return class_labels


def test(layer='Decoder', sublayer='output', time_step=0, imsize=224):
    """
    Suitable for small image sets. If you have thousands of images or it is
    taking too long to extract features, consider using
    `torchvision.datasets.ImageFolder`, using `ImageNetVal` as an example.

    Kwargs:
        - layers (choose from: V1, V2, V4, IT, decoder)
        - sublayer (e.g., output, conv1, avgpool)
        - time_step (which time step to use for storing features)
        - imsize (resize image to how many pixels, default: 224)
    """
    # model = get_model(pretrained=True).cpu()
    model = get_model(pretrained=False).cpu()
    
    # Load model weights from ckpt
    ckpt_data = torch.load(FLAGS.ckpt_path)
    # Adjust the keys
    ckpt_data['state_dict'] = {k.replace('module.', ''): v for k, v in ckpt_data['state_dict'].items()}
    model.load_state_dict(ckpt_data['state_dict'])
    
    transform = torchvision.transforms.Compose([
                    torchvision.transforms.Resize((imsize, imsize)),
                    torchvision.transforms.ToTensor(),
                    normalize,
                ])
    model.eval()

    def _store_feats(layer, inp, output):
        """An ugly but effective way of accessing intermediate model features
        """
        output = output.cpu().numpy()
        _model_feats.append(np.reshape(output, (len(output), -1)))

    try:
        m = model.module
    except:
        m = model
    model_layer = getattr(getattr(m, layer), sublayer)
    model_layer.register_forward_hook(_store_feats)

    model_feats = []
    with torch.no_grad():
        model_feats = []
        if os.path.isfile(FLAGS.data_path):
            fnames = [FLAGS.data_path]
        else:
            fnames = sorted(glob.glob(os.path.join(FLAGS.data_path, '*.JPEG')))
        if len(fnames) == 0:
            raise FileNotFoundError(f'No files found in {FLAGS.data_path}')
        for fname in tqdm.tqdm(fnames):
            try:
                im = Image.open(fname).convert('RGB')
            except:
                raise FileNotFoundError(f'Unable to load {fname}')
            im = transform(im)
            im = im.unsqueeze(0)  # adding extra dimension for batch size of 1
            _model_feats = []
            model(im)
            model_feats.append(_model_feats[time_step])
        model_feats = np.concatenate(model_feats)

    if FLAGS.output_path is not None:
        os.makedirs(FLAGS.output_path, exist_ok=True)
        fname = f'CORnet-{FLAGS.model}_{layer}_{sublayer}_feats.npy'
        np.save(os.path.join(FLAGS.output_path, fname), model_feats)
    
    # Load the class labels
    class_labels = load_class_labels(FLAGS.classes_path)

    # Get the class prediction indices from the model
    prediction_idxs = np.argmax(model_feats, axis=1)
    print(prediction_idxs)

    # Map prediction to a class label
    prediction_labels = [class_labels[idx] for idx in prediction_idxs]
    
    # Print unique class labels
    print(set(prediction_labels))
    


class ImageNetTrain(object):

    def __init__(self, model):
        self.name = 'train'
        self.model = model
        self.data_loader = self.data()
        self.optimizer = torch.optim.SGD(self.model.parameters(),
                                         FLAGS.lr,
                                         momentum=FLAGS.momentum,
                                         weight_decay=FLAGS.weight_decay)
        self.lr = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=FLAGS.step_size)
        self.loss = nn.CrossEntropyLoss()
        if FLAGS.ngpus > 0:
            self.loss = self.loss.cuda()

    def data(self):
        dataset = torchvision.datasets.ImageFolder(
            os.path.join(FLAGS.data_path, 'train'),
            torchvision.transforms.Compose([
                torchvision.transforms.RandomResizedCrop(224),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ToTensor(),
                normalize,
            ]))
        data_loader = torch.utils.data.DataLoader(dataset,
                                                  batch_size=FLAGS.batch_size,
                                                  shuffle=True,
                                                  num_workers=FLAGS.workers,
                                                  pin_memory=True)
        return data_loader

    def __call__(self, frac_epoch, inp, target):
        start = time.time()

        self.lr.step(epoch=frac_epoch)
        if FLAGS.ngpus > 0:
            target = target.cuda(non_blocking=True)
        output = self.model(inp)

        record = {}
        loss = self.loss(output, target)
        record['loss'] = loss.item()
        record['top1'], record['top5'] = accuracy(output, target, topk=(1, 5))
        record['top1'] /= len(output)
        record['top5'] /= len(output)
        record['learning_rate'] = self.lr.get_lr()[0]

        self.optimizer.zero_grad()
        loss.backward()
        
        if np.isclose(frac_epoch % 1.0, 0.0, atol=1e-3):
            # Loop to print average gradients per layer in each block
            module = next(self.model.named_children())[1]
            for block_name, block in module.named_children():
                print(f"Block: {block_name}")

                for layer_name, layer in block.named_children():
                    total_grad = 0
                    total_params = 0

                    # Iterate through parameters in each layer
                    for parameter in layer.parameters():
                        if parameter.grad is not None:
                            total_grad += parameter.grad.abs().sum()
                            total_params += parameter.numel()

                    # Calculate and print the average gradient
                    if total_params > 0:
                        avg_grad = total_grad / total_params
                        print(f"\tLayer {layer_name} average gradient: {avg_grad}")
            
        self.optimizer.step()

        record['dur'] = time.time() - start
        return record


class ImageNetVal(object):

    def __init__(self, model):
        self.name = 'val'
        self.model = model
        self.data_loader = self.data()
        self.loss = nn.CrossEntropyLoss(size_average=False)
        if FLAGS.ngpus > 0:
            self.loss = self.loss.cuda()

    def data(self):
        dataset = torchvision.datasets.ImageFolder(
            os.path.join(FLAGS.data_path, 'val'),
            torchvision.transforms.Compose([
                torchvision.transforms.Resize(256),
                torchvision.transforms.CenterCrop(224),
                torchvision.transforms.ToTensor(),
                normalize,
            ]))
        data_loader = torch.utils.data.DataLoader(dataset,
                                                  batch_size=FLAGS.batch_size,
                                                  shuffle=False,
                                                  num_workers=FLAGS.workers,
                                                  pin_memory=True)

        return data_loader

    def __call__(self):
        self.model.eval()
        start = time.time()
        record = {'loss': 0, 'top1': 0, 'top5': 0}
        with torch.no_grad():
            for (inp, target) in tqdm.tqdm(self.data_loader, desc=self.name):
                if FLAGS.ngpus > 0:
                    target = target.cuda(non_blocking=True)
                output = self.model(inp)

                record['loss'] += self.loss(output, target).item()
                p1, p5 = accuracy(output, target, topk=(1, 5))
                record['top1'] += p1
                record['top5'] += p5

        for key in record:
            record[key] /= len(self.data_loader.dataset.samples)
        record['dur'] = (time.time() - start) / len(self.data_loader)

        return record


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        _, pred = output.topk(max(topk), dim=1, largest=True, sorted=True)
        pred = pred.t()
        print(pred)
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = [correct[:k].sum().item() for k in topk]
        return res


if __name__ == '__main__':
    fire.Fire(command=FIRE_FLAGS)
