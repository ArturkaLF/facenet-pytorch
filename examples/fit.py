import configparser
from facenet_pytorch import MTCNN, InceptionResnetV1, fixed_image_standardization

import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
import numpy as np
import os
from apex import amp
import time
import logging

import training

if __name__ == '__main__':

    start_time = time.time()

    # config
    CONFIG_PATH = 'cfg.txt'
    config = configparser.ConfigParser()
    config.read(CONFIG_PATH)
    training_name = config["Parameters"]["training_name"]

    # logging
    logging.basicConfig(filename=f"../Logs/{training_name}.log", level=logging.INFO, format='%(asctime)s %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p')
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s: %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # parameters
    data_dir = config["Parameters"]["data_dir"]
    batch_size = int(config["Parameters"]["batch_size"])
    epochs = int(config["Parameters"]["epochs"])
    opt = int(config["Parameters"]["opt_enabled"])
    opt_level = config["Parameters"]["opt_level"]
    cropping = int(config["Parameters"]["cropping"])
    loss_func = config["Parameters"]["loss_func"]
    pin_memory = bool(int(config["Parameters"]["pin_memory"]))

    # parameters mtcnn
    image_size_mtcnn = int(config["Parameters"]["image_size_mtcnn"])
    margin_mtcnn = int(config["Parameters"]["margin_mtcnn"])
    min_face_size_mtcnn = int(config["Parameters"]["min_face_size_mtcnn"])
    thresholds_mtcnn = [float(x) for x in config["Parameters"]["thresholds_mtcnn"].split(" ")]
    factor_mtcnn = float(config["Parameters"]["factor_mtcnn"])
    post_process_mtcnn = bool(int(config["Parameters"]["post_process_mtcnn"]))

    # cpu or cuda:0
    device = config["Parameters"]["device"]

    # docker
    docker = bool(int(config["Parameters"]["docker"]))

    logger.info('\n------------ Options -------------\n'
                f"training_name: {training_name}\n"
                f"data-dir: {data_dir}\n"
                f"Cropping: {cropping}\n"
                f"Opt: {opt}\n"
                f"Opt_level: {opt_level}\n"
                f"device: {device}\n"
                f"docker: {docker}\n"
                f"batch_size: {batch_size}\n"
                f"epochs: {epochs}\n"
                f"pin_memory: {pin_memory}\n"
                f"loss_func: {loss_func}\n"
                '-------------- End ---------------')

    if docker:
        workers = 0
        logger.info("Running with docker")
    else:
        workers = 0 if os.name == 'nt' else 8

    logger.warning(f'Running on device: {device} (no CUDA device)') if device == 'cpu' else \
        logger.info(f'Running on device: {device}')
    logger.info(f'Running on CUDA: {torch.cuda.is_available()}')

    dataset = datasets.ImageFolder(data_dir, transform=transforms.Resize((512, 512)))

    # Cropping
    if cropping:
        logger.info("Cropping")
        mtcnn = MTCNN(
            image_size=image_size_mtcnn, margin=margin_mtcnn, min_face_size=min_face_size_mtcnn,
            thresholds=thresholds_mtcnn, factor=factor_mtcnn, post_process=post_process_mtcnn,
            device=device
        )

        dataset.samples = [
            (p, p.replace(data_dir, data_dir + '_cropped'))
            for p, _ in dataset.samples
        ]

        loader = DataLoader(
            dataset,
            num_workers=workers,
            batch_size=batch_size,
            collate_fn=training.collate_pil
        )

        for i, (x, y) in enumerate(loader):
            mtcnn(x, save_path=y)
            logger.info(f'Batch {i + 1} of {len(loader)}')

    resnet = InceptionResnetV1(
        classify=True,
        num_classes=len(dataset.class_to_idx),
        pretrained='vggface2'
    ).to(device)

    optimizer = optim.Adam(resnet.parameters(), lr=0.001)

    if opt:
        model, optimizer = amp.initialize(resnet, optimizer, opt_level=opt_level)

    scheduler = MultiStepLR(optimizer, [5, 10])

    trans = transforms.Compose([
        np.float32,
        transforms.ToTensor(),
        fixed_image_standardization
    ])

    dataset = datasets.ImageFolder(data_dir + '_cropped', transform=trans)
    classes = [x for x in dataset.class_to_idx.keys()]

    img_inds = np.arange(len(dataset))
    np.random.shuffle(img_inds)
    train_inds = img_inds[:int(0.8 * len(img_inds))]
    val_inds = img_inds[int(0.8 * len(img_inds)):]

    subSet = SubsetRandomSampler(train_inds)

    train_loader = DataLoader(
        dataset,
        num_workers=workers,
        pin_memory=pin_memory,
        batch_size=batch_size,
        sampler=SubsetRandomSampler(train_inds)
    )
    val_loader = DataLoader(
        dataset,
        num_workers=workers,
        pin_memory=pin_memory,
        batch_size=batch_size,
        sampler=SubsetRandomSampler(val_inds)
    )

    if loss_func == "CrossEntropyLoss":
        loss_fn = torch.nn.CrossEntropyLoss()
    elif loss_func == "MultiMarginLoss":
        loss_fn = torch.nn.MultiMarginLoss()

    metrics = {
        'fps': training.BatchTimer(),
        'acc': training.accuracy
    }

    writer = SummaryWriter()
    writer.iteration, writer.interval = 0, 10

    logger.info('Initial')
    resnet.eval()
    training.pass_epoch(
        resnet, loss_fn, val_loader,
        batch_metrics=metrics, show_running=True, device=device,
        writer=writer,
        opt=opt
    )

    logger.info("Start training")

    for epoch in range(epochs):
        logger.info(f'Epoch {epoch + 1}/{epochs}')

        resnet.train()
        logger.info("Training")
        training.pass_epoch(
            resnet, loss_fn, train_loader, optimizer, scheduler,
            batch_metrics=metrics, show_running=True, device=device,
            writer=writer, opt=opt
        )

        logger.info("Validating")
        resnet.eval()
        training.pass_epoch(
            resnet, loss_fn, val_loader,
            batch_metrics=metrics, show_running=True, device=device,
            writer=writer, opt=opt, classes=classes, epoch=epoch + 1,
        )

    try:
        os.mkdir("checkpoints")
    except FileExistsError:
        logger.info("Dir checkpoints exists")

    # saving model's checkpoint
    if opt:
        checkpoint = {
            'model': resnet.state_dict(),
            'optimizer': optimizer.state_dict(),
            'amp': amp.state_dict()
        }
    else:
        checkpoint = {
            'model': resnet.state_dict(),
            'optimizer': optimizer.state_dict(),
        }

    torch.save(checkpoint, f"checkpoints/{training_name}.pt")

    # writer.close()
    all_time = time.time() - start_time
    logger.info(f"All time: {all_time}")
    logger.info("Finish\n")
