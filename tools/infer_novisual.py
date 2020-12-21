import argparse

import cv2
import numpy as np
import torch

from vedacore.image import imread, imwrite
from vedacore.misc import Config, color_val, load_weights
from vedacore.parallel import collate, scatter
from vedadet.datasets.pipelines import Compose
from vedadet.engines import build_engine


def parse_args():
    parser = argparse.ArgumentParser(description='Infer a detector')
    parser.add_argument('config', help='config file path')
    parser.add_argument('imgname', help='image file name')

    args = parser.parse_args()
    return args


def prepare(cfg):
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
    else:
        device = 'cpu'
    engine = build_engine(cfg.infer_engine)

    engine.model.to(device)
    load_weights(engine.model, cfg.weights.filepath)

    data_pipeline = Compose(cfg.data_pipeline)
    return engine, data_pipeline, device


def bbx_result(result, imgfp, class_names, i):

    bboxes = np.vstack(result)
    labels = [
        np.full(bbox.shape[0], idx, dtype=np.int32)
        for idx, bbox in enumerate(result)
    ]
    labels = np.concatenate(labels)
    outfp = imgfp[:-7] + "submit/" + str(i) + ".txt"
    fo = open(outfp, "w")

    for bbox, label in zip(bboxes, labels):
        bbox_int = bbox[:4].astype(np.int32)

        label_text = class_names[
            label] if class_names is not None else f'cls {label}'
        print(i)
        if len(bbox) > 4:
            label_text += f'|{bbox[-1]:.02f}'
        print(f'{bbox[-1]:.02f}')

        STR = class_names + " " + f'{bbox[-1]:.02f}' + " " + str(bbox_int[0]) + " " + str(bbox_int[1]) + " " + str(
            bbox_int[2]) + " " + str(bbox_int[3]) + "\n"
        fo.write(STR)

def main():

    args = parse_args()
    cfg = Config.fromfile(args.config)
    imgname = args.imgname
    class_names = cfg.class_names

    engine, data_pipeline, device = prepare(cfg)
    for i in range(42500, 50000):
        STR = imgname + str(i) + ".jpg"
        data = dict(img_info=dict(filename=STR), img_prefix=None)
        data = data_pipeline(data)
        data = collate([data], samples_per_gpu=1)
        if device != 'cpu':
            # scatter to specified GPU
            data = scatter(data, [device])[0]
        else:
            # just get the actual data from DataContainer
            data['img_metas'] = data['img_metas'][0].data
            data['img'] = data['img'][0].data
        result = engine.infer(data['img'], data['img_metas'])[0]
        bbx_result(result, imgname, class_names, i)


if __name__ == '__main__':
    main()