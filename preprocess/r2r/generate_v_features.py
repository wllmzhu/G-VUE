import csv
import os
from models.v_backbone import build_v_backbone
from transforms.r2r_transforms import make_r2r_transforms
from models.r2r_decoder.constants import DISCRETIZED_VIEWS, VFOV, TSV_FIELDNAMES
import hydra
import torch
import torch.nn.functional as F
from PIL import Image

import numpy as np
import cv2
import json
import math
import base64
import sys

csv.field_size_limit(sys.maxsize)

def generate_img_features(cfg):
    img_feature_tsv = cfg.precompute_v_feature.data.tsv_path
    print(img_feature_tsv)
    if os.path.exists(img_feature_tsv):
        if cfg.precompute_v_feature.use_cached:
            print('Using cache for R2R visual features')
            return
        else:
            print('Clearing cache for R2R visual features, generating a new one')
            os.remove(img_feature_tsv)

    BATCH_SIZE = cfg.precompute_v_feature.batch_size  # Some fraction of viewpoint size - batch size 4 equals 11GB memory

    # Simulator image parameters
    WIDTH=cfg.model.image_size
    HEIGHT=cfg.model.image_size

    sys.path.insert(0, cfg.simulator.build_path)
    import MatterSim    
    sim = MatterSim.Simulator()
    sim.setCameraResolution(WIDTH, HEIGHT)
    sim.setCameraVFOV(math.radians(VFOV))
    sim.setDiscretizedViewingAngles(True)
    sim.setNavGraphPath(cfg.simulator.data.conn_path)
    sim.setDatasetPath(cfg.simulator.data.scans_path)
    sim.setBatchSize(1)
    sim.initialize()

    v_backbone = build_v_backbone(cfg.model.v_backbone)
    v_backbone.requires_pyramid = True
    v_backbone.cuda()
    FEATURE_SIZE = cfg.model.v_sizes[-1] #size of the last extracted pyramid layer

    with open(img_feature_tsv, 'wt') as tsvfile:
        writer = csv.DictWriter(tsvfile, delimiter = '\t', fieldnames = TSV_FIELDNAMES)

        count = 0

        # Loop all the viewpoints in the simulator
        viewpointIds = load_viewpointids(cfg)
        for scanId,viewpointId in viewpointIds:
            # Loop all discretized views from this location
            blobs = []
            features = np.empty([DISCRETIZED_VIEWS, FEATURE_SIZE], dtype=np.float32)
            for ix in range(DISCRETIZED_VIEWS):
                if ix == 0:
                    sim.newEpisode([scanId], [viewpointId], [0], [math.radians(-30)])
                elif ix % 12 == 0:
                    sim.makeAction([0], [1.0], [1.0])
                else:
                    sim.makeAction([0], [1.0], [0])

                state = sim.getState()[0]
                assert state.viewIndex == ix

                img = np.array(state.rgb, copy=True)
                img = Image.fromarray(img)
                img, _ = make_r2r_transforms()(img, None)

                # import matplotlib.pyplot as plt
                # fig = plt.figure()
                # plt.imshow(img.permute(1, 2, 0))
                # plt.show()
                # fig.savefig(f'hi{ix}.jpg') # Use fig. here

                blobs.append(img)

            # Run forward passes to get all 36 images for 1 pano
            assert DISCRETIZED_VIEWS % BATCH_SIZE == 0
            forward_passes = DISCRETIZED_VIEWS // BATCH_SIZE
            ix = 0
            for f in range(forward_passes):
                # Create batch for model
                data = torch.zeros([BATCH_SIZE, 3, HEIGHT, WIDTH])
                for n in range(BATCH_SIZE):
                    data[n,] = blobs[ix]
                    ix += 1
                data = data.cuda()                    
                out = v_backbone(data)
                out = out[-1]                    
                out = F.adaptive_avg_pool2d(out, (1,1))
                out = torch.squeeze(out)                                        
                features[f*BATCH_SIZE : (f+1)*BATCH_SIZE, :] = out.cpu()   

            writer.writerow({
                'scanId': scanId,
                'viewpointId': viewpointId,
                'image_w': WIDTH,
                'image_h': HEIGHT,
                'vfov' : VFOV,
                'features': str(base64.b64encode(features), "utf-8")
            })
            count += 1
            if count % 100 == 0:
                print(f'Processed {count}, {len(viewpointIds)} viewpoints')


def load_viewpointids(cfg):
    viewpointIds = []
    with open(os.path.join(cfg.simulator.data.conn_path, 'scans.txt')) as f:
        scans = [scan.strip() for scan in f.readlines()]
        for scan in scans:
            with open(os.path.join(cfg.simulator.data.conn_path, scan + '_connectivity.json'))  as j:
                data = json.load(j)
                for item in data:
                    if item['included']:
                        viewpointIds.append((scan, item['image_id']))
    print('Loaded %d viewpoints' % len(viewpointIds))
    return viewpointIds


def read_tsv(cfg):
    # is_ResNet = infile.find('ResNet') != -1
    # FEATURE_SIZE = 2048 if is_ResNet else 768
    # print(f'Feature size set to {FEATURE_SIZE}')
    FEATURE_SIZE = cfg.model.v_sizes[-1]

    # Verify we can read a tsv
    in_data = []
    with open(cfg.precompute_v_feature.data.tsv_path, "rt") as tsv_in_file:
        reader = csv.DictReader(tsv_in_file, delimiter='\t', fieldnames = TSV_FIELDNAMES)
        for item in reader:
            item['scanId'] = item['scanId']
            item['image_h'] = int(item['image_h'])
            item['image_w'] = int(item['image_w'])
            item['vfov'] = int(item['vfov'])
            item['features'] = np.frombuffer(base64.b64decode(item['features']),
                    dtype=np.float32).reshape((DISCRETIZED_VIEWS, FEATURE_SIZE))
            in_data.append(item)
            break
    return in_data


@hydra.main(config_path='../../configs', config_name='r2r.yaml')
def main(cfg):
    generate_img_features(cfg)
    # data = read_tsv(cfg)
    # print(data[0]['features'].shape)
    # print('Checked %d viewpoints to be loadable' % len(data))


if __name__ == '__main__':
    main() 