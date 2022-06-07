import csv
import os
from models.v_backbone import build_v_backbone
from transforms.r2r_transforms import make_r2r_transforms
import hydra
import omegaconf
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

TSV_FIELDNAMES = ['scanId', 'viewpointId', 'image_w','image_h', 'vfov', 'features']
VIEWPOINT_SIZE = 36 # Number of discretized views from one viewpoint
FEATURE_SIZE = 2048
BATCH_SIZE = 4  # Some fraction of viewpoint size - batch size 4 equals 11GB memory

# Simulator image parameters
WIDTH=224
HEIGHT=224
VFOV=60


def generate_img_features(info):
    sys.path.insert(0, info['mattersim_build_dir'])
    #os.environ["MATTERPORT_DATA_DIR"] = info['matterport_data_dir']
    import MatterSim    
    sim = MatterSim.Simulator()
    sim.setCameraResolution(WIDTH, HEIGHT)
    sim.setCameraVFOV(math.radians(VFOV))
    sim.setDiscretizedViewingAngles(True)
    sim.setNavGraphPath(info['connectivity_dir'])
    sim.setDatasetPath(info['matterport_data_dir'])
    sim.setBatchSize(1)
    sim.initialize()
    
    for v_backbone_name in info['v_backbone_candidates']:
        cfg = info['candidate_configs']['ResNet_ImageNet']
        v_backbone = build_v_backbone(cfg)
        
        img_feature_tsv = os.path.join(info['img_feature_dir'], v_backbone_name + '1.tsv')
        with open(img_feature_tsv, 'wt') as tsvfile:
            writer = csv.DictWriter(tsvfile, delimiter = '\t', fieldnames = TSV_FIELDNAMES)

            count = 0
        
            # Loop all the viewpoints in the simulator
            viewpointIds = load_viewpointids(info)
            for scanId,viewpointId in viewpointIds:
                # Loop all discretized views from this location
                blobs = []
                features = np.empty([VIEWPOINT_SIZE, FEATURE_SIZE], dtype=np.float32)
                for ix in range(VIEWPOINT_SIZE):
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
                assert VIEWPOINT_SIZE % BATCH_SIZE == 0
                forward_passes = VIEWPOINT_SIZE // BATCH_SIZE
                ix = 0
                for f in range(forward_passes):
                    # Create batch for model
                    data = torch.zeros([BATCH_SIZE, 3, HEIGHT, WIDTH])
                    for n in range(BATCH_SIZE):
                        data[n,] = blobs[ix]
                        ix += 1                    
                    out = v_backbone(data)
                    out = out[-1]                    
                    out = F.adaptive_avg_pool2d(out, (1,1))
                    out = torch.squeeze(out)                                        
                    features[f*BATCH_SIZE : (f+1)*BATCH_SIZE, :] = out     
                    
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
                    
            break


def load_viewpointids(info):
    viewpointIds = []
    with open(os.path.join(info['connectivity_dir'], 'scans.txt')) as f:
        scans = [scan.strip() for scan in f.readlines()]
        for scan in scans:
            with open(os.path.join(info['connectivity_dir'], scan + '_connectivity.json'))  as j:
                data = json.load(j)
                for item in data:
                    if item['included']:
                        viewpointIds.append((scan, item['image_id']))
    print('Loaded %d viewpoints' % len(viewpointIds))
    return viewpointIds


# def transform_img(im):
#     ''' Prep opencv 3 channel image for the network '''
#     im = np.array(im, copy=True)
#     im_orig = im.astype(np.float32, copy=True)
#     im_orig -= np.array([[[103.1, 115.9, 123.2]]]) # BGR pixel mean
#     blob = np.zeros((1, im.shape[0], im.shape[1], 3), dtype=np.float32)
#     blob[0, :, :, :] = im_orig
#     blob = blob.transpose((0, 3, 1, 2))
#     return blob


# def build_tsv():
#     # # Set up the simulator
#     # sim = MatterSim.Simulator()
#     # sim.setCameraResolution(WIDTH, HEIGHT)
#     # sim.setCameraVFOV(math.radians(VFOV))
#     # sim.setDiscretizedViewingAngles(True)
#     # sim.setBatchSize(1)
#     # sim.initialize()

#     # # Set up Caffe resnet
#     # caffe.set_device(GPU_ID)
#     # caffe.set_mode_gpu()
#     # net = caffe.Net(PROTO, MODEL, caffe.TEST)
#     # net.blobs['data'].reshape(BATCH_SIZE, 3, HEIGHT, WIDTH)

#     # count = 0
#     # t_render = Timer()
#     # t_net = Timer()
#     with open(OUTFILE, 'wt') as tsvfile:
#         writer = csv.DictWriter(tsvfile, delimiter = '\t', fieldnames = TSV_FIELDNAMES)

#         # Loop all the viewpoints in the simulator
#         viewpointIds = load_viewpointids()
#         for scanId,viewpointId in viewpointIds:
#             t_render.start()
#             # Loop all discretized views from this location
#             blobs = []
#             features = np.empty([VIEWPOINT_SIZE, FEATURE_SIZE], dtype=np.float32)
#             for ix in range(VIEWPOINT_SIZE):
#                 if ix == 0:
#                     sim.newEpisode([scanId], [viewpointId], [0], [math.radians(-30)])
#                 elif ix % 12 == 0:
#                     sim.makeAction([0], [1.0], [1.0])
#                 else:
#                     sim.makeAction([0], [1.0], [0])

#                 state = sim.getState()[0]
#                 assert state.viewIndex == ix
                
#                 # Transform and save generated image
#                 blobs.append(transform_img(state.rgb))

#             t_render.stop()
#             t_net.start()
#             # Run as many forward passes as necessary
#             assert VIEWPOINT_SIZE % BATCH_SIZE == 0
#             forward_passes = VIEWPOINT_SIZE // BATCH_SIZE
#             ix = 0
#             for f in range(forward_passes):
#                 for n in range(BATCH_SIZE):
#                     # Copy image blob to the net
#                     net.blobs['data'].data[n, :, :, :] = blobs[ix]
#                     ix += 1
#                 # Forward pass
#                 output = net.forward()
#                 features[f*BATCH_SIZE:(f+1)*BATCH_SIZE, :] = net.blobs['pool5'].data[:,:,0,0]
#             writer.writerow({
#                 'scanId': scanId,
#                 'viewpointId': viewpointId,
#                 'image_w': WIDTH,
#                 'image_h': HEIGHT,
#                 'vfov' : VFOV,
#                 'features': str(base64.b64encode(features), "utf-8")
#             })
#             count += 1
#             t_net.stop()
#             if count % 100 == 0:
#                 print('Processed %d / %d viewpoints, %.1fs avg render time, %.1fs avg net time, projected %.1f hours' %\
#                   (count,len(viewpointIds), t_render.average_time, t_net.average_time, 
#                   (t_render.average_time+t_net.average_time)*len(viewpointIds)/3600))


def read_tsv(infile):
    # Verify we can read a tsv
    in_data = []
    with open(infile, "rt") as tsv_in_file:
        reader = csv.DictReader(tsv_in_file, delimiter='\t', fieldnames = TSV_FIELDNAMES)
        for item in reader:
            item['scanId'] = item['scanId']
            item['image_h'] = int(item['image_h'])
            item['image_w'] = int(item['image_w'])
            item['vfov'] = int(item['vfov'])
            item['features'] = np.frombuffer(base64.b64decode(item['features']),
                    dtype=np.float32).reshape((VIEWPOINT_SIZE, FEATURE_SIZE))
            in_data.append(item)
    return in_data


@hydra.main(config_path='../../configs/task', config_name='navigation.yaml')
def main(cfg):
    generate_img_features(cfg.dataset.info)
    outfile_path = os.path.join(cfg.dataset.info.img_feature_dir, cfg.dataset.info.matterport_data_dir[0])
    data = read_tsv(outfile_path)
    print('Completed %d viewpoints' % len(data))
    
    

if __name__ == '__main__':
    main()