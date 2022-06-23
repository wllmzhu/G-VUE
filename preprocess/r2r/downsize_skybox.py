#!/usr/bin/env python3

''' Script for downsizing skybox images. '''

import os
import math
import cv2
import numpy as np
from multiprocessing import Pool
from depth_to_skybox import camera_parameters
import hydra
from itertools import repeat


NUM_WORKER_PROCESSES = 20

# Constants
SKYBOX_WIDTH = 1024
SKYBOX_HEIGHT = 1024
skybox_template = '%s/%s/matterport_skybox_images/%s_skybox%d_sami.jpg'
skybox_small_template = '%s/%s/matterport_skybox_images/%s_skybox%d_small.jpg'
skybox_merge_template = '%s/%s/matterport_skybox_images/%s_skybox_small.jpg'


def downsizeWithMerge(scan, cfg_var):
  # Load pano ids
  intrinsics,_ = camera_parameters(scan, cfg_var)
  pano_ids = list(set([item.split('_')[0] for item in intrinsics.keys()]))
  print('Processing scan %s with %d panoramas' % (scan, len(pano_ids)))

  for pano in pano_ids:

    ims = []
    for skybox_ix in range(6):

      # Load and downsize skybox image
      skybox = cv2.imread(skybox_template % (cfg_var['base_dir'],scan,pano,skybox_ix))
      ims.append(cv2.resize(skybox,(cfg_var['DOWNSIZED_WIDTH'],cfg_var['DOWNSIZED_HEIGHT']),interpolation=cv2.INTER_AREA))

    # Save output
    newimg = np.concatenate(ims, axis=1)
    assert cv2.imwrite(skybox_merge_template % (cfg_var['base_dir'],scan,pano), newimg)


def downsize(scan, cfg_var):

  # Load pano ids
  intrinsics,_ = camera_parameters(scan)
  pano_ids = list(set([item.split('_')[0] for item in intrinsics.keys()]))
  print('Processing scan %s with %d panoramas' % (scan, len(pano_ids)))

  for pano in pano_ids:

    for skybox_ix in range(6):

      # Load and downsize skybox image
      skybox = cv2.imread(skybox_template % (cfg_var['base_dir'],scan,pano,skybox_ix))
      newimg = cv2.resize(skybox,(cfg_var['DOWNSIZED_WIDTH'],cfg_var['DOWNSIZED_HEIGHT']),interpolation=cv2.INTER_AREA)

      # Save output
      assert cv2.imwrite(skybox_small_template % (cfg_var['base_dir'],scan,pano,skybox_ix), newimg)


@hydra.main(config_path='../../configs', config_name='r2r.yaml')
def main(cfg):
  cfg_var = {
    'base_dir': cfg.simulator.data.scans_path,
    'DOWNSIZED_WIDTH': cfg.model.image_size,
    'DOWNSIZED_HEIGHT': cfg.model.image_size,
  }

  with open(os.path.join(cfg.simulator.data.conn_path, 'scans.txt')) as f:
    scans = [scan.strip() for scan in f.readlines()]
    p = Pool(NUM_WORKER_PROCESSES)
    p.starmap(downsizeWithMerge, zip(scans, repeat(cfg_var)))


if __name__ == '__main__':
    main()
    