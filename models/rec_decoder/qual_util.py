import mcubes
import torch
import pytorch3d
from pytorch3d.transforms import RotateAxisAngle
from pytorch3d.structures import Meshes
import torchvision.utils as vutils
import numpy as np
import imageio
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras, 
    FoVOrthographicCameras,
    PointLights, 
    DirectionalLights, 
    Materials, 
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer,  
    SoftPhongShader,
    HardPhongShader,
    TexturesUV,
    TexturesVertex
)


def sdf_to_mesh(sdf, level=0.02, color=None, render_all=False):
    # device='cuda'
    device=sdf.device

    # extract meshes from sdf
    n_cell = sdf.shape[-1]
    bs, nc = sdf.shape[:2]

    assert nc == 1

    nimg_to_render = bs
    if not render_all:
        if bs > 16:
            print('Warning! Will not return all meshes')
        nimg_to_render = min(bs, 16) # no need to render that much..

    verts = []
    faces = []
    verts_rgb = []

    for i in range(nimg_to_render):
        sdf_i = sdf[i, 0].detach().cpu().numpy()
        # verts_i, faces_i = mcubes.marching_cubes(sdf_i, 0.02)
        verts_i, faces_i = mcubes.marching_cubes(sdf_i, level)
        verts_i = verts_i / n_cell - .5 

        verts_i = torch.from_numpy(verts_i).float().to(device)
        faces_i = torch.from_numpy(faces_i.astype(np.int64)).to(device)
        text_i = torch.ones_like(verts_i).to(device)
        if color is not None:
            for i in range(3):
                text_i[:, i] = color[i]

        verts.append(verts_i)
        faces.append(faces_i)
        verts_rgb.append(text_i)

    try:
        p3d_mesh = pytorch3d.structures.Meshes(verts, faces, textures=pytorch3d.renderer.Textures(verts_rgb=verts_rgb))
    except:
        p3d_mesh = None

    return p3d_mesh

def save_mesh_as_gif(mesh_renderer, mesh, nrow=3, out_name='1.gif', device='cuda'):
    """ save batch of mesh into gif """

    # img_comb = render_mesh(mesh_renderer, mesh, norm=False)    

    # rotate
    rot_comb = rotate_mesh_360(mesh_renderer, mesh, device=device) # save the first one
    
    # gather img into batches
    nimgs = len(rot_comb)
    nrots = len(rot_comb[0])
    H, W, C = rot_comb[0][0].shape
    rot_comb_img = []
    for i in range(nrots):
        img_grid_i = torch.zeros(nimgs, H, W, C)
        for j in range(nimgs):
            img_grid_i[j] = torch.from_numpy(rot_comb[j][i])
            
        img_grid_i = img_grid_i.permute(0, 3, 1, 2)
        img_grid_i = vutils.make_grid(img_grid_i, nrow=nrow)
        img_grid_i = img_grid_i.permute(1, 2, 0).numpy().astype(np.uint8)
            
        rot_comb_img.append(img_grid_i)
    
    with imageio.get_writer(out_name, mode='I', duration=.08) as writer:
        
        # combine them according to nrow
        for rot in rot_comb_img:
            writer.append_data(rot)

def rotate_mesh_360(mesh_renderer, mesh, device='cuda'):
    cur_mesh = mesh

    B = len(mesh.verts_list())
    ret = [ [] for i in range(B)]

    for i in range(36):
        cur_mesh = rotate_mesh(cur_mesh, device=device)
        img = render_mesh(mesh_renderer, cur_mesh, norm=False) # b c h w # important!! no norm here or they will not align
        img = img.permute(0, 2, 3, 1) # b h w c
        img = img.detach().cpu().numpy()
        img = (img * 255).astype(np.uint8)
        for j in range(B):
            ret[j].append(img[j])

    return ret

def rotate_mesh(mesh, axis='Y', angle=10, device='cuda'):
    rot_func = RotateAxisAngle(angle, axis, device=device)

    verts = mesh.verts_list()
    faces = mesh.faces_list()
    textures = mesh.textures
    
    B = len(verts)

    rot_verts = []
    for i in range(B):
        v = rot_func.transform_points(verts[i])
        rot_verts.append(v)
    new_mesh = Meshes(verts=rot_verts, faces=faces, textures=textures)
    return new_mesh

def render_mesh(renderer, mesh, color=None, norm=True):
    # verts: tensor of shape: B, V, 3
    # return: image tensor with shape: B, C, H, W
    if mesh.textures is None:
        verts = mesh.verts_list()
        verts_rgb_list = []
        for i in range(len(verts)):
        # print(verts.min(), verts.max())
            verts_rgb_i = torch.ones_like(verts[i])
            if color is not None:
                for i in range(3):
                    verts_rgb_i[:, i] = color[i]
            verts_rgb_list.append(verts_rgb_i)

        texture = pytorch3d.renderer.Textures(verts_rgb=verts_rgb_list)
        mesh.textures = texture

    images = renderer(mesh)
    return images.permute(0, 3, 1, 2)

def init_mesh_renderer(image_size=512, dist=3.5, elev=90, azim=90, camera='0', device='cuda:0'):
    # Initialize a camera.
    # With world coordinates +Y up, +X left and +Z in, the front of the cow is facing the -Z direction. 
    # So we move the camera by 180 in the azimuth direction so it is facing the front of the cow. 

    if camera == '0':
        # for vox orientation
        # dist, elev, azim = 1.7, 20, 20 # shapenet
        # dist, elev, azim = 3.5, 90, 90 # front view

        # dist, elev, azim = 3.5, 0, 135 # front view
        camera_cls = FoVPerspectiveCameras
    else:
        # dist, elev, azim = 5, 45, 135 # shapenet
        camera_cls = FoVOrthographicCameras
    
    R, T = look_at_view_transform(dist, elev, azim)
    cameras = camera_cls(device=device, R=R, T=T)

    # Define the settings for rasterization and shading. Here we set the output image to be of size
    # 512x512. As we are rendering images for visualization purposes only we will set faces_per_pixel=1
    # and blur_radius=0.0. We also set bin_size and max_faces_per_bin to None which ensure that 
    # the faster coarse-to-fine rasterization method is used. Refer to rasterize_meshes.py for 
    # explanations of these parameters. Refer to docs/notes/renderer.md for an explanation of 
    # the difference between naive and coarse-to-fine rasterization. 
    raster_settings = RasterizationSettings(
        image_size=image_size, 
        blur_radius=0, 
        faces_per_pixel=1, 
    )

    # Place a point light in front of the object. As mentioned above, the front of the cow is facing the 
    # -z direction. 
    lights = PointLights(device=device, location=[[1.0, 1.0, 0.0]])

    # Create a Phong renderer by composing a rasterizer and a shader. The textured Phong shader will 
    # interpolate the texture uv coordinates for each vertex, sample from a texture image and 
    # apply the Phong lighting model
    # renderer = MeshRenderer(
    #     rasterizer=MeshRasterizer(
    #         cameras=cameras, 
    #         raster_settings=raster_settings
    #     ),
    #     shader=SoftPhongShader(
    #         device=device, 
    #         cameras=cameras,
    #         lights=lights
    #     )
    # )
    renderer = MeshRenderer(
            rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
            shader=HardPhongShader(device=device, cameras=cameras)
        )
    return renderer
