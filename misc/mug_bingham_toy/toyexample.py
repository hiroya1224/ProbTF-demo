#!/usr/bin/env python3
"""Measure and draw the five-column thesis-format learned-mug example.

Render and plot with system Python; inference uses the existing torch venv.
The plot reads a saved prediction packet and never runs the network itself.
"""
from pathlib import Path
import argparse
import hashlib
import json

import numpy as np


HERE=Path(__file__).resolve().parent
DEFAULT_OUTPUT=HERE/'results/toyexample_vgg16_10000'
DEFAULT_RUN=HERE/'runs/vgg16_10000_20261126'
REFERENCE=Path('/home/leus/dthesis/figs/symaware-bingham/toyexample.pdf')
REFERENCE_SOURCE=Path('/home/leus/dthesis/scripts/render_mug_learning_figure.py')
ANGLES=(0,60,90,120,180)
COLORS=('#c33130','#247b44','#2b64b2')
VIEW=np.array([[.8,.6,0.],[-.2683281572999748,.3577708763999664,.8944271909999159]])
CROP=(157,74,483,330)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def save_json(path,value):
    Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')


def render_inputs(args):
    from PIL import Image
    from render import MugRenderer
    from figure_geometry import camera_up_world_poses

    output=args.output.resolve()
    output.mkdir(parents=True,exist_ok=True)
    poses=camera_up_world_poses(ANGLES,elevation_degrees=10.)
    renderer=MugRenderer()
    images=[]
    try:
        for angle,q in zip(ANGLES,poses['q_camera_object_wxyz']):
            rgb,_=renderer.render(q)
            images.append(rgb)
            Image.fromarray(rgb,'RGB').save(output/f'input_{angle:03d}.png')
        metadata=renderer.metadata()
    finally:
        renderer.close()
    arrays={key:np.asarray(value) for key,value in poses.items()}
    arrays.update(rgb=np.stack(images),K=np.asarray(metadata['K']),
        t_camera=np.tile([0.,0.,2.5],(len(ANGLES),1)))
    np.savez_compressed(output/'inputs.npz',**arrays)
    save_json(output/'render.json',dict(renderer=metadata,
        reference_pdf=str(REFERENCE),reference_sha256=sha(REFERENCE) if REFERENCE.exists() else None,
        reference_source_sha256=sha(REFERENCE_SOURCE) if REFERENCE_SOURCE.exists() else None,
        source_sha256={name:sha(HERE/name) for name in ('render.py','figure_geometry.py','toyexample.py')},
        poses={key:value.tolist() for key,value in arrays.items() if key not in ('rgb',)},
        angle_convention='World yaw = 180 deg - displayed angle; camera elevation 10 deg',
        input_to_network='Whole original 640x360 RGB; shared crop is display only',
        shared_display_crop_xyxy=list(CROP),
        mesh_sha256=sha(HERE/'assets/mug_centered_colored.ply'),
        old_reference_images_or_predictions_used=False))
    print('Saved '+str(output/'inputs.npz'),flush=True)


def infer(args):
    import torch
    from train import load_checkpoint,normalize_images,seed_everything
    from model import VGG16Bingham
    from bingham import symmetric_matrix,PARAMETER_ORDER
    from figure_geometry import frame_change_bingham,mode_quaternion

    output=args.output.resolve()
    inputs=dict(np.load(output/'inputs.npz',allow_pickle=False))
    run=args.run.resolve()
    checkpoint_path=run/'best.pt'
    checkpoint=load_checkpoint(checkpoint_path)
    saved=json.loads((run/'test_metrics.json').read_text())
    seed_everything(20261126,4)
    device=torch.device(args.device)
    if device.type=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    model=VGG16Bingham(amp_bfloat16=device.type=='cuda' and torch.cuda.is_bf16_supported())
    model.load_state_dict(checkpoint['model'],strict=True)
    model.to(device=device,memory_format=torch.channels_last).eval()
    with torch.no_grad():
        parameters=model(normalize_images(torch.from_numpy(inputs['rgb']),device))
        camera_A=symmetric_matrix(parameters.double()).cpu().numpy()
        parameters=parameters.cpu().numpy()
    world_A=np.stack([frame_change_bingham(A,inputs['R_world_camera']) for A in camera_A])
    world_modes=np.stack([mode_quaternion(A) for A in world_A])
    camera_modes=np.stack([mode_quaternion(A) for A in camera_A])
    arrays=dict(inputs,parameters_9=parameters,A_camera=camera_A,A_world=world_A,
        mode_camera_wxyz=camera_modes,mode_world_wxyz=world_modes)
    np.savez_compressed(output/'figure_data.npz',**arrays)
    save_json(output/'inference.json',dict(checkpoint=str(checkpoint_path),checkpoint_sha256=sha(checkpoint_path),
        checkpoint_epoch=int(checkpoint['epoch']),trained_epochs=saved['epochs_completed'],
        checkpoint_selection='Saved best validation NLL; no selection using these five images',
        test_metrics=saved,input_sha256=sha(output/'inputs.npz'),figure_data_sha256=sha(output/'figure_data.npz'),
        source_sha256={name:sha(HERE/name) for name in ('model.py','train.py','bingham.py','figure_geometry.py','toyexample.py')},
        frozen_training_source_matches={name:sha(HERE/name)==digest for name,digest in checkpoint['source_sha256'].items()},
        pytorch_version=torch.__version__,device=str(device),
        amp_features='bfloat16' if model.amp_bfloat16 else 'float32',head_dtype=str(parameters.dtype),
        quaternion_order='wxyz',network_frame='object to OpenCV camera',
        display_frame='upright world related to camera by the saved fixed proper rotation',
        parameter_order=list(PARAMETER_ORDER),output_parameter_rows=[1,2,9],
        frame_change='A_world = L(q_world_camera) A_camera L(q_world_camera)^T',
        samples_from_training_or_test_predictions_reused=False,
        distribution_rescaled_or_refitted=False,network_evaluations_per_image=1))
    print('Saved '+str(output/'figure_data.npz'),flush=True)


def plot(args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from figure_geometry import half_height_quaternions,yaw_half_width,quaternion_matrix

    output=args.output.resolve()
    data=dict(np.load(output/'figure_data.npz',allow_pickle=False))
    reference=json.loads((output/'render.json').read_text())
    plt.rcParams.update({'font.family':['DejaVu Sans','IPAexGothic'],'pdf.fonttype':42,
                         'font.size':10,'axes.linewidth':.6})
    jp=dict(fontfamily='IPAexGothic')
    fig=plt.figure(figsize=(7.35,4.06),facecolor='white')
    left,step=.125,.173
    centers=left+step*(np.arange(5)+.5)
    def rule(x0,x1,y):
        fig.add_artist(Line2D([x0,x1],[y,y],transform=fig.transFigure,color='0.72',lw=.55,solid_capstyle='butt'))
    for start,stop,label in ((0,0,'取っ手が左'),(1,3,'取っ手が胴の後ろ'),(4,4,'取っ手が右')):
        fig.text((centers[start]+centers[stop])/2,.978,label,**jp,fontsize=11,ha='center',va='top')
        rule(centers[start]-step*.47,centers[stop]+step*.47,.920)
    for y in (.604,.326,.239,.011):
        rule(.008,centers[-1]+step*.47,y)
    for y,label in ((.875,'回転角'),(.722,'入力画像'),(.466,'方位分布'),(.283,'半値角 [deg]'),(.125,'出力')):
        fig.text(.013,y,label,va='center',**jp)
    parameter_rows=((0,.197),(1,.126),(8,.054))
    for component,y in parameter_rows:
        fig.text(.109,y,rf'$\theta_{{{component+1}}}$',ha='right',va='center',fontsize=10)
    circle=np.linspace(0,2*np.pi,181)
    geometry={}
    panels=[]
    x0,y0,x1,y1=CROP
    for j,(angle,image,A,mode,parameters) in enumerate(zip(data['angles_deg'],data['rgb'],data['A_world'],
            data['mode_world_wxyz'],data['parameters_9'])):
        cx=centers[j]
        fig.text(cx,.875,f'{int(angle)} [deg]',ha='center',va='center',fontsize=10.5)
        photo=fig.add_axes([cx-step*.49,.608,step*.98,.238])
        photo.imshow(image[y0:y1,x0:x1],interpolation='none')
        photo.axis('off')
        sphere=fig.add_axes([cx-step*.49,.330,step*.98,.268])
        sphere.set(aspect='equal',xlim=(-1.07,1.07),ylim=(-1.07,1.07))
        sphere.axis('off')
        for axis in range(3):
            ring=np.zeros((len(circle),3))
            ring[:,(axis+1)%3]=np.cos(circle)
            ring[:,(axis+2)%3]=np.sin(circle)
            sphere.plot(*(ring@VIEW.T).T,color='0.79',lw=.47)
        quaternions,diagnostic=half_height_quaternions(A,count=960)
        rotations=quaternion_matrix(quaternions)
        tips=np.einsum('ij,njk->nik',VIEW,rotations)
        peak=VIEW@quaternion_matrix(mode)
        for k,color in enumerate(COLORS):
            sphere.scatter(tips[:,0,k],tips[:,1,k],s=.75,color=color,linewidths=0,alpha=.30,rasterized=True,zorder=2)
            sphere.plot([0,peak[0,k]],[0,peak[1,k]],lw=1.05,color=color,zorder=3)
        half_width=yaw_half_width(A,mode=mode,axis=np.array([0.,0.,1.]))
        width=half_width['width_deg']
        fig.text(cx,.283,f'{width:.1f}' if width is not None else '>180',ha='center',va='center',fontsize=10)
        for component,y in parameter_rows:
            fig.text(cx,y,f'{parameters[component]:.1f}'.replace('-','−'),ha='center',va='center',fontsize=10)
        geometry[f'contour_{int(angle):03d}_wxyz']=quaternions
        geometry[f'axis_tips_{int(angle):03d}']=tips
        geometry[f'peak_{int(angle):03d}']=peak
        panels.append(dict(angle_deg=float(angle),parameters_9=parameters.tolist(),
            A_camera=data['A_camera'][j].tolist(),A_world=A.tolist(),mode_world_wxyz=mode.tolist(),
            contour=diagnostic,half_height=half_width))
    fig.savefig(output/'toyexample.pdf',dpi=280,metadata={'CreationDate':None,'ModDate':None})
    fig.savefig(output/'toyexample.png',dpi=200)
    plt.close(fig)
    np.savez_compressed(output/'geometry.npz',**geometry)
    save_json(output/'figure.json',dict(panels=panels,figure_inches=[7.35,4.06],
        reference_sha256=reference['reference_sha256'],
        reference_source_sha256=reference['reference_source_sha256'],
        input_packet_sha256=sha(output/'figure_data.npz'),geometry_sha256=sha(output/'geometry.npz'),
        source_sha256={name:sha(HERE/name) for name in ('figure_geometry.py','toyexample.py')},
        display_projection=VIEW.tolist(),shared_display_crop_xyxy=list(CROP),
        color_axis_mapping=dict(x=COLORS[0],y=COLORS[1],z=COLORS[2]),
        contour_meaning='Deterministic half-height orientations: q^T A_world q = lambda_max - log(2); not density samples',
        network_rerun_by_plot=False,distribution_modified_by_plot=False,monte_carlo=False))
    print('Saved '+str(output/'toyexample.pdf'),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('render','infer','plot'))
    parser.add_argument('--output',type=Path,default=DEFAULT_OUTPUT)
    parser.add_argument('--run',type=Path,default=DEFAULT_RUN)
    parser.add_argument('--device',default='cuda')
    args=parser.parse_args()
    {'render':render_inputs,'infer':infer,'plot':plot}[args.command](args)
