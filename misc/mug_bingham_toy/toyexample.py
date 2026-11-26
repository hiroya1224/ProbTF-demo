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
DEFAULT_OUTPUT=HERE/'results/toyexample_vgg16_10000_cylinder_center_no_blur'
DEFAULT_RUN=HERE/'runs/vgg16_10000_cylinder_center_no_blur_20261127'
REFERENCE=Path('/home/leus/dthesis/figs/symaware-bingham/toyexample.pdf')
REFERENCE_SOURCE=Path('/home/leus/dthesis/scripts/render_mug_learning_figure.py')
ANGLES=(0,60,90,120,180)
COLORS=('#c33130','#247b44','#2b64b2')
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
    poses=camera_up_world_poses(ANGLES,elevation_degrees=getattr(args,'elevation_degrees',0.))
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
    packet_path=output/'inputs.npz'
    unchanged=False
    if packet_path.exists():
        with np.load(packet_path,allow_pickle=False) as previous:
            unchanged=set(previous.files)==set(arrays) and all(
                np.array_equal(previous[key],value) for key,value in arrays.items())
    # Re-rendering identical inputs must not change the frozen viewer's hash
    # merely through new ZIP-entry timestamps at final training completion.
    if not unchanged:
        np.savez_compressed(packet_path,**arrays)
    save_json(output/'render.json',dict(renderer=metadata,
        reference_pdf=str(REFERENCE),reference_sha256=sha(REFERENCE) if REFERENCE.exists() else None,
        reference_source_sha256=sha(REFERENCE_SOURCE) if REFERENCE_SOURCE.exists() else None,
        source_sha256={name:sha(HERE/name) for name in ('render.py','figure_geometry.py','toyexample.py')},
        poses={key:value.tolist() for key,value in arrays.items() if key not in ('rgb',)},
        angle_convention=f"World yaw = 180 deg - displayed angle; camera elevation {poses['elevation_degrees']:g} deg",
        input_to_network='Whole original 640x360 RGB; shared crop is display only',
        shared_display_crop_xyxy=list(CROP),
        mesh_sha256=metadata['mesh']['colored_mesh_sha256'],
        old_reference_images_or_predictions_used=False))
    print('Saved '+str(output/'inputs.npz'),flush=True)


def preview(args):
    from PIL import Image, ImageDraw, ImageFont

    render_inputs(args)
    x0,y0,x1,y1=CROP
    width,height=x1-x0,y1-y0
    sheet=Image.new('RGB',(len(ANGLES)*width,height+36),'white')
    draw=ImageDraw.Draw(sheet)
    font=ImageFont.truetype('DejaVuSans.ttf',22)
    for j,angle in enumerate(ANGLES):
        with Image.open(args.output/f'input_{angle:03d}.png') as rgb:
            sheet.paste(rgb.crop(CROP),(j*width,36))
        label=f'{angle} deg'
        box=draw.textbbox((0,0),label,font=font)
        draw.text((j*width+(width-box[2]+box[0])/2,3),label,fill='black',font=font)
    path=args.output/'render_preview.png'
    sheet.save(path)
    print('Saved '+str(path.resolve()),flush=True)


def infer(args):
    import torch
    from train import load_checkpoint,normalize_images,seed_everything
    from model import VGG16Bingham
    from bingham import symmetric_matrix,PARAMETER_ORDER,BinghamNLL
    from figure_geometry import frame_change_bingham,mode_quaternion
    from tangent_audit_math import direction_tangent_covariance

    output=args.output.resolve()
    inputs=dict(np.load(output/'inputs.npz',allow_pickle=False))
    run=args.run.resolve()
    checkpoint_path=run/'best.pt'
    # The trainer atomically replaces best.pt. Hold one inode for both the
    # loaded weights and their hash, without modifying or copying its bytes.
    with checkpoint_path.open('rb') as stream:
        checkpoint=load_checkpoint(stream)
        stream.seek(0)
        checksum=hashlib.sha256()
        for block in iter(lambda:stream.read(1024*1024),b''):
            checksum.update(block)
        checkpoint_hash=checksum.hexdigest()
    saved_path=run/'test_metrics.json'
    saved=json.loads(saved_path.read_text()) if saved_path.exists() else None
    training_rows=[json.loads(line) for line in (run/'metrics.jsonl').read_text().splitlines()
                   if line.strip()]
    checkpoint_epoch=int(checkpoint['epoch'])
    epoch_metrics=next((row for row in reversed(training_rows) if row['epoch']==checkpoint_epoch),
        dict(train_nll_mean=None,validation=dict(nll_mean=checkpoint['best_val_nll'])))
    seed_everything(20261126,4)
    device=torch.device(args.device)
    if device.type=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    model=VGG16Bingham(amp_bfloat16=device.type=='cuda' and torch.cuda.is_bf16_supported())
    model.load_state_dict(checkpoint['model'],strict=True)
    model.to(device=device,memory_format=torch.channels_last).eval()
    with torch.no_grad():
        parameters=model(normalize_images(torch.from_numpy(inputs['rgb']),device))
        camera_matrix=symmetric_matrix(parameters.double())
        log_Z=BinghamNLL(device=device).log_normalizer(camera_matrix).cpu().numpy()
        camera_A=camera_matrix.cpu().numpy()
        parameters=parameters.cpu().numpy()
    world_A=np.stack([frame_change_bingham(A,inputs['R_world_camera']) for A in camera_A])
    world_modes=np.stack([mode_quaternion(A) for A in world_A])
    camera_modes=np.stack([mode_quaternion(A) for A in camera_A])
    arrays=dict(inputs,parameters_9=parameters,A_camera=camera_A,A_world=world_A,
        mode_camera_wxyz=camera_modes,mode_world_wxyz=world_modes,log_Z_s3=log_Z)
    np.savez_compressed(output/'figure_data.npz',**arrays)
    save_json(output/'inference.json',dict(checkpoint=str(checkpoint_path),checkpoint_sha256=checkpoint_hash,
        checkpoint_epoch=checkpoint_epoch,trained_epochs=training_rows[-1]['epoch'],
        training_complete=saved is not None,
        checkpoint_selection='Saved best validation NLL; no selection using these five images',
        test_metrics=saved,input_sha256=sha(output/'inputs.npz'),figure_data_sha256=sha(output/'figure_data.npz'),
        source_sha256={name:sha(HERE/name) for name in ('model.py','train.py','bingham.py','figure_geometry.py','toyexample.py')},
        frozen_training_source_matches={name:sha(HERE/name)==digest for name,digest in checkpoint['source_sha256'].items()},
        pytorch_version=torch.__version__,device=str(device),
        amp_features='bfloat16' if model.amp_bfloat16 else 'float32',head_dtype=str(parameters.dtype),
        quaternion_order='wxyz',network_frame='object to OpenCV camera',
        display_frame='same OpenCV camera as network prediction and RGB input: x right, y image-down, z forward',
        log_Z_s3_method='Same 200-node Chen-Tanaka deterministic normalizer as training NLL',
        parameter_order=list(PARAMETER_ORDER),output_parameter_rows=[1,2,9],
        frame_change='A_world = L(q_world_camera) A_camera L(q_world_camera)^T',
        samples_from_training_or_test_predictions_reused=False,
        distribution_rescaled_or_refitted=False,network_evaluations_per_image=1))
    # This is a separate requested-elevation display history. The trainer's
    # frozen ten-degree observer and its metrics remain untouched. All views
    # and tangent covariances below refer to the one forward pass above.
    geometry_keys=('K','t_camera','R_world_camera','R_camera_world','R_world_object',
                   'R_camera_object','q_world_camera_wxyz','q_world_object_wxyz',
                   'q_camera_object_wxyz','elevation_degrees')
    observed=dict(epoch=checkpoint_epoch,optimizer_step=checkpoint.get('optimizer_step'),
        weights='saved best validation-NLL checkpoint; independent display inference',
        checkpoint_sha256=checkpoint_hash,inputs_path=str(output/'inputs.npz'),
        inputs_sha256=sha(output/'inputs.npz'),angles_deg=inputs['angles_deg'].tolist(),
        input_geometry={key:inputs[key].tolist() for key in geometry_keys if key in inputs},
        status='observed',parameters_9=parameters.tolist(),A_camera=camera_A.tolist())
    for label,vector in (('X',(1.,0.,0.)),('Z',(0.,0.,1.))):
        observed[label]=[direction_tangent_covariance(A,v=vector) for A in camera_A]
    record=dict(epoch=checkpoint_epoch,train_nll_mean=epoch_metrics['train_nll_mean'],
        validation=epoch_metrics['validation'],tangent_xz=observed)
    metrics_path=output/'metrics.jsonl'
    existing=[json.loads(line) for line in metrics_path.read_text().splitlines()
              if line.strip()] if metrics_path.exists() else []
    if not any(row['tangent_xz'].get('checkpoint_sha256')==checkpoint_hash for row in existing):
        with metrics_path.open('a') as stream:
            stream.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+'\n')
    if saved is not None:
        save_json(output/'test_metrics.json',dict(saved,source_training_run=str(run)))
    print('Saved '+str(output/'figure_data.npz'),flush=True)


def plot(args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from figure_geometry import yaw_half_width,quaternion_matrix
    from sphere_display import GRID_COLOR,GRID_ALPHA,GRID_LINEWIDTH
    from sample_display import sample_camera_axes,draw_sample_sphere

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
    camera_projection=np.array([[1.,0.,0.],[0.,1.,0.]])
    yaw_axis_camera=data['R_camera_world']@np.array([0.,0.,1.])
    geometry={}
    panels=[]
    x0,y0,x1,y1=CROP
    for j,(angle,image,A,mode,parameters) in enumerate(zip(data['angles_deg'],data['rgb'],data['A_camera'],
            data['mode_camera_wxyz'],data['parameters_9'])):
        cx=centers[j]
        fig.text(cx,.875,f'{int(angle)} [deg]',ha='center',va='center',fontsize=10.5)
        photo=fig.add_axes([cx-step*.49,.608,step*.98,.238])
        photo.imshow(image[y0:y1,x0:x1],interpolation='none')
        photo.axis('off')
        sphere=fig.add_axes([cx-step*.49,.330,step*.98,.268])
        sphere.set(aspect='equal',xlim=(-1.07,1.07),ylim=(1.07,-1.07))
        sphere.axis('off')
        samples=sample_camera_axes(A,count=args.samples,seed=(args.sample_seed+j) % (1 << 64))
        mode_rotation=quaternion_matrix(mode)
        eigenvalues=np.linalg.eigvalsh(A)
        artists=draw_sample_sphere(sphere,samples,
            mode_rotation if eigenvalues[-1]>eigenvalues[-2] else None,COLORS,mode_linewidth=1.8)
        for name,artist in artists.items():
            if name.endswith('_support'):
                artist.set_rasterized(True)
            elif name.endswith('_points'):
                for collection in artist:
                    collection.set_rasterized(True)
        peak=camera_projection@mode_rotation
        half_width=yaw_half_width(A,mode=mode,axis=yaw_axis_camera)
        width=half_width['width_deg']
        fig.text(cx,.283,f'{width:.1f}' if width is not None else '—',ha='center',va='center',fontsize=10)
        for component,y in parameter_rows:
            fig.text(cx,y,f'{parameters[component]:.1f}'.replace('-','−'),ha='center',va='center',fontsize=10)
        geometry[f'quaternions_camera_wxyz_{int(angle):03d}']=samples['quaternions']
        geometry[f'axis_tips_camera_{int(angle):03d}']=samples['tips']
        geometry[f'sample_seed_{int(angle):03d}']=np.asarray(samples['seed'],dtype=np.uint64)
        geometry[f'peak_{int(angle):03d}']=peak
        panels.append(dict(angle_deg=float(angle),parameters_9=parameters.tolist(),
            A_camera=A.tolist(),mode_camera_wxyz=mode.tolist(),
            sphere_samples=samples['metadata'],sample_display=artists['display_metadata'],half_height=half_width))
    fig.savefig(output/'toyexample.pdf',dpi=280,metadata={'CreationDate':None,'ModDate':None})
    fig.savefig(output/'toyexample.png',dpi=200)
    plt.close(fig)
    np.savez_compressed(output/'geometry.npz',**geometry)
    save_json(output/'figure.json',dict(panels=panels,figure_inches=[7.35,4.06],
        reference_sha256=reference['reference_sha256'],
        reference_source_sha256=reference['reference_source_sha256'],
        input_packet_sha256=sha(output/'figure_data.npz'),geometry_sha256=sha(output/'geometry.npz'),
        source_sha256={name:sha(HERE/name) for name in ('figure_geometry.py','sphere_display.py','sample_display.py','toyexample.py')},
        display_projection=camera_projection.tolist(),camera_image_y_down=True,
        shared_display_crop_xyxy=list(CROP),
        color_axis_mapping=dict(x=COLORS[0],y=COLORS[1],z=COLORS[2]),
        sample_meaning='Shared BACG quaternion samples from saved A_camera; each sample rotates object X/Y/Z, giving the displayed camera-frame axis tips',
        display_alpha='all axis-tip points alpha=1; far camera Z>0 points use white-mixed RGB at the same hue and 1.5 times HSL lightness; mode axes fully opaque at 1.8pt',
        display_grid=f'Latitude/longitude wireframe, camera-Z poles; gray {GRID_COLOR}, alpha {GRID_ALPHA}, width {GRID_LINEWIDTH}; far wires use one-fifth opacity',
        network_rerun_by_plot=False,distribution_modified_by_plot=False,
        random_sampling_for_visualization_only=True,covariance_estimated_from_samples=False))
    print('Saved '+str(output/'toyexample.pdf'),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('render','preview','infer','plot'))
    parser.add_argument('--output',type=Path)
    parser.add_argument('--run',type=Path,default=DEFAULT_RUN)
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--elevation-degrees',type=float,default=0.,
                        help='Display-input camera elevation in degrees; default 0 is horizontal')
    parser.add_argument('--samples',type=int,default=4096,help='Display quaternion samples per panel; does not affect inference')
    parser.add_argument('--sample-seed',type=int,default=20261127,help='Local display seed; panel index is added')
    args=parser.parse_args()
    if not np.isfinite(args.elevation_degrees):
        parser.error('--elevation-degrees must be finite')
    if args.samples < 1:
        parser.error('--samples must be positive')
    if not 0 <= args.sample_seed < 1 << 64:
        parser.error('--sample-seed must be an unsigned 64-bit integer')
    if args.output is None:
        args.output=HERE/'examples/cylinder_pointlight_preview' if args.command=='preview' else DEFAULT_OUTPUT
    {'render':render_inputs,'preview':preview,'infer':infer,'plot':plot}[args.command](args)
