#!/usr/bin/env python3
"""Start the two recordings x two requested views in four owned tmux panes."""
from pathlib import Path
import argparse
import shlex
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]


def launch(session,output,grid_sizes=(3,5,9,17,33,65),nodes=128):
    found = subprocess.run(['tmux','has-session','-t',session],
                           stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    if found.returncode == 0:
        raise RuntimeError('Session already exists; attach to it or choose another --session')
    output.mkdir(parents=True,exist_ok=True)
    for index,(case,mode) in enumerate((('failure1','3d'),('failure1','plane'),
                                       ('failure2','3d'),('failure2','plane'))):
        log = output/(case+'_'+mode+'.log')
        arguments = ['env','PYTHONDONTWRITEBYTECODE=1','OPENBLAS_NUM_THREADS=1',
            'OMP_NUM_THREADS=1','MKL_NUM_THREADS=1','MPLCONFIGDIR='+str(output/('.mpl-'+case+'-'+mode)),
            'nice','-n','10','/usr/bin/python3','-u',str(HERE/'run_imu_pid_maps.py'),
            '--case',case,'--mode',mode,'--output',str(output),'--nodes',str(nodes),
            '--grid-sizes',*[str(n) for n in grid_sizes]]
        script = 'set -o pipefail\n'+shlex.join(arguments)+' 2>&1 | tee '+shlex.quote(str(log))
        command = shlex.join(['bash','-lc',script])
        if index == 0:
            subprocess.run(['tmux','new-session','-d','-s',session,'-n','maps',
                            '-c',str(ROOT),command],check=True)
            subprocess.run(['tmux','set-option','-w','-t',session+':maps',
                            'remain-on-exit','on'],check=True)
        else:
            subprocess.run(['tmux','split-window','-h','-t',session+':maps',
                            '-c',str(ROOT),command],check=True)
        subprocess.run(['tmux','select-pane','-t',session+':maps',
                        '-T',case+' / '+('3D' if mode=='3d' else 'I=1')],check=True)
        subprocess.run(['tmux','select-layout','-t',session+':maps','tiled'],check=True)
    subprocess.run(['tmux','set-option','-w','-t',session+':maps',
                    'pane-border-status','top'],check=True)
    subprocess.run(['tmux','set-option','-w','-t',session+':maps',
                    'pane-border-format','#{pane_title}'],check=True)
    print('Started:',session)
    print('Attach:',shlex.join(['tmux','attach','-t',session]))
    print('Output:',output)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session',default='grape-imu-only-pid-20261106')
    parser.add_argument('--output',type=Path,default=Path('/tmp/grape-imu-only-pid-20261106'))
    parser.add_argument('--grid-sizes',nargs='+',type=int,default=[3,5,9,17,33,65])
    parser.add_argument('--nodes',type=int,default=128)
    args=parser.parse_args()
    launch(args.session,args.output.resolve(),args.grid_sizes,args.nodes)


if __name__=='__main__': main()
