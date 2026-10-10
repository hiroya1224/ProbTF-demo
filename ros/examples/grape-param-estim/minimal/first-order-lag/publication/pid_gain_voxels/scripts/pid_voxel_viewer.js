'use strict';
function nodeIndex(p,i,d,shape){return (p*shape[1]+i)*shape[2]+d}
function prepare(g){
 if(g.c)return g;
 const raw=atob(g.breaks);g.b=new Uint8Array(raw.length);g.c=new Uint8Array(raw.length);
 if(raw.length!==g.nodes)throw Error('Gain data length differs');
 for(let k=0;k<raw.length;k++){g.b[k]=raw.charCodeAt(k);g.c[k]=1-g.b[k]/g.baseline_count>=DATA.alpha?3:2}
 return g;
}
function surfaceGeometry(g,cutaway=true){
 prepare(g);const start=cutaway?g.cut_index:0;
 const out={2:{x:[],y:[],z:[],i:[],j:[],k:[],customdata:[],faces:0},3:{x:[],y:[],z:[],i:[],j:[],k:[],customdata:[],faces:0},
            cut2:{x:[],y:[],z:[],i:[],j:[],k:[],customdata:[],faces:0},cut3:{x:[],y:[],z:[],i:[],j:[],k:[],customdata:[],faces:0}};
 const corners=[[[0,0,0],[0,0,1],[0,1,1],[0,1,0]],[[1,0,0],[1,1,0],[1,1,1],[1,0,1]],
 [[0,0,0],[1,0,0],[1,0,1],[0,0,1]],[[0,1,0],[0,1,1],[1,1,1],[1,1,0]],
 [[0,0,0],[0,1,0],[1,1,0],[1,0,0]],[[0,0,1],[1,0,1],[1,1,1],[0,1,1]]];
 const neighbors=[[-1,0,0],[1,0,0],[0,-1,0],[0,1,0],[0,0,-1],[0,0,1]];
 function cls(p,i,d){return p<0||i<start||d<0||p>=g.shape[0]||i>=g.shape[1]||d>=g.shape[2]?0:g.c[nodeIndex(p,i,d,g.shape)]}
 for(let p=0;p<g.shape[0];p++)for(let i=start;i<g.shape[1];i++)for(let d=0;d<g.shape[2];d++){
  const k=nodeIndex(p,i,d,g.shape),c=g.c[k];
  const detail=[g.axes[0][p],g.axes[1][i],g.axes[2][d],100*(1-g.b[k]/g.baseline_count)];
  for(let f=0;f<6;f++){
   const step=neighbors[f];if(cls(p+step[0],i+step[1],d+step[2])===c)continue;
   const mesh=out[cutaway&&i===start&&f===2?'cut'+c:c],offset=mesh.x.length;
   for(const corner of corners[f]){mesh.x.push(g.edges[0][p+corner[0]]);mesh.y.push(cutaway&&i+corner[1]===start?g.cut_i:g.edges[1][i+corner[1]]);mesh.z.push(g.edges[2][d+corner[2]]);mesh.customdata.push(detail)}
   mesh.i.push(offset,offset);mesh.j.push(offset+1,offset+2);mesh.k.push(offset+2,offset+3);mesh.faces++;
  }
 }
 return out;
}
function plotTraces(g,cutaway=true){
 const mesh=surfaceGeometry(g,cutaway),traces=[];
 for(const key of [2,3,'cut2','cut3']){const code=String(key).endsWith('3')?3:2,m=mesh[key],name=code===3?'green':'red';
  traces.push({type:'mesh3d',...m,color:DATA.styles[name],opacity:DATA.styles[name+'_opacity'],flatshading:true,
   lighting:{ambient:1,diffuse:0,specular:0,fresnel:0,roughness:1},showlegend:false,
   hovertemplate:'P=%{customdata[0]}<br>I=%{customdata[1]}<br>D=%{customdata[2]}<br>rate : %{customdata[3]:.1f}%<extra></extra>'});
 }
 const same=g.marks.success.gain.every((v,k)=>v===g.marks.failure.gain[k]);
 for(const name of ['failure','success']){const q=g.marks[name].gain;
  traces.push({type:'scatter3d',name,mode:'markers',x:[q[0]],y:[q[1]],z:[q[2]],showlegend:false,
   marker:{color:DATA.styles[name==='success'?'success_color':'recorded_color'],
    symbol:name==='success'?'diamond':same?'circle-open':'x',size:name==='success'?8:same?14:10,opacity:1,
    line:{color:name==='success'?'white':DATA.styles.recorded_color,width:2}},
   hovertemplate:(name==='success'?'Success':'Failure')+'<br>P='+q[0]+'<br>I='+q[1]+'<br>D='+q[2]+'<br>rate : '+(100*g.marks[name].survival).toFixed(1)+'%<extra></extra>'});
 }
 return traces;
}
if(typeof document!=='undefined'){
 const $=id=>document.getElementById(id),initialCamera={eye:{x:1.45,y:-1.75,z:1.0}},config={responsive:true,displaylogo:false};
 let group='roll_pitch',syncing=false,camera=initialCamera,listenersAttached=false;
 const selected=()=>DATA.cases.filter(g=>g.group===group);
 function layout(g){const cutaway=$('cutaway').checked;
  const ranges=g.axes.map((a,k)=>{const lo=cutaway&&k===1?g.cut_i:a[0],s=a.at(-1)-lo;return[lo-.1*s,a.at(-1)+.1*s]});
  const same=g.marks.success.gain.every((v,k)=>v===g.marks.failure.gain[k]);
  return {margin:{l:0,r:0,t:0,b:0},paper_bgcolor:'white',font:{family:'Arial,sans-serif',size:13},showlegend:false,
   scene:{camera,aspectmode:'cube',xaxis:{title:'P gain',range:ranges[0]},yaxis:{title:'I gain',range:ranges[1]},zaxis:{title:'D gain',range:ranges[2]},
    annotations:['success','failure'].map(name=>{const m=g.marks[name],q=m.gain,color=DATA.styles[name==='success'?'success_color':'recorded_color'];return{x:q[0],y:q[1],z:q[2],
     text:(name==='success'?'Success':same?'Failure (= Success)':'Failure')+'<br>('+q.join(', ')+')<br>rate : '+(100*m.survival).toFixed(1)+'%',
     showarrow:true,arrowhead:2,arrowcolor:color,arrowwidth:2,ax:name==='success'?-45:45,ay:name==='success'?-55:50,
     bgcolor:'rgba(255,255,255,.97)',bordercolor:color,borderwidth:1,borderpad:4,font:{color,size:14}}})}};
 }
 function pointInfo(g){return['success','failure'].map(name=>{const m=g.marks[name],label=name==='success'?'Success':'Failure',tag=name==='success'?'strong':'b';
  return'<'+tag+'>'+label+' ('+m.gain.join(', ')+')</'+tag+'>　rate : '+(100*m.survival).toFixed(1)+'%';}).join('<br>');}
 async function drawGroup(){
  document.body.dataset.ready='false';document.body.dataset.group=group;$('status').textContent='3Dボクセルを読み込み中…';
  for(const button of document.querySelectorAll('[data-group]'))button.className=button.dataset.group===group?'active':'';
  for(const g of selected()){await Plotly.react('plot_'+g.case,plotTraces(g,$('cutaway').checked),layout(g),config);$('points_'+g.case).innerHTML=pointInfo(g)}
  if(!listenersAttached){for(const g of selected())$('plot_'+g.case).on('plotly_relayout',async event=>{
   if(syncing||!event['scene.camera'])return;camera=event['scene.camera'];syncing=true;
   try{for(const other of selected())if(other.case!==g.case)await Plotly.relayout('plot_'+other.case,{'scene.camera':camera})}finally{syncing=false}
  });listenersAttached=true;}
  document.body.dataset.ready='true';$('status').textContent='紫の印がFailureです。ドラッグで視点を変えられます。両ケースの視点は連動します。';
 }
 async function setView(view){
  $('overview').hidden=view==='interactive';$('volume').hidden=view!=='interactive';
  for(const [id,mode] of [['show3D','3D'],['showPD','PD'],['showInteractive','interactive']])$(id).className=view===mode?'active':'';
  const figure=view==='PD'?'PD':'3D';$('primaryImage').src='figures/all_pid_success_failure_'+figure+'.png';
  for(const ext of ['pdf','svg','png'])$(ext).href='figures/all_pid_success_failure_'+figure+'.'+ext;
  document.body.dataset.view=view;
  if(view==='interactive'){await drawGroup();return}
  $('status').textContent='';document.body.dataset.ready='true';
 }
 function run(task){task.catch(error=>{document.body.dataset.error=error.message;$('status').textContent=error.message})}
 $('show3D').onclick=()=>run(setView('3D'));$('showPD').onclick=()=>run(setView('PD'));
 $('showInteractive').onclick=()=>run(setView('interactive'));
 for(const button of document.querySelectorAll('[data-group]'))button.onclick=()=>{group=button.dataset.group;run(drawGroup())};
 $('cutaway').onchange=()=>run(drawGroup());
 $('resetCamera').onclick=()=>run((async()=>{camera=initialCamera;syncing=true;try{for(const g of selected())await Plotly.relayout('plot_'+g.case,{'scene.camera':camera})}finally{syncing=false}})());
 run(setView(location.hash==='#3d'?'interactive':'3D'));
}
