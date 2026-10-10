'use strict';
function nodeIndex(p,i,d,shape){return (p*shape[1]+i)*shape[2]+d}
function prepare(g){
 if(g.c)return g;
 const raw=atob(g.breaks);g.b=new Uint8Array(raw.length);g.c=new Uint8Array(raw.length);
 if(raw.length!==g.nodes)throw Error('Gain data length differs');
 for(let k=0;k<raw.length;k++){g.b[k]=raw.charCodeAt(k);g.c[k]=1-g.b[k]/g.baseline_count>=DATA.alpha?3:2}
 return g;
}
function surfaceGeometry(g){
 prepare(g);
 const out={2:{x:[],y:[],z:[],i:[],j:[],k:[],customdata:[],faces:0},3:{x:[],y:[],z:[],i:[],j:[],k:[],customdata:[],faces:0},
            cut2:{x:[],y:[],z:[],i:[],j:[],k:[],customdata:[],faces:0},cut3:{x:[],y:[],z:[],i:[],j:[],k:[],customdata:[],faces:0}};
 const corners=[[[0,0,0],[0,0,1],[0,1,1],[0,1,0]],[[1,0,0],[1,1,0],[1,1,1],[1,0,1]],
 [[0,0,0],[1,0,0],[1,0,1],[0,0,1]],[[0,1,0],[0,1,1],[1,1,1],[1,1,0]],
 [[0,0,0],[0,1,0],[1,1,0],[1,0,0]],[[0,0,1],[1,0,1],[1,1,1],[0,1,1]]];
 const neighbors=[[-1,0,0],[1,0,0],[0,-1,0],[0,1,0],[0,0,-1],[0,0,1]];
 function cls(p,i,d){return p<0||i<g.cut_index||d<0||p>=g.shape[0]||i>=g.shape[1]||d>=g.shape[2]?0:g.c[nodeIndex(p,i,d,g.shape)]}
 for(let p=0;p<g.shape[0];p++)for(let i=g.cut_index;i<g.shape[1];i++)for(let d=0;d<g.shape[2];d++){
  const k=nodeIndex(p,i,d,g.shape),c=g.c[k];
  const detail=[g.axes[0][p],g.axes[1][i],g.axes[2][d],100*(1-g.b[k]/g.baseline_count)];
  for(let f=0;f<6;f++){
   const step=neighbors[f];if(cls(p+step[0],i+step[1],d+step[2])===c)continue;
   const mesh=out[i===g.cut_index&&f===2?'cut'+c:c],offset=mesh.x.length;
   for(const corner of corners[f]){mesh.x.push(g.edges[0][p+corner[0]]);mesh.y.push(g.edges[1][i+corner[1]]);mesh.z.push(g.edges[2][d+corner[2]]);mesh.customdata.push(detail)}
   mesh.i.push(offset,offset);mesh.j.push(offset+1,offset+2);mesh.k.push(offset+2,offset+3);mesh.faces++;
  }
 }
 return out;
}
function plotTraces(g){
 const mesh=surfaceGeometry(g),traces=[];
 for(const key of [2,3,'cut2','cut3']){const code=String(key).endsWith('3')?3:2,m=mesh[key],name=code===3?'green':'red';
  traces.push({type:'mesh3d',...m,color:DATA.styles[name],opacity:String(key).startsWith('cut')?.90:DATA.styles[name+'_opacity'],flatshading:true,
   lighting:{ambient:1,diffuse:0,specular:0,fresnel:0,roughness:1},showlegend:false,
   hovertemplate:'P=%{customdata[0]}<br>I=%{customdata[1]}<br>D=%{customdata[2]}<br>生存率=%{customdata[3]:.2f}%<extra></extra>'});
 }
 for(const name of ['success','failure']){const q=g.marks[name].gain;
  traces.push({type:'scatter3d',mode:'markers',x:[q[0]],y:[q[1]],z:[q[2]],showlegend:false,
   marker:{color:'#161616',symbol:name==='success'?'diamond':'circle',size:name==='success'?8:5,opacity:1},
   hovertemplate:(name==='success'?'Success':'Failure')+'<br>P='+q[0]+'<br>I='+q[1]+'<br>D='+q[2]+'<br>生存率='+(100*g.marks[name].survival).toFixed(2)+'%<extra></extra>'});
 }
 return traces;
}
if(typeof document!=='undefined'){
 const $=id=>document.getElementById(id),initialCamera={eye:{x:1.55,y:-1.55,z:1.15}},config={responsive:true,displaylogo:false};
 let drawn=false,syncing=false,camera=initialCamera;
 function layout(g){const ranges=g.axes.map((a,k)=>{const lo=k===1?g.cut_i:a[0],s=a.at(-1)-lo;return [lo-.1*s,a.at(-1)+.1*s]});
  return {margin:{l:0,r:0,t:0,b:0},paper_bgcolor:'white',font:{family:'Arial,sans-serif',size:12},showlegend:false,
   scene:{camera,aspectmode:'cube',xaxis:{title:'P gain',range:ranges[0]},yaxis:{title:'I gain',range:ranges[1]},zaxis:{title:'D gain',range:ranges[2]},
    annotations:['success','failure'].map(name=>{const q=g.marks[name].gain;return {x:q[0],y:q[1],z:q[2],
     text:(name==='success'?'Success':'Failure')+'<br>('+q.join(', ')+')',showarrow:true,arrowhead:2,
     ax:name==='success'?-35:35,ay:name==='success'?-45:35,bgcolor:'rgba(255,255,255,.95)',borderpad:3,font:{color:'#111',size:12}}})}};
 }
 async function setView(view){
  $('pd').hidden=view!=='PD';$('volume').hidden=view!=='3D';
  $('showPD').className=view==='PD'?'active':'';$('show3D').className=view==='3D'?'active':'';
  for(const ext of ['pdf','svg','png'])$(ext).href='figures/roll_pitch_success_failure_'+view+'.'+ext;
  document.body.dataset.view=view;
  if(view==='PD'){$('status').textContent='';document.body.dataset.ready='true';return}
  if(!drawn){document.body.dataset.ready='false';$('status').textContent='3Dボクセルを読み込み中…';
   for(const g of DATA.cases){await Plotly.react('plot_'+g.case,plotTraces(g),layout(g),config)}
   for(const g of DATA.cases){$('plot_'+g.case).on('plotly_relayout',async event=>{
    if(syncing||!event['scene.camera'])return;camera=event['scene.camera'];syncing=true;
    try{for(const other of DATA.cases)if(other.case!==g.case)await Plotly.relayout('plot_'+other.case,{'scene.camera':camera})}finally{syncing=false}
   })}drawn=true;
  }else for(const g of DATA.cases)await Plotly.Plots.resize($('plot_'+g.case));
  $('status').textContent='ドラッグで視点を変えられます。両ケースの視点は連動します。';document.body.dataset.ready='true';
 }
 function show(view){setView(view).catch(error=>{document.body.dataset.error=error.message;$('status').textContent=error.message})}
 $('showPD').onclick=()=>show('PD');$('show3D').onclick=()=>show('3D');
 $('resetCamera').onclick=async()=>{camera=initialCamera;for(const g of DATA.cases)await Plotly.relayout('plot_'+g.case,{'scene.camera':camera})};
 show(location.hash==='#3d'?'3D':'PD');
}
