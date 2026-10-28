'use strict';
const $=id=>document.getElementById(id);
const CASE=$('case'),GROUP=$('group'),MODE=$('mode'),COLORS=$('colors'),I=$('sliceI');
const COLORS_BY_CLASS={2:'#d62728',3:'#169b46'};
const LABELS={2:'安定標本の割合 < 95%',3:'安定標本の割合 ≥ 95%'};
const CONFIG={responsive:true,displaylogo:false,toImageButtonOptions:{format:'png',scale:2}};
const meshCache=new Map();
let camera={eye:{x:1.55,y:1.5,z:1.15}},revision=0,syncing=false;
function activeCase(){return DATA.cases[CASE.value]}
function activeGroup(){return activeCase().groups[GROUP.value]}
function nodeIndex(p,i,d,shape){return (p*shape[1]+i)*shape[2]+d}
function decode(encoded){const raw=atob(encoded),bytes=new Uint8Array(raw.length);for(let i=0;i<raw.length;i++)bytes[i]=raw.charCodeAt(i);const view=new DataView(bytes.buffer),a=new Uint32Array(raw.length/4);for(let i=0;i<a.length;i++)a[i]=view.getUint32(4*i,true);return a}
function prepare(g,n){
 if(g.decoded)return g;
 g.b=decode(g.nonstable);g.u=decode(g.unresolved);g.c=new Uint8Array(g.nodes);
 if(g.b.length!==g.nodes||g.u.length!==g.nodes)throw Error('ボクセルデータの長さが不正です');
 for(let k=0;k<g.nodes;k++)g.c[k]=1-g.b[k]/n>=g.alpha?3:2;
 g.decoded=true;return g;
}
function range(axis){const span=axis.at(-1)-axis[0];return [axis[0]-.1*span,axis.at(-1)+.1*span]}
function selectedI(){return Number(I.value)}
function selectedCode(g,p,i,d,mode,requestedI){
 if(p<0||i<0||d<0||p>=g.shape[0]||i>=g.shape[1]||d>=g.shape[2])return 0;
 if(mode==='slice'&&g.axes[1][i]!==requestedI)return 0;
 if(mode==='cut'&&g.axes[1][i]>requestedI)return 0;
 const c=g.c[nodeIndex(p,i,d,g.shape)];
 if((mode==='green'&&c!==3)||(mode==='red'&&c!==2))return 0;
 return c;
}
function surfaceGeometry(g,n,mode,requestedI){
 const result={};
 for(const c of [2,3])result[c]={x:[],y:[],z:[],i:[],j:[],k:[],intensity:[],customdata:[],lx:[],ly:[],lz:[],faces:0};
 const corners=[[[0,0,0],[0,0,1],[0,1,1],[0,1,0]],[[1,0,0],[1,1,0],[1,1,1],[1,0,1]],
  [[0,0,0],[1,0,0],[1,0,1],[0,0,1]],[[0,1,0],[0,1,1],[1,1,1],[1,1,0]],
  [[0,0,0],[0,1,0],[1,1,0],[1,0,0]],[[0,0,1],[1,0,1],[1,1,1],[0,1,1]]];
 const neighbors=[[-1,0,0],[1,0,0],[0,-1,0],[0,1,0],[0,0,-1],[0,0,1]];
 for(let p=0;p<g.shape[0];p++)for(let i=0;i<g.shape[1];i++)for(let d=0;d<g.shape[2];d++){
  const c=selectedCode(g,p,i,d,mode,requestedI);if(!c)continue;
  const k=nodeIndex(p,i,d,g.shape),survival=1-g.b[k]/n,mesh=result[c];
  const details=[g.axes[0][p],g.axes[1][i],g.axes[2][d],100*survival,g.b[k],g.u[k]];
  for(let face=0;face<6;face++){
   const step=neighbors[face];if(selectedCode(g,p+step[0],i+step[1],d+step[2],mode,requestedI)===c)continue;
   const offset=mesh.x.length;
   for(const corner of corners[face]){
    mesh.x.push(g.edges[0][p+corner[0]]);mesh.y.push(g.edges[1][i+corner[1]]);mesh.z.push(g.edges[2][d+corner[2]]);
    mesh.customdata.push(details);
   }
   mesh.i.push(offset,offset);mesh.j.push(offset+1,offset+2);mesh.k.push(offset+2,offset+3);
   mesh.intensity.push(survival,survival);mesh.faces++;
   for(let edge=0;edge<4;edge++){
    const a=offset+edge,b=offset+(edge+1)%4;
    mesh.lx.push(mesh.x[a],mesh.x[b],null);mesh.ly.push(mesh.y[a],mesh.y[b],null);mesh.lz.push(mesh.z[a],mesh.z[b],null);
   }
  }
 }
 return result;
}
function cachedGeometry(g,size){
 const key=[CASE.value,GROUP.value,size,MODE.value,selectedI()].join(':');
 if(meshCache.has(key)){const v=meshCache.get(key);meshCache.delete(key);meshCache.set(key,v);return v}
 const mesh=surfaceGeometry(g,activeCase().sample_count,MODE.value,selectedI());meshCache.set(key,mesh);
 while(meshCache.size>4)meshCache.delete(meshCache.keys().next().value);
 return mesh;
}
function marks(g,twoD=false){
 if(!$('reference').checked)return [];
 const ref=g.marks.failure,q=ref.gain;if(twoD&&q[1]!==selectedI())return [];
 const trace={type:twoD?'scatter':'scatter3d',mode:'markers+text',name:CASE.value+' の全12記録ゲイン',
  x:[q[0]],y:[twoD?q[2]:q[1]],text:[CASE.value],textposition:'bottom center',
  textfont:{size:13,color:'#9c27b0'},marker:{size:twoD?13:9,symbol:'x',color:'#9c27b0',opacity:1},
  hovertemplate:`${CASE.value}<br>P=${q[0]}<br>I=${q[1]}<br>D=${q[2]}<br>安定標本=${ref.stable_count}/${activeCase().sample_count}<br>割合=${(100*ref.stable_fraction).toFixed(3)}%<br>他の9ゲインも同じ失敗記録の設定<extra></extra>`};
 if(!twoD)trace.z=[q[2]];return [trace];
}

function meshTraces(g,size){
 const geometry=cachedGeometry(g,size),out=[];
 for(const c of [2,3]){
  const m=geometry[c];if(!m.faces)continue;
  const opacity=Number($(c===3?'greenOpacity':'redOpacity').value);
  const trace={type:'mesh3d',x:m.x,y:m.y,z:m.z,i:m.i,j:m.j,k:m.k,customdata:m.customdata,
   name:LABELS[c],showlegend:true,color:COLORS_BY_CLASS[c],opacity,flatshading:true,
   lighting:{ambient:1,diffuse:0,specular:0,fresnel:0,roughness:1},
   hovertemplate:`計算点 P=%{customdata[0]}<br>I=%{customdata[1]}<br>D=%{customdata[2]}<br>安定標本の割合=%{customdata[3]:.3f}%<br>安定と確認できない標本数=%{customdata[4]}/${activeCase().sample_count}<br>うち未解決=%{customdata[5]}<extra>%{fullData.name}</extra>`};
  if(COLORS.value==='continuous'){trace.intensity=m.intensity;trace.intensitymode='cell';trace.coloraxis='coloraxis'}
  out.push(trace);
  if($('edges').checked)out.push({type:'scatter3d',mode:'lines',x:m.lx,y:m.ly,z:m.lz,showlegend:false,
   name:'ボクセル境界',line:{color:c===3?'#347655':'#96534c',width:1},opacity:.2,hoverinfo:'skip'});
 }
 return out.concat(marks(g));
}
function layout(g,twoD=false){
 const continuous=COLORS.value==='continuous';
 const base={paper_bgcolor:'white',plot_bgcolor:'white',font:{family:'system-ui,sans-serif',size:12},
  margin:{l:twoD?55:0,r:continuous?60:15,t:45,b:twoD?45:0},showlegend:!twoD,
  legend:{orientation:'h',x:0,y:1.05,font:{size:11}},uirevision:'voxel-'+CASE.value+'-'+GROUP.value,
  coloraxis:{colorscale:continuous?DATA.colorscale:[[0,COLORS_BY_CLASS[2]],[.949999999,COLORS_BY_CLASS[2]],[.95,COLORS_BY_CLASS[3]],[1,COLORS_BY_CLASS[3]]],
   cmin:0,cmax:1,autocolorscale:false,showscale:continuous,colorbar:{title:'安定標本<br>の割合',len:.7,thickness:12,tickvals:[0,.5,.95,1]}}};
 if(twoD){base.xaxis={title:'P gain',range:range(g.axes[0]),gridcolor:'#edf0f3'};base.yaxis={title:'D gain',range:range(g.axes[2]),gridcolor:'#edf0f3'}}
 else base.scene={camera,aspectmode:'cube',xaxis:{title:'P gain',range:range(g.axes[0])},yaxis:{title:'I gain',range:range(g.axes[1])},zaxis:{title:'D gain',range:range(g.axes[2])}};
 return base;
}
function sliceTraces(g){
 const n=activeCase().sample_count,i=g.axes[1].indexOf(selectedI());
 if(i<0)throw Error('断面のIが格子にありません');
 const z=[],custom=[];
 for(let d=0;d<g.shape[2];d++){
  const values=[],details=[];
  for(let p=0;p<g.shape[0];p++){
   const k=nodeIndex(p,i,d,g.shape),survival=1-g.b[k]/n;
   values.push(survival);details.push([g.axes[0][p],selectedI(),g.axes[2][d],100*survival,g.b[k]]);
  }
  z.push(values);custom.push(details);
 }
 return [{type:'heatmap',x:g.edges[0],y:g.edges[2],z,customdata:custom,coloraxis:'coloraxis',zsmooth:false,
  hovertemplate:`計算点 P=%{customdata[0]}<br>I=%{customdata[1]}<br>D=%{customdata[2]}<br>安定標本の割合=%{customdata[3]:.3f}%<br>安定と確認できない標本数=%{customdata[4]}/${n}<extra></extra>`}].concat(marks(g,true));
}
function updateText(){
 const data=activeCase(),group=activeGroup(),fine=group.grids[String(DATA.sizes.at(-1))],ref=fine.marks.failure;
 document.title=CASE.value+' PID — 全標本評価';
 $('title').textContent=CASE.value+' のPID安定割合 — 全標本評価';
 $('criterion').textContent=COLORS.value==='binary'?'安定標本の割合95%以上を緑、95%未満を赤で表示します。100%全標本安定とは別の条件です。':'全標本に占める安定標本の割合を0〜100%の連続色で表示します。実飛行の成功確率ではありません。';
 $('baseline').textContent=`${CASE.value}から得た元の${data.sample_count}標本すべてが固定分母です。成功ゲインで標本を選別せず、評価不能も分母に残します。他の9ゲインは同じ失敗記録の値です。`;
 const comparisons=Object.entries(data.full_controller_references).map(([name,row])=>`${name==='recorded_case'?'失敗記録の全12ゲイン':'成功記録の全12ゲイン（解析後の比較のみ）'}：${row.stable_count}/${row.sample_count}、評価不能${row.unresolved_count}`);
 $('referenceInfo').textContent=`× ${CASE.value}：P=${ref.gain[0]}、I=${ref.gain[1]}、D=${ref.gain[2]}（この断面の安定標本${ref.stable_count}/${data.sample_count}） / `+comparisons.join(' / ');
 $('bounds').textContent='計算した探索範囲：'+['P','I','D'].map((name,k)=>`${name}=${fine.axes[k][0]}〜${fine.axes[k].at(-1)}`).join('、');
 $('sliceTitle').textContent=`${group.label}：I=${selectedI()} のP・D断面`;
 $('greenValue').textContent=Math.round(100*Number($('greenOpacity').value))+'%';$('redValue').textContent=Math.round(100*Number($('redOpacity').value))+'%';
 $('source').textContent='今回の計算データ：'+data.source;
}

async function draw(){
 const token=++revision;document.body.dataset.ready='false';$('status').textContent='ボクセルを描画中…';
 try{
  updateText();await new Promise(resolve=>setTimeout(resolve,0));
  const group=activeGroup(),n=activeCase().sample_count;
  for(const size of DATA.sizes){
   const g=prepare(group.grids[String(size)],n);
   $('head'+size).innerHTML=`<strong>${size}点/軸の格子</strong>　${g.shape.join(' × ')} = ${g.nodes.toLocaleString()}ボクセル<div class="stats"><span class="green">95%以上 ${g.counts.green.toLocaleString()}</span><span class="red">95%未満 ${g.counts.red.toLocaleString()}</span></div>`;
   const traces=meshTraces(g,size);if(token!==revision)return;
   await Plotly.react('plot'+size,traces,layout(g),CONFIG);if(token!==revision)return;
   await Plotly.react('slice'+size,sliceTraces(g),layout(g,true),CONFIG);if(token!==revision)return;
  }
  $('status').textContent=DATA.sizes.join('点・')+'点の実計算データを表示中。各パネルの緑点を組み合わせた制御器の安定性は示しません。';
  document.body.dataset.ready='true';document.body.dataset.case=CASE.value;document.body.dataset.group=GROUP.value;
 }catch(error){$('status').textContent='表示エラー：'+error.message;document.body.dataset.error=error.message;console.error(error)}
}
function changeGroup(){
 const group=activeGroup();I.innerHTML='';
 for(const value of group.grids[String(DATA.sizes[0])].axes[1]){const option=document.createElement('option');option.value=String(value);option.textContent=Number(value.toPrecision(10)).toString();I.appendChild(option)}
 I.value=String(group.failure[1]);return draw();
}
for(const id of ['mode','colors','sliceI','greenOpacity','redOpacity','edges','reference'])$(id).addEventListener('change',draw);
CASE.addEventListener('change',changeGroup);GROUP.addEventListener('change',changeGroup);
async function setCamera(value){camera=value;await Promise.all(DATA.sizes.map(s=>Plotly.relayout('plot'+s,{'scene.camera':camera})))}
$('diagonal').onclick=()=>setCamera({eye:{x:1.55,y:1.5,z:1.15}});
$('top').onclick=()=>setCamera({eye:{x:0,y:0,z:2.6},up:{x:0,y:1,z:0}});
$('front').onclick=()=>setCamera({eye:{x:0,y:-2.6,z:0},up:{x:0,y:0,z:1}});
CASE.innerHTML='';for(const name of Object.keys(DATA.cases)){const option=document.createElement('option');option.value=name;option.textContent=name;CASE.appendChild(option)}
GROUP.innerHTML='';for(const name of DATA.groups){const option=document.createElement('option');option.value=name;option.textContent=DATA.cases[CASE.value].groups[name].label;GROUP.appendChild(option)}
if(DATA.groups.includes('roll_pitch'))GROUP.value='roll_pitch';
changeGroup();
for(const size of DATA.sizes){
 const div=$('plot'+size);
 const attach=()=>{if(!div.on)return setTimeout(attach,100);div.on('plotly_relayout',async e=>{
  if(syncing||!e['scene.camera'])return;camera=e['scene.camera'];syncing=true;
  try{await Promise.all(DATA.sizes.filter(other=>other!==size).map(other=>Plotly.relayout('plot'+other,{'scene.camera':camera})))}finally{syncing=false}
 })};attach();
}
