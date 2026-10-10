'use strict';
const $=id=>document.getElementById(id);
const CASE=$('case'),GROUP=$('group'),MODE=$('mode'),COLORS=$('colors'),I=$('sliceI');
const COLORS_BY_CLASS={2:DATA.styles.red,3:DATA.styles.green};
const LABELS={2:'条件付き生存率 < 95%',3:'条件付き生存率 ≥ 95%'};
const CONFIG={responsive:true,displaylogo:false,toImageButtonOptions:{format:'png',scale:2}};
const meshCache=new Map();
let camera={eye:{x:1.55,y:1.5,z:1.15}},revision=0,syncing=false;
function activeCase(){return DATA.cases[CASE.value]}
function activeGroup(){return activeCase().groups[GROUP.value]}
function nodeIndex(p,i,d,shape){return (p*shape[1]+i)*shape[2]+d}
function decode(encoded){const raw=atob(encoded),a=new Uint8Array(raw.length);for(let i=0;i<raw.length;i++)a[i]=raw.charCodeAt(i);return a}
function prepare(g,n){
 if(g.decoded)return g;
 g.b=decode(g.breaks);g.u=g.unresolved?decode(g.unresolved):new Uint8Array(g.nodes);g.c=new Uint8Array(g.nodes);
 if(g.b.length!==g.nodes||g.u.length!==g.nodes)throw Error('ボクセルデータの長さが不正です');
 for(let k=0;k<g.nodes;k++)g.c[k]=1-g.b[k]/n>=g.alpha?3:2;
 g.decoded=true;return g;
}
function range(axis){const span=axis.at(-1)-axis[0],pad=DATA.styles.frame_padding*span;return [axis[0]-pad,axis.at(-1)+pad]}
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
 const mesh=surfaceGeometry(g,activeCase().baseline_count,MODE.value,selectedI());meshCache.set(key,mesh);
 while(meshCache.size>4)meshCache.delete(meshCache.keys().next().value);
 return mesh;
}
function marks(g,twoD=false){
 if(!$('reference').checked)return [];
 const out=[];
 for(const name of ['success','failure']){
  const ref=g.marks[name],q=ref.gain;if(twoD&&q[1]!==selectedI())continue;
  const success=name==='success',label=success?'Success':CASE.value;
  const trace={type:twoD?'scatter':'scatter3d',mode:'markers+text',name:success?'成功した飛行のゲイン':CASE.value+' の記録ゲイン',
   x:[q[0]],y:[twoD?q[2]:q[1]],text:[label],textposition:success?'top center':'bottom center',
   textfont:{size:13,color:success?'#2070c0':'#9c27b0'},marker:{size:twoD?13:9,symbol:success?'diamond':'x',color:success?'#2070c0':'#9c27b0',opacity:1},
   hovertemplate:`${label}<br>P=${q[0]}<br>I=${q[1]}<br>D=${q[2]}<br>条件付き生存率=${(100*ref.survival).toFixed(3)}%<br>基準集合内の不安定=${ref.breaks}/${activeCase().baseline_count}<br>他の9ゲインは成功飛行の設定<extra></extra>`};
  if(!twoD)trace.z=[q[2]];out.push(trace);
 }
 return out;
}
function meshTraces(g,size){
 const geometry=cachedGeometry(g,size),out=[];
 for(const c of [2,3]){
  const m=geometry[c];if(!m.faces)continue;
  const opacity=Number($(c===3?'greenOpacity':'redOpacity').value);
  const trace={type:'mesh3d',x:m.x,y:m.y,z:m.z,i:m.i,j:m.j,k:m.k,customdata:m.customdata,
   name:LABELS[c],showlegend:true,color:COLORS_BY_CLASS[c],opacity,flatshading:true,
   lighting:{ambient:1,diffuse:0,specular:0,fresnel:0,roughness:1},
   hovertemplate:`計算点 P=%{customdata[0]}<br>I=%{customdata[1]}<br>D=%{customdata[2]}<br>条件付き生存率=%{customdata[3]:.3f}%<br>基準集合内の不安定=%{customdata[4]}/${activeCase().baseline_count}<br>うち未解決=%{customdata[5]}<extra>%{fullData.name}</extra>`};
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
  coloraxis:{colorscale:continuous?DATA.colorscale:[[0,COLORS_BY_CLASS[2]],[DATA.alpha-1e-9,COLORS_BY_CLASS[2]],[DATA.alpha,COLORS_BY_CLASS[3]],[1,COLORS_BY_CLASS[3]]],
   cmin:0,cmax:1,autocolorscale:false,showscale:continuous,colorbar:{title:'条件付き<br>生存率',len:.7,thickness:12,tickvals:[0,.5,DATA.alpha,1]}}};
 if(twoD){base.xaxis={title:'P gain',range:range(g.axes[0]),gridcolor:'#edf0f3'};base.yaxis={title:'D gain',range:range(g.axes[2]),gridcolor:'#edf0f3'}}
 else base.scene={camera,aspectmode:'cube',xaxis:{title:'P gain',range:range(g.axes[0])},yaxis:{title:'I gain',range:range(g.axes[1])},zaxis:{title:'D gain',range:range(g.axes[2])}};
 return base;
}
function sliceTraces(g){
 const n=activeCase().baseline_count,i=g.axes[1].indexOf(selectedI());
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
  hovertemplate:`計算点 P=%{customdata[0]}<br>I=%{customdata[1]}<br>D=%{customdata[2]}<br>条件付き生存率=%{customdata[3]:.3f}%<br>基準集合内の不安定=%{customdata[4]}/${n}<extra></extra>`}].concat(marks(g,true));
}
function updateText(){
 const data=activeCase(),group=activeGroup(),fine=group.grids['65'],ref=fine.marks.failure;
 document.title=CASE.value+' PID — 透明ボクセル';
 $('title').textContent=CASE.value+' のPID分布 — 半透明ボクセル';
 $('criterion').innerHTML=COLORS.value==='binary'?'<span class="green">緑：条件付き生存率95%以上</span>、<span class="red">赤：95%未満</span>。各計算点の値を、対応するボクセルに表示します。':'過去と同じ条件付き生存率を、0%：赤 → 100%：緑の連続色で表示します。95%境界と基準集合は同じです。';
 $('baseline').textContent=`${CASE.value}のモデル分布で、成功ゲインで安定だった${data.baseline_count}/${data.samples}サンプルが共通の基準集合です。成功ゲイン自身の条件付き生存率は100%。`;
 $('referenceInfo').innerHTML=`<span class="blue">◆ Success</span>：P=${group.success[0]}、I=${group.success[1]}、D=${group.success[2]}　<span class="purple">× ${CASE.value}</span>：P=${ref.gain[0]}、I=${ref.gain[1]}、D=${ref.gain[2]}（この図での生存率 ${(100*ref.survival).toFixed(2)}%）`;
 $('bounds').textContent='計算した探索範囲：'+['P','I','D'].map((name,k)=>`${name}=${fine.axes[k][0]}〜${fine.axes[k].at(-1)}`).join('、');
 $('sliceTitle').textContent=`${group.label}：I=${selectedI()} のP・D断面`;
 $('greenValue').textContent=Math.round(100*Number($('greenOpacity').value))+'%';$('redValue').textContent=Math.round(100*Number($('redOpacity').value))+'%';
 $('source').textContent='保存済みの計算データ：'+data.source;
}
async function draw(){
 const token=++revision;document.body.dataset.ready='false';$('status').textContent='ボクセルを描画中…';
 try{
  updateText();await new Promise(resolve=>setTimeout(resolve,0));
  const group=activeGroup(),n=activeCase().baseline_count;
  for(const size of DATA.sizes){
   const g=prepare(group.grids[String(size)],n);
   $('head'+size).innerHTML=`<strong>${size}点/軸の格子</strong>　${g.shape.join(' × ')} = ${g.nodes.toLocaleString()}ボクセル<div class="stats"><span class="green">95%以上 ${g.counts.green.toLocaleString()}</span><span class="red">95%未満 ${g.counts.red.toLocaleString()}</span></div>`;
   const traces=meshTraces(g,size);if(token!==revision)return;
   await Plotly.react('plot'+size,traces,layout(g),CONFIG);if(token!==revision)return;
   await Plotly.react('slice'+size,sliceTraces(g),layout(g,true),CONFIG);if(token!==revision)return;
  }
  $('status').textContent='33点・65点とも保存済みデータを表示中。ドラッグで回転し、透明度やI断面を調整できます。';
  document.body.dataset.ready='true';document.body.dataset.case=CASE.value;document.body.dataset.group=GROUP.value;
 }catch(error){$('status').textContent='表示エラー：'+error.message;document.body.dataset.error=error.message;console.error(error)}
}
function changeGroup(){
 const group=activeGroup();I.innerHTML='';
 for(const value of group.grids['33'].axes[1]){const option=document.createElement('option');option.value=String(value);option.textContent=Number(value.toPrecision(10)).toString();I.appendChild(option)}
 I.value=String(group.success[1]);return draw();
}
for(const id of ['mode','colors','sliceI','greenOpacity','redOpacity','edges','reference'])$(id).addEventListener('change',draw);
CASE.addEventListener('change',changeGroup);GROUP.addEventListener('change',changeGroup);
async function setCamera(value){camera=value;await Promise.all(DATA.sizes.map(s=>Plotly.relayout('plot'+s,{'scene.camera':camera})))}
$('diagonal').onclick=()=>setCamera({eye:{x:1.55,y:1.5,z:1.15}});
$('top').onclick=()=>setCamera({eye:{x:0,y:0,z:2.6},up:{x:0,y:1,z:0}});
$('front').onclick=()=>setCamera({eye:{x:0,y:-2.6,z:0},up:{x:0,y:0,z:1}});
changeGroup();
for(const size of DATA.sizes){
 const div=$('plot'+size);
 const attach=()=>{if(!div.on)return setTimeout(attach,100);div.on('plotly_relayout',async e=>{
  if(syncing||!e['scene.camera'])return;camera=e['scene.camera'];syncing=true;
  try{await Plotly.relayout('plot'+(size===33?65:33),{'scene.camera':camera})}finally{syncing=false}
 })};attach();
}
