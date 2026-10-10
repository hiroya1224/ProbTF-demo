const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict');
const html=fs.readFileSync(process.argv[2],'utf8');
const data=JSON.parse(html.match(/<script>const DATA=([\s\S]*?);<\/script>/)[1]);
const elements=new Map();
function element(id){
 if(!elements.has(id))elements.set(id,{value:'',checked:true,innerHTML:'',textContent:'',appendChild(){},addEventListener(){},on(){}});
 return elements.get(id);
}
for(const [id,value] of Object.entries({case:'failure1',group:'roll_pitch',mode:'all',colors:'binary',sliceI:'1',greenOpacity:'.24',redOpacity:'.16'}))element(id).value=value;
const context={DATA:data,Uint8Array,Map,Number,Math,Array,Promise,atob,setTimeout,console,
 document:{getElementById:element,createElement:()=>({}),body:{dataset:{}},title:''},
 Plotly:{react:async(id,traces)=>{element(id).data=traces},relayout:async()=>{}}};
vm.createContext(context);
vm.runInContext(fs.readFileSync(__dirname+'/../scripts/pid_voxel_viewer.js','utf8'),context);
(async()=>{
 await new Promise(resolve=>setTimeout(resolve,1));
 const small={shape:[2,2,2],axes:[[0,2],[0,1],[0,3]],edges:[[0,1,2],[0,.5,1],[0,1.5,3]],
  c:new Uint8Array(8).fill(3),b:new Uint8Array(8),u:new Uint8Array(8)};
 let faces=context.surfaceGeometry(small,127,'all',1);
 assert.equal(faces[3].faces,24);assert.equal(faces[3].i.length,48);assert.equal(faces[2].faces,0);
 faces=context.surfaceGeometry(small,127,'slice',0);assert.equal(faces[3].faces,16);
 small.c[0]=2;small.b[0]=56;
 faces=context.surfaceGeometry(small,127,'all',1);assert.equal(faces[2].faces,6);
 let grids=0;
 for(const [caseName,caseData] of Object.entries(data.cases))for(const [groupName,group] of Object.entries(caseData.groups))for(const [size,grid] of Object.entries(group.grids)){
  context.prepare(grid,caseData.baseline_count);
  const mesh=context.surfaceGeometry(grid,caseData.baseline_count,'all',group.success[1]);
  for(const code of [2,3]){
   assert.equal(mesh[code].faces,grid.faces[String(code)]);
   assert.equal(mesh[code].i.length,2*mesh[code].faces);
   assert.equal(mesh[code].x.length,4*mesh[code].faces);
   for(let axis=0;axis<3;axis++){
    const values=mesh[code][['x','y','z'][axis]],lo=grid.axes[axis][0],hi=grid.axes[axis].at(-1);
    assert(values.every(value=>value>=lo&&value<=hi));
   }
  }
  for(const [kind,ref] of Object.entries(grid.marks)){
   const index=ref.gain.map((value,axis)=>grid.axes[axis].indexOf(value));
   assert(index.every(i=>i>0));
   for(let axis=0;axis<3;axis++)assert(index[axis]<grid.axes[axis].length-1);
   const k=context.nodeIndex(...index,grid.shape);
   assert.equal(grid.c[k],ref.classification);
   assert.equal(1-grid.b[k]/caseData.baseline_count,ref.survival);
   if(kind==='success'){assert.equal(ref.survival,1);assert.equal(grid.c[k],3)}
  }
  grids++;
 }
 assert.equal(grids,16);
 assert.equal(data.cases.failure1.groups.roll_pitch.grids['65'].marks.failure.classification,2);
 assert.equal(data.cases.failure2.groups.roll_pitch.grids['65'].marks.failure.classification,2);
 const traces=element('plot65').data;
 assert(traces.some(t=>t.type==='mesh3d'&&t.opacity>0&&t.opacity<1));
 assert(traces.filter(t=>t.type==='scatter3d'&&t.mode==='markers+text').every(t=>!['#169b46','#d62728'].includes(t.marker.color)));
 console.log(JSON.stringify({passed:true,gridsChecked:grids,realNodeCountsPreserved:true,successAlwaysGreen:true,
  failureRollPitchAlwaysRed:true,voxelFacesExact:true,strictInteriorMarkers:true}));
})().catch(error=>{console.error(error);process.exitCode=1});
