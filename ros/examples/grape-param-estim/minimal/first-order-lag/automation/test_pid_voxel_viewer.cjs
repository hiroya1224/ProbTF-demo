const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict');
const html=fs.readFileSync(process.argv[2],'utf8');
const dataMatch=html.match(/<script>const DATA=([\s\S]*?);<\/script>/);
assert(dataMatch,'embedded DATA must exist');
const data=JSON.parse(dataMatch[1]);
const viewerCode=[...html.matchAll(/<script>([\s\S]*?)<\/script>/g)]
 .map(match=>match[1]).find(code=>code.includes('function surfaceGeometry'));
assert(viewerCode,'embedded viewer implementation must exist');
assert(data.sizes.length>0);
const elements=new Map();
function element(id){
 if(!elements.has(id))elements.set(id,{value:'',checked:true,innerHTML:'',textContent:'',appendChild(){},addEventListener(){},on(){}});
 return elements.get(id);
}
for(const [id,value] of Object.entries({case:Object.keys(data.cases)[0],group:data.groups[0],mode:'all',colors:'binary',sliceI:'1',greenOpacity:'.24',redOpacity:'.16'}))element(id).value=value;
const context={DATA:data,Uint8Array,Uint32Array,DataView,ArrayBuffer,Map,Number,Math,Array,Promise,atob,setTimeout,console,
 document:{getElementById:element,createElement:()=>({}),body:{dataset:{}},title:''},
 Plotly:{react:async(id,traces)=>{element(id).data=traces},relayout:async()=>{}}};
vm.createContext(context);
vm.runInContext(viewerCode,context);
(async()=>{
 await context.draw();
 const small={shape:[2,2,2],axes:[[0,2],[0,1],[0,3]],edges:[[0,1,2],[0,.5,1],[0,1.5,3]],
  c:new Uint8Array(8).fill(3),b:new Uint32Array(8),u:new Uint32Array(8)};
 let faces=context.surfaceGeometry(small,128,'all',1);
 assert.equal(faces[3].faces,24);assert.equal(faces[3].i.length,48);assert.equal(faces[2].faces,0);
 faces=context.surfaceGeometry(small,128,'slice',0);assert.equal(faces[3].faces,16);
 small.c[0]=2;small.b[0]=56;
 faces=context.surfaceGeometry(small,128,'all',1);assert.equal(faces[2].faces,6);
 let grids=0;
 let expectedGrids=0,drawnGroups=0;
 for(const [caseName,caseData] of Object.entries(data.cases)){
 assert(Number.isInteger(caseData.sample_count)&&caseData.sample_count>0);
 for(const [groupName,group] of Object.entries(caseData.groups)){
  expectedGrids+=data.sizes.length;
  assert.deepEqual(Object.keys(group.grids).sort(),data.sizes.map(String).sort());
  for(const [size,grid] of Object.entries(group.grids)){
  context.prepare(grid,caseData.sample_count);
  const mesh=context.surfaceGeometry(grid,caseData.sample_count,'all',group.failure[1]);
  for(const code of [2,3]){
   assert.equal(mesh[code].faces,grid.faces[String(code)]);
   assert.equal(mesh[code].i.length,2*mesh[code].faces);
   assert.equal(mesh[code].x.length,4*mesh[code].faces);
   for(let axis=0;axis<3;axis++){
    const values=mesh[code][['x','y','z'][axis]],lo=grid.axes[axis][0],hi=grid.axes[axis].at(-1);
    assert(values.every(value=>value>=lo&&value<=hi));
   }
  }
  assert.deepEqual(Object.keys(grid.marks),['failure']);
  for(const ref of Object.values(grid.marks)){
   const index=ref.gain.map((value,axis)=>grid.axes[axis].indexOf(value));
   assert(index.every(i=>i>0));
   for(let axis=0;axis<3;axis++)assert(index[axis]<grid.axes[axis].length-1);
   const k=context.nodeIndex(...index,grid.shape);
   assert.equal(grid.c[k],ref.classification);
   assert.equal(caseData.sample_count-grid.b[k],ref.stable_count);
   assert(Math.abs(ref.stable_count/caseData.sample_count-ref.stable_fraction)<=2*Number.EPSILON);
   assert.equal(grid.u[k],ref.unresolved);
   assert.equal(ref.classification,ref.stable_fraction>=grid.alpha?3:2);
  }
  grids++;
  }
  element('case').value=caseName;element('group').value=groupName;
  await context.changeGroup();
  assert.equal(context.document.body.dataset.ready,'true');
  for(const size of data.sizes){
   const traces=element('plot'+size).data;
   assert(traces.some(t=>t.type==='mesh3d'&&t.opacity>0&&t.opacity<1));
   const markers=traces.filter(t=>t.type==='scatter3d'&&t.mode==='markers+text');
   assert.equal(markers.length,1);
   assert.equal(markers[0].text[0],caseName);
   assert(!['#169b46','#d62728'].includes(markers[0].marker.color));
  }
  drawnGroups++;
 }
 }
 assert.equal(grids,expectedGrids);
 console.log(JSON.stringify({passed:true,gridsChecked:grids,groupsDrawn:drawnGroups,sizes:data.sizes,
  realNodeCountsPreserved:true,allOriginalSamplesInDenominator:true,
  onlyRecordedFailureMarkers:true,voxelFacesExact:true,strictInteriorMarkers:true}));
})().catch(error=>{console.error(error);process.exitCode=1});
