const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict');
const html=fs.readFileSync(process.argv[2],'utf8');
const data=JSON.parse(html.match(/<script>const DATA=([\s\S]*?);<\/script>/)[1]);
const context={DATA:data,Uint8Array,atob,Math,Array};vm.createContext(context);
vm.runInContext(fs.readFileSync(__dirname+'/../scripts/pid_voxel_viewer.js','utf8'),context);
assert.equal(data.cases.length,8);assert.equal((html.match(/<img /g)||[]).length,1);
assert.equal(new Set(data.cases.map(g=>g.case+':'+g.group)).size,8);
for(const name of ['green','red']){const rgb=data.styles[name].slice(1).match(/../g).map(v=>parseInt(v,16));assert(Math.min(...rgb)>=190,'Expected a light pastel color')}
for(const g of data.cases){
 for(const cut of [false,true]){
  const faces=context.surfaceGeometry(g,cut);
  for(const code of [2,3]){
   assert.equal(faces[code].faces+faces['cut'+code].faces,g.faces[cut?'cut':'full'][String(code)]);
   for(const mesh of [faces[code],faces['cut'+code]]){
    assert.equal(mesh.i.length,mesh.faces*2);assert.equal(mesh.x.length,mesh.faces*4);
    for(let axis=0;axis<3;axis++)assert(mesh[['x','y','z'][axis]].every(v=>v>=(cut&&axis===1?g.cut_i:g.axes[axis][0])&&v<=g.axes[axis].at(-1)));
   }
   if(cut)assert(faces['cut'+code].y.every(v=>v===g.cut_i));else assert.equal(faces['cut'+code].faces,0);
  }
 }
 for(const [name,mark] of Object.entries(g.marks)){
  const indices=mark.gain.map((value,axis)=>g.axes[axis].indexOf(value));
  assert(indices.every((value,axis)=>value>0&&value<g.axes[axis].length-1));
  const k=context.nodeIndex(...indices,g.shape);
  assert.equal(g.c[k],mark.classification);assert.equal(1-g.b[k]/g.baseline_count,mark.survival);
  assert.equal(mark.classification,name==='success'||g.group!=='roll_pitch'?3:2);assert.equal(mark.gain[1],g.cut_i);
 }
 assert.equal(g.marks.success.survival,1);
 const traces=context.plotTraces(g),points=traces.filter(t=>t.type==='scatter3d');assert.equal(points.length,2);
 for(const point of points){const mark=g.marks[point.name];assert.deepEqual([point.x[0],point.y[0],point.z[0]],mark.gain)}
 const failure=points.find(t=>t.name==='failure');assert.equal(failure.marker.color,data.styles.recorded_color);
 assert.equal(failure.marker.symbol,g.group==='z'?'circle-open':'x');assert(failure.marker.size>=10);
 if(g.group==='z')assert.deepEqual(g.marks.failure.gain,g.marks.success.gain);
}
console.log(JSON.stringify({passed:true,groups:4,cases:2,panels:8,allSuccessGreen:true,onePrimaryFigure:true,
 pastelColors:true,visibleFailureMarkers:true,coincidentZReferences:true,exactNodeReferences:true,fullAndCutFacesMatchSavedCounts:true}));
