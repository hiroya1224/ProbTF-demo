const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict');
const html=fs.readFileSync(process.argv[2],'utf8');
const data=JSON.parse(html.match(/<script>const DATA=([\s\S]*?);<\/script>/)[1]);
const context={DATA:data,Uint8Array,atob,Math,Array};vm.createContext(context);
vm.runInContext(fs.readFileSync(__dirname+'/../scripts/pid_voxel_viewer.js','utf8'),context);
assert.equal(data.cases.length,2);assert.equal((html.match(/<img /g)||[]).length,1);
for(const g of data.cases){
 const faces=context.surfaceGeometry(g);
 for(const code of [2,3]){
  assert.equal(faces[code].faces+faces['cut'+code].faces,g.faces[String(code)]);
  for(const mesh of [faces[code],faces['cut'+code]]){
   assert.equal(mesh.i.length,mesh.faces*2);assert.equal(mesh.x.length,mesh.faces*4);
   for(let axis=0;axis<3;axis++)assert(mesh[['x','y','z'][axis]].every(v=>v>=(axis===1?g.cut_i:g.axes[axis][0])&&v<=g.axes[axis].at(-1)));
  }
  assert(faces['cut'+code].y.every(v=>v===1));
 }
 for(const [name,mark] of Object.entries(g.marks)){
  const indices=mark.gain.map((value,axis)=>g.axes[axis].indexOf(value));
  assert(indices.every((value,axis)=>value>0&&value<g.axes[axis].length-1));
  const k=context.nodeIndex(...indices,g.shape);
  assert.equal(g.c[k],mark.classification);assert.equal(1-g.b[k]/g.baseline_count,mark.survival);
  assert.equal(mark.classification,name==='success'?3:2);assert.equal(mark.gain[1],1);
 }
 const traces=context.plotTraces(g);
 assert.equal(traces.filter(t=>t.type==='scatter3d').length,2);
 assert.equal(traces.filter(t=>t.type==='mesh3d'&&t.opacity===.90).length,2);
}
console.log(JSON.stringify({passed:true,cases:2,onePrimaryFigure:true,successGreen:true,failureRed:true,
 exactNodeReferences:true,exactSharedIntegralPlane:true,cutSurfaceMatchesSavedCounts:true}));
