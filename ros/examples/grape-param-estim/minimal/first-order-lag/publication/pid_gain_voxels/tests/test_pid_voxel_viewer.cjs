const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict');
const html=fs.readFileSync(process.argv[2],'utf8');
const data=JSON.parse(html.match(/<script>const DATA=([\s\S]*?);<\/script>/)[1]);
const context={DATA:data,Uint8Array,atob,Math,Array};vm.createContext(context);
vm.runInContext(fs.readFileSync(__dirname+'/../scripts/pid_voxel_viewer.js','utf8'),context);
assert.equal(data.cases.length,8);assert.equal((html.match(/<img /g)||[]).length,1);
assert.equal(new Set(data.cases.map(g=>g.case+':'+g.group)).size,8);
assert(!('alpha' in data));assert(!('alpha' in data.styles));assert.deepEqual(data.rate_range,[0,1]);
assert.equal(data.colorscale.length,256);assert.equal(data.colorscale[0][0],0);assert.equal(data.colorscale.at(-1)[0],1);
for(const [position,color] of data.colorscale){
 assert(position>=0&&position<=1);
 const rgb=color.slice(1).match(/../g).map(v=>parseInt(v,16));assert(Math.min(...rgb)>=190,'Expected a light pastel color');
}
assert(data.colorscale[128][1]!==data.colorscale[0][1]&&data.colorscale[128][1]!==data.colorscale.at(-1)[1]);
for(const g of data.cases){
 for(const cut of [false,true]){
  const faces=context.surfaceGeometry(g,cut);
  for(const name of ['surface','cut']){
   const mesh=faces[name];assert.equal(mesh.faces,g.faces[cut?'cut':'full'][name]);
   assert.equal(mesh.i.length,mesh.faces*2);assert.equal(mesh.x.length,mesh.faces*4);assert.equal(mesh.intensity.length,mesh.faces*2);
   for(let axis=0;axis<3;axis++)assert(mesh[['x','y','z'][axis]].every(v=>v>=(cut&&axis===1?g.cut_i:g.axes[axis][0])&&v<=g.axes[axis].at(-1)));
   for(let f=0;f<mesh.faces;f++){
    const rate=mesh.intensity[2*f];assert(rate>=0&&rate<=1);assert.equal(mesh.intensity[2*f+1],rate);
    assert(Math.abs(mesh.customdata[4*f][3]-100*rate)<1e-12);
   }
  }
  if(cut)assert(faces.cut.y.every(v=>v===g.cut_i));else assert.equal(faces.cut.faces,0);
 }
 for(const [name,mark] of Object.entries(g.marks)){
  const indices=mark.gain.map((value,axis)=>g.axes[axis].indexOf(value));
  assert(indices.every((value,axis)=>value>0&&value<g.axes[axis].length-1));
  const k=context.nodeIndex(...indices,g.shape);
  assert.equal(1-g.b[k]/g.baseline_count,mark.survival);assert(!('classification' in mark));assert.equal(mark.gain[1],g.cut_i);
 }
 assert.equal(g.marks.success.survival,1);
 const traces=context.plotTraces(g),points=traces.filter(t=>t.type==='scatter3d');assert.equal(points.length,2);
 for(const mesh of traces.filter(t=>t.type==='mesh3d')){
  assert(!('color' in mesh));assert.equal(mesh.intensitymode,'cell');assert.equal(mesh.cmin,0);assert.equal(mesh.cmax,1);
  assert.equal(mesh.opacity,data.styles.voxel_opacity);assert.deepEqual(mesh.colorscale,data.colorscale);
 }
 for(const point of points){const mark=g.marks[point.name];assert.deepEqual([point.x[0],point.y[0],point.z[0]],mark.gain)}
 const failure=points.find(t=>t.name==='failure');assert.equal(failure.marker.color,data.styles.recorded_color);
 assert.equal(failure.marker.symbol,g.group==='z'?'circle-open':'x');assert(failure.marker.size>=10);
 if(g.group==='z')assert.deepEqual(g.marks.failure.gain,g.marks.success.gain);
}
// Both rates would previously have been merged into the same >=95% class.
const sample={axes:[[0,2],[0,1],[0,3]],edges:[[0,1,2],[0,.5,1],[0,1.5,3]],shape:[2,2,2],nodes:8,
 baseline_count:100,cut_index:0,cut_i:0,breaks:Buffer.from([4,1,1,1,1,1,1,1]).toString('base64')};
const faces=context.surfaceGeometry(sample,false);
assert.equal(faces.surface.faces,30);assert(faces.surface.intensity.includes(.96));assert(faces.surface.intensity.includes(.99));
console.log(JSON.stringify({passed:true,groups:4,cases:2,panels:8,allSuccessGreen:true,onePrimaryFigure:true,
 continuousRateColors:true,noThreshold:true,pastelColors:true,perCellIntensitiesExact:true,visibleFailureMarkers:true,
 coincidentZReferences:true,exactNodeReferences:true,fullAndCutFacesMatchSavedCounts:true}));
