const {spawn}=require('node:child_process');
const fs=require('node:fs'),assert=require('node:assert/strict'),crypto=require('node:crypto');
const chrome=spawn('/usr/bin/google-chrome',['--headless','--no-sandbox','--disable-background-networking',
 '--no-first-run','--remote-debugging-pipe','--user-data-dir=/tmp/pid-all-axes-browser-check',
 '--use-angle=swiftshader','--enable-unsafe-swiftshader','--enable-webgl','--disable-dev-shm-usage'],
 {stdio:['ignore','ignore','pipe','pipe','pipe']});
let next=1,buffer='',pending=new Map(),errors=[];
chrome.stdio[4].on('data',chunk=>{buffer+=chunk.toString();let at;while((at=buffer.indexOf('\0'))>=0){
 const x=JSON.parse(buffer.slice(0,at));buffer=buffer.slice(at+1);
 if(x.method==='Runtime.exceptionThrown')errors.push(x.params.exceptionDetails);
 if(x.id&&pending.has(x.id)){pending.get(x.id)(x);pending.delete(x.id)}
}});
let stderr='';chrome.stderr.on('data',chunk=>stderr+=chunk.toString());
chrome.stdio[4].on('error',error=>{console.error('Chrome pipe:',error.message,stderr.slice(-2500));process.exitCode=1});
chrome.on('exit',code=>{if(code)console.error('Chrome exit:',code,stderr.slice(-2500))});
function rpc(method,params={},sessionId){return new Promise((resolve,reject)=>{
 const id=next++;pending.set(id,x=>x.error?reject(x.error):resolve(x.result));
 chrome.stdio[3].write(JSON.stringify({id,method,params,sessionId})+'\0');
})}
const snapshot=`(typeof DATA==='undefined'||!document.querySelector('h1')?{ready:'false'}:(()=>({ready:document.body.dataset.ready,error:document.body.dataset.error,
 view:document.body.dataset.view,group:document.body.dataset.group,title:document.title,heading:document.querySelector('h1').textContent,
 summary:document.querySelector('.summary').textContent,primaryImages:document.querySelectorAll('img').length,
 imageLoaded:document.querySelector('img').complete&&document.querySelector('img').naturalWidth>0,
 imageSrc:document.querySelector('img').getAttribute('src'),overviewVisible:!document.getElementById('overview').hidden,
 volumeVisible:!document.getElementById('volume').hidden,cutaway:document.getElementById('cutaway').checked,
 allReferences:DATA.cases.map(g=>({case:g.case,group:g.group,success:g.marks.success,failure:g.marks.failure})),
 downloads:['pdf','svg','png'].map(id=>document.getElementById(id).getAttribute('href')),
 plots:DATA.cases.filter(g=>g.group===document.body.dataset.group).map(g=>{const p=document.getElementById('plot_'+g.case);return{case:g.case,group:g.group,
  references:g.marks,canvas:p.querySelectorAll('canvas').length,
  traces:(p.data||[]).map(t=>({type:t.type,name:t.name,mode:t.mode,opacity:t.opacity,faces:t.i?t.i.length/2:0,
   marker:t.marker?{color:t.marker.color,symbol:t.marker.symbol,size:t.marker.size,x:t.x,y:t.y,z:t.z}:null})),
  annotations:p.layout?.scene?.annotations,pointText:document.getElementById('points_'+g.case).textContent}})
}))())`;
(async()=>{try{
 const{targetId}=await rpc('Target.createTarget',{url:'about:blank'});
 const{sessionId}=await rpc('Target.attachToTarget',{targetId,flatten:true});
 await rpc('Runtime.enable',{},sessionId);await rpc('Page.enable',{},sessionId);
 await rpc('Emulation.setDeviceMetricsOverride',{width:1800,height:1300,deviceScaleFactor:1,mobile:false},sessionId);
 async function evaluate(expression){const result=await rpc('Runtime.evaluate',{expression,returnByValue:true,awaitPromise:true},sessionId);
  if(result.exceptionDetails)throw Error(JSON.stringify(result.exceptionDetails));return result.result.value;}
 async function waitFor(view,group){for(let attempt=0;attempt<180;attempt++){
  await new Promise(resolve=>setTimeout(resolve,250));const state=await evaluate(snapshot);if(state.error)throw Error(state.error);
  if(state.ready==='true'&&state.view===view&&state.imageLoaded&&(!group||state.group===group))return state;
 }throw Error('View did not become ready: '+view+' / '+group)}
 async function capture(file){const result=await rpc('Page.captureScreenshot',{format:'png'},sessionId);
  fs.writeFileSync(process.argv[3]+'/'+file,Buffer.from(result.data,'base64'));}
 await rpc('Page.navigate',{url:process.argv[2]},sessionId);
 const primary=await waitFor('3D');assert.equal(primary.primaryImages,1);assert(primary.overviewVisible&&!primary.volumeVisible);
 assert(primary.imageSrc.includes('all_pid_success_failure_3D'));assert(primary.summary.includes('8/8'));
 assert.equal(primary.allReferences.length,8);assert(primary.allReferences.every(g=>g.success.classification===3&&g.success.survival===1));
 await capture('primary_browser.png');
 await evaluate("document.getElementById('showPD').click()");const slice=await waitFor('PD');
 assert(slice.imageSrc.includes('_PD.'));assert(slice.downloads.every(p=>p.includes('_PD.')));
 await evaluate("document.getElementById('showInteractive').click()");
 const checks=[];
 for(const group of ['roll_pitch','yaw','xy','z']){
  if(group!=='roll_pitch')await evaluate(`document.querySelector('[data-group="${group}"]').click()`);
  const state=await waitFor('interactive',group);assert(state.volumeVisible&&!state.overviewVisible);assert.equal(state.plots.length,2);
  for(const plot of state.plots){assert(plot.canvas>0);assert.equal(plot.traces.filter(t=>t.type==='mesh3d').length,4);
   const points=plot.traces.filter(t=>t.type==='scatter3d');assert.equal(points.length,2);assert.equal(plot.annotations.length,2);
   for(const mesh of plot.traces.filter(t=>t.type==='mesh3d'))assert(mesh.opacity>0&&mesh.opacity<=.32);
   for(const annotation of plot.annotations)assert(/rate : \d+\.\d%/.test(annotation.text));
   assert.equal((plot.pointText.match(/rate : \d+\.\d%/g)||[]).length,2);
   assert.equal(plot.references.success.classification,3);assert.equal(plot.references.failure.classification,group==='roll_pitch'?2:3);
   for(const point of points)assert.deepEqual([point.marker.x[0],point.marker.y[0],point.marker.z[0]],plot.references[point.name].gain);
   const failure=points.find(t=>t.name==='failure');assert.equal(failure.marker.symbol,group==='z'?'circle-open':'x');assert(failure.marker.size>=10);
   assert(plot.pointText.includes('Success')&&plot.pointText.includes('Failure'));
  }
  checks.push(state);
  if(group==='roll_pitch')await capture('failure_points_browser.png');
  if(group==='z')await capture('coincident_z_browser.png');
 }
 const changedCamera={eye:{x:1.2,y:-1.8,z:.9}};
 await evaluate(`Plotly.relayout('plot_failure1',{'scene.camera':${JSON.stringify(changedCamera)}}).then(()=>true)`);
 await new Promise(resolve=>setTimeout(resolve,500));
 assert.deepEqual(await evaluate("document.getElementById('plot_failure2').layout.scene.camera.eye"),changedCamera.eye);
 await evaluate("document.getElementById('cutaway').click()");const full=await waitFor('interactive','z');assert(!full.cutaway);
 await evaluate("document.getElementById('show3D').click()");const returned=await waitFor('3D');assert(returned.overviewVisible&&!returned.volumeVisible);
 assert.equal(errors.length,0);
 const source=fs.readFileSync(new URL(process.argv[2]));
 fs.writeFileSync(process.argv[3]+'/browser_validation.json',JSON.stringify({passed:true,html_sha256:crypto.createHash('sha256').update(source).digest('hex'),
  primary,slice,interactiveChecks:checks,fullVolume:full,returnToOverview:true,cameraSynchronized:true,errors},null,2)+'\n');
 console.log(JSON.stringify({passed:true,groups:4,cases:2,allSuccessGreen:true,primaryFigurePanels:8,largeFailureMarkers:true,coincidentZ:true,fullVolume:true,cameraSynchronized:true,errors:0}));
 await rpc('Browser.close');
}catch(error){console.error(error);console.error(stderr.slice(-2000));chrome.kill();process.exitCode=1}})();
setTimeout(()=>{chrome.kill();process.exit(2)},180000).unref();
