const {spawn}=require('node:child_process');
const fs=require('node:fs'),assert=require('node:assert/strict'),crypto=require('node:crypto');
const chrome=spawn('/usr/bin/google-chrome',['--headless','--no-sandbox','--disable-background-networking',
 '--no-first-run','--remote-debugging-pipe','--user-data-dir=/tmp/pid-focus-browser-check',
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
 view:document.body.dataset.view,title:document.title,heading:document.querySelector('h1').textContent,
 primaryImages:document.querySelectorAll('img').length,
 imageLoaded:document.querySelector('img').complete&&document.querySelector('img').naturalWidth>0,
 sliceVisible:!document.getElementById('pd').hidden,volumeVisible:!document.getElementById('volume').hidden,
 downloads:['pdf','svg','png'].map(id=>document.getElementById(id).getAttribute('href')),
 plots:DATA.cases.map(g=>{const p=document.getElementById('plot_'+g.case);return {case:g.case,
  references:g.marks,canvas:p.querySelectorAll('canvas').length,
  traces:(p.data||[]).map(t=>({type:t.type,mode:t.mode,opacity:t.opacity,
   faces:t.i?t.i.length/2:0,marker:t.marker?{color:t.marker.color,x:t.x,y:t.y,z:t.z}:null})),
  annotations:p.layout?.scene?.annotations}})
}))())`;
(async()=>{try{
 const {targetId}=await rpc('Target.createTarget',{url:'about:blank'});
 const {sessionId}=await rpc('Target.attachToTarget',{targetId,flatten:true});
 await rpc('Runtime.enable',{},sessionId);await rpc('Page.enable',{},sessionId);
 await rpc('Emulation.setDeviceMetricsOverride',{width:1440,height:1050,deviceScaleFactor:1,mobile:false},sessionId);
 async function evaluate(expression){
  const result=await rpc('Runtime.evaluate',{expression,returnByValue:true,awaitPromise:true},sessionId);
  if(result.exceptionDetails)throw Error(JSON.stringify(result.exceptionDetails));return result.result.value;
 }
 async function waitFor(view){
  for(let attempt=0;attempt<160;attempt++){
   await new Promise(resolve=>setTimeout(resolve,250));
   const state=await evaluate(snapshot);if(state.error)throw Error(state.error);
   if(state.ready==='true'&&state.view===view&&state.imageLoaded)return state;
  }throw Error('View did not become ready: '+view);
 }
 async function capture(file){const result=await rpc('Page.captureScreenshot',{format:'png'},sessionId);
  fs.writeFileSync(process.argv[3]+'/'+file,Buffer.from(result.data,'base64'));
 }
 await rpc('Page.navigate',{url:process.argv[2]},sessionId);
 const slice=await waitFor('PD');assert.equal(slice.primaryImages,1);assert(slice.sliceVisible&&!slice.volumeVisible);
 assert(slice.downloads.every(path=>path.includes('_PD.')));
 await capture('primary_browser.png');
 await evaluate("document.getElementById('show3D').click()");
 const volume=await waitFor('3D');assert(volume.volumeVisible&&!volume.sliceVisible);
 assert(volume.downloads.every(path=>path.includes('_3D.')));
 for(const plot of volume.plots){assert(plot.canvas>0);assert.equal(plot.traces.filter(t=>t.type==='mesh3d').length,4);
  assert.equal(plot.traces.filter(t=>t.type==='mesh3d'&&t.opacity===.90).length,2);
  assert.equal(plot.traces.filter(t=>t.type==='scatter3d').length,2);assert.equal(plot.annotations.length,2);
  assert.equal(plot.references.success.classification,3);assert.equal(plot.references.failure.classification,2);
  const points=plot.traces.filter(t=>t.type==='scatter3d');
  for(const [index,name] of ['success','failure'].entries())assert.deepEqual([points[index].marker.x[0],points[index].marker.y[0],points[index].marker.z[0]],plot.references[name].gain);
 }
 await capture('volume_browser.png');
 const changedCamera={eye:{x:1.2,y:-1.8,z:.9}};
 await evaluate(`Plotly.relayout('plot_failure1',{'scene.camera':${JSON.stringify(changedCamera)}}).then(()=>true)`);
 await new Promise(resolve=>setTimeout(resolve,500));
 const synchronized=await evaluate("document.getElementById('plot_failure2').layout.scene.camera.eye");
 assert.deepEqual(synchronized,changedCamera.eye);
 await evaluate("document.getElementById('showPD').click()");const returned=await waitFor('PD');
 assert(returned.sliceVisible&&!returned.volumeVisible);assert.equal(errors.length,0);
 const source=fs.readFileSync(new URL(process.argv[2]));
 const report={passed:true,html_sha256:crypto.createHash('sha256').update(source).digest('hex'),
  slice,volume,returnToSlice:true,cameraSynchronized:true,errors};
 fs.writeFileSync(process.argv[3]+'/browser_validation.json',JSON.stringify(report,null,2)+'\n');
 console.log(JSON.stringify({passed:true,onePrimaryFigure:true,views:['PD','3D'],successGreen:true,failureRed:true,cameraSynchronized:true,errors:0}));
 await rpc('Browser.close');
 }catch(error){console.error(error);console.error(stderr.slice(-2000));chrome.kill();process.exitCode=1}
})();
setTimeout(()=>{chrome.kill();process.exit(2)},180000).unref();
