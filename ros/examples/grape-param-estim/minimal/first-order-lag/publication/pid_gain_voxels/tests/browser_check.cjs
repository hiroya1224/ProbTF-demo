const {spawn}=require('node:child_process');
const fs=require('node:fs');
const chrome=spawn('/usr/bin/google-chrome',['--headless','--no-sandbox','--disable-background-networking',
 '--no-first-run','--remote-debugging-pipe','--user-data-dir=/tmp/pid-paper-portable-browser-check',
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
const snapshot=`(()=>({ready:document.body.dataset.ready,error:document.body.dataset.error,
 title:document.title,case:CASE.value,group:GROUP.value,mode:MODE.value,
 reference:document.getElementById('referenceInfo').textContent,
 plots:[33,65].map(size=>({size,traces:document.getElementById('plot'+size).data.map(t=>({type:t.type,mode:t.mode,
 name:t.name,opacity:t.opacity,faces:t.i?t.i.length/2:0,marker:t.marker?{color:t.marker.color,x:t.x,y:t.y,z:t.z}:null})),
 canvas:document.getElementById('plot'+size).querySelectorAll('canvas').length})),
 }))()`;
(async()=>{let sessionId;try{
 const {targetId}=await rpc('Target.createTarget',{url:'about:blank'});
 ({sessionId}=await rpc('Target.attachToTarget',{targetId,flatten:true}));
 await rpc('Runtime.enable',{},sessionId);
 await rpc('Page.enable',{},sessionId);
 await rpc('Emulation.setDeviceMetricsOverride',{width:1800,height:1420,deviceScaleFactor:1,mobile:false},sessionId);
 await rpc('Page.navigate',{url:process.argv[2]},sessionId);
 let initial;
 for(let i=0;i<100;i++){
  await new Promise(resolve=>setTimeout(resolve,300));
  const state=await rpc('Runtime.evaluate',{expression:'document.body?.dataset.ready',returnByValue:true},sessionId);
  if(state.result?.value==='true'){initial=(await rpc('Runtime.evaluate',{expression:snapshot,returnByValue:true},sessionId)).result.value;break}
 }
 if(!initial)throw Error('Initial voxel render did not complete: '+stderr.slice(-1800));
 const screenshot=await rpc('Page.captureScreenshot',{format:'png'},sessionId);
 fs.writeFileSync(process.argv[3]+'/browser.png',Buffer.from(screenshot.data,'base64'));
 const checks=[initial];console.log(JSON.stringify(initial));
 for(const caseName of ['failure1','failure2'])for(const groupName of ['roll_pitch','yaw','xy','z']){
  if(caseName==='failure1'&&groupName==='roll_pitch')continue;
  const result=await rpc('Runtime.evaluate',{expression:`(async()=>{CASE.value='${caseName}';GROUP.value='${groupName}';MODE.value='all';await changeGroup();return ${snapshot}})()`,awaitPromise:true,returnByValue:true},sessionId);
  if(result.exceptionDetails)throw Error(JSON.stringify(result.exceptionDetails));
  const state=result.result.value;if(state.ready!=='true'||state.error)throw Error('Render failed '+JSON.stringify(state));
  checks.push(state);console.log(JSON.stringify({case:caseName,group:groupName,ready:state.ready,meshes:state.plots.map(p=>p.traces.filter(t=>t.type==='mesh3d').length)}));
 }
 for(const mode of ['green','red','slice','cut']){
  const result=await rpc('Runtime.evaluate',{expression:`(async()=>{CASE.value='failure1';GROUP.value='roll_pitch';MODE.value='${mode}';await changeGroup();return ${snapshot}})()`,awaitPromise:true,returnByValue:true},sessionId);
  if(result.exceptionDetails||result.result.value.error)throw Error('Mode failed '+mode);
  checks.push(result.result.value);console.log(JSON.stringify({mode,ready:result.result.value.ready}));
 }
 await rpc('Runtime.evaluate',{expression:`(async()=>{MODE.value='all';COLORS.value='continuous';await draw();return document.body.dataset.ready})()`,awaitPromise:true,returnByValue:true},sessionId);
 const reset=await rpc('Runtime.evaluate',{expression:`(async()=>{COLORS.value='binary';await draw();return ${snapshot}})()`,awaitPromise:true,returnByValue:true},sessionId);
 if(errors.length)throw Error('Browser exception: '+JSON.stringify(errors));
 fs.writeFileSync(process.argv[3]+'/browser_validation.json',JSON.stringify({passed:true,checks,final:reset.result.value,errors},null,2));
 await rpc('Browser.close');
 }catch(error){console.error(error);console.error(stderr.slice(-2200));chrome.kill();process.exitCode=1}
})();
setTimeout(()=>{chrome.kill();process.exit(2)},180000).unref();
