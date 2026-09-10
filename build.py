import json,os,sys
R=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
tpl=open(f"{R}/web/index_template.html").read()
model=open(f"{R}/web/model.json").read()
res=open(f"{R}/web/results.json").read()
eng=open(f"{R}/web/engine.js").read()
html=(tpl.replace("__MODEL_JSON__",model)
         .replace("__RESULTS_JSON__",res)
         .replace("__ENGINE_JS__",eng))
os.makedirs("/mnt/user-data/outputs",exist_ok=True)
out="/mnt/user-data/outputs/state-surgery.html"
open(out,"w").write(html)
print("built",out,round(os.path.getsize(out)/1e6,2),"MB")
