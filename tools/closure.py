import sys, json, os, importlib
sys.path.insert(0, '/home/claude/mapper'); import syncpool; syncpool.install()
os.environ['SKIP_INTEGRATION']='0'
f=sys.argv[1]; mod=f[:-3].replace('/','.')
before=set(sys.modules)
importlib.import_module(mod)
print(json.dumps([m for m in sys.modules if m not in before and (m.startswith('sqlglot') or m.startswith('tests'))]))
