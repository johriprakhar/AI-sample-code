import re

cfg_src = open("config.py", encoding="utf-8").read()
cfg = set(re.findall(r'_env_\w+\(\s*"(\w+)"', cfg_src))
cfg |= set(re.findall(r'os\.getenv\("(\w+)"', cfg_src))

ex = set(re.findall(r"^([A-Z_]+)=", open(".env.example", encoding="utf-8").read(), re.M))

readme = open("README.md", encoding="utf-8").read()
rd = set(re.findall(r"[`|]\s*`?([A-Z][A-Z_]{3,})`?\s*[`|]", readme))

print("env vars in config.py     :", len(cfg))
for v in sorted(cfg):
    print("   ", v)
print()
print("missing from .env.example :", sorted(cfg - ex) or "none")
print("extra in .env.example     :", sorted(ex - cfg) or "none")
print("missing from README       :", sorted(cfg - rd) or "none")
