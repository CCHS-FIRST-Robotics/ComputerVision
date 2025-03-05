import yaml

with open("config.yaml", "r") as file:
    cfg = yaml.safe_load(file)
print(cfg)
for k, v in cfg.items():
    print(k, v)
