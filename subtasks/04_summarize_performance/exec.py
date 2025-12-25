import os, sys
sys.path.append(os.getcwd())

log_path = 'logs'

for filename in os.listdir(log_path):
    if not filename.endswith('.log'):
        continue

    if filename.split(' ')[0] != 'Test':
        continue

    with open(os.path.join(log_path, filename), 'r') as f:
        lines = f.readlines()

    print(filename[5:-4])
    print(lines[-1][-5:])