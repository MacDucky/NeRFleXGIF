import os


if not os.path.exists('data/nerfstudio/'):
    os.system('ns-download-data nerfstudio --capture-name=poster')

if not os.path.exists('outputs/poster'):
    os.system('ns-train nerfacto --data data/nerfstudio/poster')

