import numpy as np

# from pathlib import Path
# import sys
# import os

# # Инициализация путей и параметров
# path_init = f'/storage/thalassa/users/vkoshkina'
# folder = 'scripts/CyTRACK/src/cytrack'
# sys.path.insert(2, f'{path_init}/{folder}')


import cytrack 


args = cytrack.read_args()
if args.cytrack_help:
	cytrack.help()
elif args.get_template:
	cytrack.get_cytrack_inputs_template()
else:
	cytrack.get_cytrack_main(args.parameterfile)
