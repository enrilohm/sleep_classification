from train.load_mesa import load_mesa, MesaDataLoader
mesa_dict = load_mesa()

test=[x["label"] for x in mesa_dict.values()]

import numpy as np
(np.concatenate(test)==0).mean()
np.mean(text)
