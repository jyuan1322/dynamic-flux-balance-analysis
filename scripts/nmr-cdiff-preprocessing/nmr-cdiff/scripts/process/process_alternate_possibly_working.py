import nmrglue as ng
import numpy as np
import matplotlib.pyplot as plt

dic, data = ng.bruker.read(path_to_fid)
data = ng.bruker.remove_digital_filter(dic, data)
data = ng.proc_base.fft(data)
spec = np.real(data)

plt.plot(spec)
plt.show()
