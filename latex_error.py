import matplotlib
matplotlib.use('TkAgg')
matplotlib.use('PS')
matplotlib.rcParams['text.usetex']=True
matplotlib.rcParams['text.latex.unicode']=True
import pylab as plt
plt.switch_backend('PS')

string = r'z=${value}^{upper}_{lower}$'.format(
                value='{' + str(0.27) + '}',
                upper='{+' + str(0.01) + '}',
                lower='{-' + str(0.01) + '}')
print(string)

fig = plt.figure(figsize=(3,1))
fig.text(0.1,0.5,string,size=24,va='center')
from matplotlib.backends.backend_pdf import PdfPages
pp = PdfPages('images/issue5076.pdf')
pp.savefig(fig)
pp.close()