import time

from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd





a1 =np.linspace(0.0,2.0, 410)  #1阶




a2 = np.linspace(0,0.8, 500)     #3阶








a1, a2 = np.meshgrid(a1, a2,) #网格数组
z=((18*a2-a1)+np.sqrt((a1+14*a2)**2-64*a2))/(32*a2)
z1=[]
t=1-1.0/a1[0]

for num in t:
    if num>0:
        z1.append(num)
    else:
        z1.append(0)
# print(z)
tt=[]
z[0]=z1


for i in range(len(z)):
    for j in range(len(z[i])):
        tt.append(z[i][0])
        if z[i][j]>=0:
            pass
        else:
            z[i][j]=0
z=list(z)
z.reverse()

t1 =np.linspace(0.0,2.0, 410)  #1阶
t2 = np.linspace(0,0.8, 500)   #3阶
y1=[]


for i in range(len(t1)-1):
    for j in range(len(t2)-1):
        if z[j][i]==0:
            y1.append(t2[500-j])
            break
y1=y1[0:231]
x1=t1[0:231]

y11=[]
for i in range(len(t2) - 1, -1, -1):
    for j in range(len(t1)-1):
        if z[i][j]!=0:
            y11.append(t1[j])
            break
print('y11',y11)

y11=y11[0:52]
y12=t2[0:52]





plt.rcParams['xtick.direction'] = 'in'  # 将x周的刻度线方向设置向内
plt.rcParams['ytick.direction'] = 'in'  # 将y轴的刻度方向设置向内
#
fig = plt.figure(figsize=(5.5, 4.5))
ax=fig.subplots()
plt.rcParams['xtick.major.width'] = 1.2
plt.rcParams['ytick.major.width'] = 1.2
plt.rcParams['axes.linewidth'] = 1.2
# # im=ax.imshow(z,aspect='auto',extent=[0,3,0,0.3],cmap='magma')
im=ax.imshow(z,aspect='auto',extent=[0,2,0,0.8],cmap='RdBu',)
im.set_clim(vmin=0, vmax=1)
ax.xaxis.set_tick_params(labelsize=13)
ax.yaxis.set_tick_params(labelsize=15)
# # ax.set_xticks(fontsize=100)
# # ax.set_yticks(ticks=np.arange(3),fontsize=100)
cb=plt.colorbar(im)
cb.set_label(r'$p^{I^{ *}}$',fontsize=15)

c=0.14
d=0.19
plt.text(c, d, r'$\lambda _1=\lambda _{1_{c_{0}}}$', fontsize=17, rotation=0, c='white')

c=0.01
d=0.06
plt.text(c, d, r'$\lambda _3=\lambda _{3_{c}}$', fontsize=17, rotation=0, c='white')

c=0.69
d=0.02
plt.text(c, d, r'$\lambda _1=\lambda _{1_{c}}$', fontsize=17, rotation=0, c='white')

plt.xlabel(r'$\lambda_1$',size=15)  #为子图设置x轴标题
plt.ylabel(r'$\lambda_3$',size=15)    #为子图设置y轴标题

plt.tight_layout()
# plt.show()


ax.axvline(x=1.14, linestyle='--', color='w', )
ax.axhline(y=0.11, linestyle='--', color='w',)

ax.plot(x1,y1,linestyle='--', color='w',)

ax.plot(y11,y12,linestyle='--', color='w',)

print(y12)
plt.show()


