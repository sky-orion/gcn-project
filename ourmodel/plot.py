from matplotlib import pyplot as plt
import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import make_interp_spline
from matplotlib.pyplot import MultipleLocator
filename = 'fclstm.log'
step, accall, acclow,accmed,acchigh = [], [], [],[],[]
gmean,bac=[],[]
# 相比open(),with open()不用手动调用close()方法
cutdown=0
interalx=5
interaly=0.01
with open(filename, 'r') as f:
    # 将txt中的数据逐行存到列表lines里 lines的每一个元素对应于txt中的一行。然后将每个元素中的不同信息提取出来
    lines = f.readlines()
    for line in lines:
        temp = line.split(' ')
        if temp[2] != 'Horizon' or int(temp[3])< cutdown:
            continue
        accvalue=temp[4].replace('\n', '')
        accvalue = accvalue.split('|')
        step.append(int(temp[3]))
        accall.append(float(accvalue[0]))
        acchigh.append(float(accvalue[1]))
        accmed.append(float(accvalue[2]))
        acclow.append(float(accvalue[3]))
        bac.append((float(accvalue[0])+float(accvalue[1])+float(accvalue[3]))/3)
        gmean.append(pow((float(accvalue[0])*float(accvalue[1])*float(accvalue[3])), 1/3))
print(step,"\n",accall,"\n",acclow,"\n",accmed,"\n",acchigh,"\n",bac,"\n",gmean)
for i in range(len(step)):
    # print(step,"\n",accall,"\n",acclow,"\n",accmed,"\n",acchigh,"\n",bac,"\n",gmean)
    print("step",step[i],"gmean",format(gmean[i],".4f"),"bac",format(bac[i],".4f"),"accall",accall[i],"acchigh",acchigh[i] ,"accmed",accmed[i] ,"acclow",acclow[i])

fig = plt.figure(1,figsize=(12, 6))  # 创建绘图窗口，并设置窗口大小
# 画第一张图
step = np.array(step)
accall = np.array(accall)
accmed=np.array(accmed)
acclow=np.array(acclow)
acchigh=np.array(acchigh)

stepsmooth = np.linspace(step.min(), step.max(), 300)
accallsmooth = make_interp_spline(step, accall)(stepsmooth)
acchighsmooth = make_interp_spline(step, acchigh)(stepsmooth)
accmedsmooth = make_interp_spline(step, accmed)(stepsmooth)
acclowsmooth = make_interp_spline(step, acclow)(stepsmooth)
ax1 = fig.add_subplot(111)  # 将画面分割为2行1列选第一个

ax1.plot(stepsmooth, accallsmooth, 'black', label='accall')  # 画dis-loss的值，颜色红
ax1.set_xlabel('horizon')  # 设置X轴名称
ax1.set_ylabel('accury')  # 设置Y轴名称
ax1.plot(stepsmooth, acchighsmooth, 'blue', label='acchigh')
ax1.plot(stepsmooth, acclowsmooth, 'pink', label='acclow')
ax1.plot(stepsmooth, accmedsmooth, 'red', label='accmed')
plt.title("Prediction time and accuracy image(smoothed)")
plt.legend(loc='upper right')
x_major_locator=MultipleLocator(interalx)
y_major_locator=MultipleLocator(interaly)
ax=plt.gca()
ax.xaxis.set_major_locator(x_major_locator)
ax.yaxis.set_major_locator(y_major_locator)
plt.savefig(fname="smooth.png")
plt.show()  # 显示绘制的图

fig1=plt.figure(2,figsize=(12, 6))
ax1 = fig1.add_subplot(111)  # 将画面分割为2行1列选第一个
ax1.plot(step, accall, 'red', label='accall')  # 画dis-loss的值，颜色红
ax1.set_xlabel('horizon')  # 设置X轴名称
ax1.set_ylabel('accury')  # 设置Y轴名称
ax1.plot(step, acchigh, 'blue', label='acchigh')
ax1.plot(step, acclow, 'yellow', label='acclow')
ax1.plot(step, accmed, 'black', label='accmed')
plt.title("Prediction time and accuracy image(unsmoothed)")
plt.legend(loc='upper right')
x_major_locator=MultipleLocator(interalx)
y_major_locator=MultipleLocator(interaly)
ax=plt.gca()
ax.xaxis.set_major_locator(x_major_locator)
ax.yaxis.set_major_locator(y_major_locator)
plt.savefig(fname="unsmooth.png")
plt.show()  # 显示绘制的图


