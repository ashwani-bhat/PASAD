
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'qt')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import hankel
from pyts.decomposition import ssa


# In[4]:


def Pasad_train(oneSensor,n,L):
    trajectory=hankel(oneSensor[:L],oneSensor[L:n])
    trajectory_t = trajectory.T
    U,S,Vt = np.linalg.svd(trajectory_t)
    print(U.shape)
    r=1
    #r = int(input("Enter r\n"))
    U = U[:,:r]
    return (U.T,trajectory_t)

def trajectoryMatrix(oneSensor,n,L):
    trajectory=hankel(oneSensor[:L],oneSensor[L:n])
    trajectory_t = trajectory.T
    return trajectory_t


data = pd.read_csv(r"C:\Users\Anuraag  Kansara\Desktop\Critical Infrastructure\pasad-master\pasad-master\data\4 - Scenario SA2\xmv6_twentyeight_data_1.csv")
print(data.shape)
train = data.iloc[:500,:].values
validation =  data.iloc[251:4000,:].values
test = data.iloc[3751:,:].values

oneSensor_train = train[:,9:10]
L=250
k=250
# Departure score
depScore=[]

ut,trajectory_t = Pasad_train(oneSensor_train,500,L)
print('Shape of U.t',ut.shape)
oneSensor_val = validation[:,9:10]
trajectory_val = trajectoryMatrix(oneSensor_val,3500,L)

sum1=np.zeros((k,))
print('Shape of sum',sum1.shape)
for i in range(0,k):
    sum1+=trajectory_t[:,i]
meanVal = sum1/k
print("Shape of meanval",meanVal.shape)
c = ut.dot(meanVal)
print('Shape of C',c.shape,'Shape of trajectory validation',trajectory_val.shape)
traj_val_t= trajectory_val.T
# validation phase
iterate = traj_val_t.shape[1]
i=0
valList=[]
while i< iterate:
    y= c - ut.dot(traj_val_t[:,i])
    #print('Trajectory of test',trajectory_test.shape)
    yyt = np.linalg.norm(y)
    theta_val = yyt
    depScore.append(theta_val)
    valList.append(theta_val)
    #print("theta_test::",theta_test)
    i=i+1
maxVal = max(valList)
print("maxval::",maxVal)
y= c - ut.dot(trajectory_val.T)
yyt = np.linalg.norm(y)
print("Shape of yyt",yyt.shape,"Shape of y",y.shape)
theta_val = yyt
print('Shape of theta_val',theta_val.shape)
print('theta_val',theta_val)

# testing phase
oneSensor_test = test[:,9:10]
trajectory_test = trajectoryMatrix(oneSensor_test,800,L)

trajt=trajectory_test.T
#print("trajt[:,0]::",trajt[:,0].shape)
iterate = trajt.shape[1]
i=0

while i< iterate:
    y= c - ut.dot(trajt[:,i])
    #print('Trajectory of test',trajectory_test.shape)
    yyt = np.linalg.norm(y)
    #yyt = y*y
    theta_test = yyt
    depScore.append(theta_test)
    #print("theta_test::",theta_test)
#     if theta_test>= maxVal:
#         print("Generate alarm",i)
    i=i+1


# plotting 
# plt.plot([i/100 for i in range(1000,4800)],depScore,'r-',label = 'Departure  Score')
# plt.hlines(maxVal,0,48,colors='k', linestyles='dashed',label='alarm threshold')
# # plt.yticks(np.arange(0,0.20, step=0.05))
# plt.xlabel("Time(hours)")
# plt.ylabel("Departure Score")
# plt.legend()
# plt.axis()
# plt.show()

plt.plot([i/100 for i in range(500)],train[:,9],'b',label='Training data')
plt.plot([i/100 for i in range(251,4000)],validation[:,9],'g',label ='Validation data')
plt.plot([i/100 for i in range(3751,4800)],test[:,9],'r',label='Testing data')
# plt.yticks(np.arange(0.26,0.28, step=0.01))
plt.legend()
plt.xlabel('Time(hours)')
plt.axis()
plt.show()


