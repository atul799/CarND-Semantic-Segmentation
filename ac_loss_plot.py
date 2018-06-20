# -*- coding: utf-8 -*-
"""
plot acc loss

@author: atpandey
"""

#%%
import matplotlib.pyplot as plt


#%%
ff='./to_laptop/trg_file.txt'

with open(ff,'r') as trgf:
    listidx=[]
    listloss=[]
    listacc=[]
    ctr=0
    for line in trgf:
        if(ctr>0):
            ll=line.split(',')
            listidx.append(ll[0])
            listloss.append(ll[1])
            listacc.append(ll[2])
        #listf.append(line)
        
        ctr +=1

#for i in range(len(listidx)):
#    print("idx: {}, loss: {}, acc: {}".format(listidx[i],listloss[i],listacc[i]))
    
# Make a figure
fig = plt.figure()

plt.subplots_adjust(top = 0.99, bottom=0.05, hspace=0.5, wspace=0.4)
# The axes 
ax1 = fig.add_subplot(2, 1, 1)
ax2 = fig.add_subplot(2, 1, 2)



#plots
ax1.plot(listloss,'bo-',label='loss')
ax2.plot(listacc,'go-',label='accuracy')

ax1.set_xlabel('training idx')
ax1.set_ylabel('Loss')
ax1.set_title('loss data set')
ax1.legend()
ax2.set_xlabel('training idx')
ax2.set_ylabel('accuracy')
ax2.set_title('accuracydata set')
ax2.legend()


plt.show()
plt.savefig('./outputs/loss_accuracy.png')