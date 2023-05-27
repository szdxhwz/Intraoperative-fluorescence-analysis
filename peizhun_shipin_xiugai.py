import numpy as np
import cv2
import os
from peizhun1 import peizhun
import matplotlib.pyplot as plt
import scipy.signal as signal
from scipy.optimize import curve_fit
import random
import math
import sys
import pickle
import sklearn
from datetime import datetime 
from threading import Thread
from concurrent.futures import ThreadPoolExecutor
import warnings
warnings.filterwarnings("ignore")


def process_video(openpath,path,start_frame, length):
    cap = cv2.VideoCapture(openpath)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    freq = 10
    for idx in range(length):
        ret = cap.grab()
        if not ret:
            break
        
        if (idx+start_frame) % freq == 0:
            ret, frame = cap.retrieve()
            if frame is None:    # exist broken frame
                break
            new_img = cv2.resize(frame,None,fx=0.3,fy=0.3,interpolation = cv2.INTER_LINEAR)
            cv2.imwrite(path+'/'+str((idx+start_frame)//freq) + ".jpg", new_img)  # 保存图片
    cap.release()

def duiqi(dirs1,path1,peizhun_savepath,quzhen_savepath,start,end):
    for i in range(start,end):
        path3=os.path.join(peizhun_savepath,dirs1[i])
        path2=os.path.join(quzhen_savepath,dirs1[i])
        peizhun(path1,path2,path3)


def shipin(openpath,dir,chizichangdu):
  kaishi=datetime.now()
  print('---开始---',str(kaishi))
  quzhen_savepath=dir+'/'+'quzhen'
  folder=os.path.exists(quzhen_savepath)
  if not folder:
      os.makedirs(quzhen_savepath)
#   cap = cv2.VideoCapture(openpath)
#   frameNum = 0
#   quzhen_savepath=dir+'/'+'quzhen'
#   folder=os.path.exists(quzhen_savepath)
#   if not folder:
#       os.makedirs(quzhen_savepath)
#   while (cap.isOpened()):
#       ret, frame = cap.read()
#       frameNum = frameNum + 1
#       if frameNum % 10 == 0:  # 调整帧数
#           if ret:
#               new_img = cv2.resize(frame,None,fx=0.3,fy=0.3,interpolation = cv2.INTER_LINEAR)
#               cv2.imwrite(quzhen_savepath+'/'+str(frameNum//10) + ".jpg", new_img)  # 保存图片
#           else:
#               break
#   cap.release
  
  cap = cv2.VideoCapture(openpath)  
  frames_num = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
  epoch=frames_num//3
  t1 = Thread(target=process_video, args=(openpath, quzhen_savepath,0,epoch))
  t2 = Thread(target=process_video, args=(openpath, quzhen_savepath,epoch,epoch))
  t3 = Thread(target=process_video, args=(openpath, quzhen_savepath,epoch*2,frames_num-epoch*2))

# 启动线程运行
  t1.start()
  t2.start()
  t3.start()

# 等待所有线程执行完毕
  t1.join()  # join() 等待线程终止，要不然一直挂起
  t2.join()
  t3.join()
#   cap = cv2.VideoCapture(openpath)
#   idx = 0
#   freq = 10
#   while True:
#     idx += 1
#     ret = cap.grab()
#     if not ret:
#          break
        
#     if idx % freq == 1:
#          ret, frame = cap.retrieve()
#          if frame is None:    # exist broken frame
#              break
         
#          new_img = cv2.resize(frame,None,fx=0.3,fy=0.3,interpolation = cv2.INTER_LINEAR)
#          cv2.imwrite(quzhen_savepath+'/'+str(idx//freq) + ".jpg", new_img)  # 保存图片
#   cap.release()
  quzhenjieshu=datetime.now()
  print('---取帧结束---',str(quzhenjieshu),'---耗时---',(quzhenjieshu-kaishi).seconds,'s')
  dirs1=os.listdir(quzhen_savepath+'/')
  dirs1.sort(key=lambda x: int(x.split('.')[0]))
  path1=quzhen_savepath+'/0.jpg'
  peizhun_savepath=dir+'/'+'peizhun'
  folder=os.path.exists(peizhun_savepath)
  if not folder:
      os.makedirs(peizhun_savepath)
  changdu=len(dirs1)
  cishu=changdu//3
  t4 = Thread(target=duiqi, args=(dirs1,path1,peizhun_savepath,quzhen_savepath,0,cishu))
  t5 = Thread(target=duiqi, args=(dirs1,path1,peizhun_savepath,quzhen_savepath,cishu,cishu*2))
  t6 = Thread(target=duiqi, args=(dirs1,path1,peizhun_savepath,quzhen_savepath,cishu*2,changdu))

# 启动线程运行
  t4.start()
  t5.start()
  t6.start()

# 等待所有线程执行完毕
  t4.join()  # join() 等待线程终止，要不然一直挂起
  t5.join()
  t6.join()
#   for i in range(len(dirs1)):
#     path3=os.path.join(peizhun_savepath,dirs1[i])
#     path2=os.path.join(quzhen_savepath,dirs1[i])
#     peizhun(path1,path2,path3)
  peizhunjieshu=datetime.now()
  print('---配准结束---',str(peizhunjieshu),'---耗时---',(peizhunjieshu-quzhenjieshu).seconds,'s')
  font = {'family' : 'MicroSoft YaHei',
        'weight' : 'bold',
        'size'   : 30}

  def func1(x, x0,b,c,d,e):
      return np.piecewise(x, [x < x0, x >= x0], [lambda x:0, 
                                    lambda x:np.exp(-b*x0)*c-np.exp(-b*x)*c+np.exp(-d*x)*e])
  def ransac1(x,y,rate,Iter_Number,DisT,ProT,qsd):
      N=len(x)
      best_inliers_number =0
      i=0
      p0=qsd,0.1,1,0.1,1
      while True:
          x_2c= random.sample(list(x), int(N*rate))
          y_2c=y[x_2c]
          try:
              popt, pcov = curve_fit(func1, x_2c, y_2c,p0,maxfev=800000)
          except Exception:
              continue
          finally:
              X_inliers = []
              Y_inliers = []
              inliers_current = 0 #当前模型的内点数量
              i+=1
              for j in range(N):
                  x_current = x[j]
                  y_current = y[j]
                  dis_current = np.abs(y_current-func1(x_current,*popt))
                  if (dis_current <= DisT):
                      inliers_current += 1
                      X_inliers.append(x_current)
                      Y_inliers.append(y_current)
                  #print("当前内点数量={}, 最佳内点数量={},最佳内点比例={}".format(inliers_current, best_inliers_number,best_inliers_number / N))
              if (inliers_current > best_inliers_number):
                  i=0
                  Pro_current = inliers_current / N       #当前模型的内点比例Pro_current
                  best_inliers_number = inliers_current   #更新最优内点的数量
                  best_x0,best_b,best_c,best_d,best_e = popt  #更新模型参数
              if ((best_inliers_number / N) > ProT):
                  break
              if i>0.9*Iter_Number:
                  x0_qz=math.ceil(best_x0)
                  y1=y[:x0_qz+1]
                  y2=y[x0_qz+1:]
                  y=np.append(y1,y2[y2>0])
                  N=len(y)
                  x=np.arange(N)
              if i> Iter_Number:
                  break
      return best_x0,best_b,best_c,best_d,best_e
  
  def jisuan(fenduanindex,dirs_len):
      fenduanindexlen=len(fenduanindex)
      s1=np.zeros(shape=(fenduanindexlen,dirs_len))
      s2=np.zeros(shape=(fenduanindexlen,dirs_len))
      s3=np.zeros(shape=(fenduanindexlen,12))
      s4=np.zeros(shape=fenduanindexlen)
      for num,wz in enumerate(fenduanindex):
                y_point=[]
                hl=int(wz[1]-2)
                hh=int(wz[1]+3)
                wl=int(wz[0]-2)
                wh=int(wz[0]+3)
                point=np.zeros(shape=(dirs_len))
                p=0
                for i in range(dirs_len):
                    for j in range(hl,hh):
                        for h in range(wl,wh):
                            if list1[j,h,0,i]>=35 and list1[j,h,0,i]<=77:
                                p+=list1[j,h,1,i]
                    p1=p/25
                    point[i]=p1
                    p=0
                
                y_med = signal.medfilt(point, kernel_size=13)
                y_med=np.array(y_med)
                y_med=y_med.flatten()
                s1[num,:]=y_med
                #y_med=y_med[y_med>0]
                if y_med[y_med>0].size == 0:
                    s4[num]=0
                else:
                    s4[num]=1
                    #y_med=np.insert(y_med,0,0)
                    tidu=np.asarray([y_med[i]-y_med[i-1] for i in range(1,len(y_med))])
                    tidu_index=np.where(tidu>0)[0][0]+1
                    x=np.arange(len(y_med))
                    x0,b,c,d_x,e=ransac1(x,y_med,0.8,400,3,0.8,tidu_index)
                    
                    for i in x:
                        y_point.append(func1(i,x0,b,c,d_x,e))
                    s2[num,:]=y_point
                    index=np.where(y_point>=(max(y_point)*0.99))[0][0]
                    y_point_new=y_point[tidu_index-1:index+1]
                    y_12=int(y_point[index])/2
                    for i in range(len(y_point_new)):
                        if y_point_new[i] >= y_12:
                            t12max=round((i/y_point_new[i]*y_12)/3,2)
                            break
                    Fmax=round(int(y_point[index]),1)
                    Tmax=round(int(x[index]-x0)/3,2)
                    Slope=round(Fmax/Tmax,2)
                    TR=round(t12max/Tmax,2)
                    with open('./model.pickle', 'rb') as f:
                        model = pickle.load(f)
                    score=round(model.predict([[Tmax,Slope,t12max,TR]])[0],1)
                    if score>100:
                        score=100
                    elif score<0:
                        score=0
                    s3[num,:]=[score,Tmax,Fmax,Slope,t12max,TR,x0,b,c,d_x,e,index]
      return s1,s2,s3,s4

  def OnMouseAction(event, x, y, flags, param):
      global frame_1, position_line,position_line1,position_line2,position_line3,position_line4,position_line5,d,d1,d2
      if event == cv2.EVENT_RBUTTONDOWN:
          position_line = (x,y)
          cv2.circle(frame_1, position_line, 2, (0,0,0), -1)
      # elif event == cv2.EVENT_MOUSEMOVE and (flags & cv2.EVENT_FLAG_RBUTTON):
      #     cv2.line(frame, position_line, (x,y), (0,0,0), 2)
      elif event == cv2.EVENT_RBUTTONUP:
          position_line1 = (x,y)
          d=math.sqrt((position_line[0]-position_line1[0])**2+(position_line[1]-position_line1[1])**2)
          cv2.line(frame_1, position_line, position_line1, (0,0,0), 2)
          cv2.putText(frame_1,f"{chizi}cm" , (position_line[0]+2,position_line[1]+2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
      elif event == cv2.EVENT_LBUTTONDOWN:
          position_line2 = (x,y)
          cv2.circle(frame_1, position_line2, 2, (255,0,0), -1)
      elif event == cv2.EVENT_LBUTTONUP:
          position_line3 = (x,y)
          cv2.line(frame_1, position_line2, position_line3, (255,0,0), 1)
          d1=math.sqrt((position_line3[0]-position_line2[0])**2+(position_line3[1]-position_line2[1])**2)
          zhixian_x=np.arange(position_line2[0],position_line3[0],(position_line3[0]-position_line2[0])/d1*(d/int(chizi)))
          zhixian_x=[int(zhixian_x[i]) for i in range(len(zhixian_x))]
          zhixian_y=np.arange(position_line2[1],position_line3[1],(position_line3[1]-position_line2[1])/d1*(d/int(chizi)))
          zhixian_y=[int(zhixian_y[i]) for i in range(len(zhixian_y))]
          zhixian=np.hstack((np.array(zhixian_x).reshape(-1,1),np.array(zhixian_y).reshape(-1,1)))
          print(zhixian)
          kaishijisuan=datetime.now()
          print('---开始计算---',str(kaishijisuan))
          for j,i in enumerate(zhixian):
          #cv2.rectangle(frame, position_line2, (position_line3[0],i), (255,0,0), 1)
          # d1=math.sqrt((position_line2[0]-position_line3[0])**2+(position_line2[1]-position_line3[1])**2)
          # cv2.line(frame, position_line2, position_line3, (255,0,0), 2)
              cv2.circle(frame_1, center=i, radius=2,color=(255, 0, 0), thickness=-1)
              cv2.putText(frame_1,f"{j}cm" , (i[0]+4,i[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
          #cv2.putText(frame,f"{round((int(chizi)/d)*d1,1)}mm" , (position_line3[0]+4,position_line3[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)
          xianduan_len=len(zhixian)
          if xianduan_len<4:
            plt.rc("font", **font)
            plt.figure(figsize=(40,20*xianduan_len), dpi=80)
            for num,wz in enumerate(zhixian):
                y_point=[]
                hl=int(wz[1]-2)
                hh=int(wz[1]+3)
                wl=int(wz[0]-2)
                wh=int(wz[0]+3)
                point=np.zeros(shape=(dirs_len))
                p=0
                for i in range(dirs_len):
                    for j in range(hl,hh):
                        for h in range(wl,wh):
                            if list1[j,h,0,i]>=35 and list1[j,h,0,i]<=77:
                                p+=list1[j,h,1,i]
                    p1=p/25
                    point[i]=p1
                    p=0
                
                y_med = signal.medfilt(point, kernel_size=13)
                y_med=np.array(y_med)
                y_med=y_med.flatten()
                #y_med=y_med[y_med>0]
                if y_med[y_med>0].size == 0:
                    x=np.arange(dirs_len)
                    cv2.putText(frame_1,'0', (wz[0]+40,wz[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
                    plt.subplot(xianduan_len,1,xianduan_len-num)
                    plt.scatter(x,y_med,s=60)
                    plt.plot(x, y_med, '-p', color='grey',
                        marker = 'o',
                        markersize=3, linewidth=5,
                        markerfacecolor='red',
                        markeredgecolor='red',
                        markeredgewidth=2)
                    plt.xlabel('帧数', fontsize=35)
                    plt.ylabel('荧光强度', fontsize=35)
                    plt.title(f'距离起点{(num)}cm的ICG曲线', fontsize=50)
                else:
                    #y_med=np.insert(y_med,0,0)
                    tidu=np.asarray([y_med[i]-y_med[i-1] for i in range(1,len(y_med))])
                    tidu_index=np.where(tidu>0)[0][0]+1
                    x=np.arange(len(y_med))
                    x0,b,c,d_x,e=ransac1(x,y_med,0.8,400,3,0.8,tidu_index)
                    
                    #y_med1 = preprocess1.fit_transform(y_med)
                    #y_med2 = preprocess1.fit_transform(y_med)
                    #tmax=(np.log(0.01*(np.exp(-b*x0)+t/c))/-b)
                    #thalfmax=(np.log(0.5*(np.exp(-b*x0)+t/c))/-b)
                    #print(position_line3[0]+2,wz)
                    for i in x:
                        y_point.append(func1(i,x0,b,c,d_x,e))
                    index=np.where(y_point>=(max(y_point)*0.99))[0][0]
                    y_point_new=y_point[tidu_index-1:index+1]
                    y_12=int(y_point[index])/2
                    
                    for i in range(len(y_point_new)):
                        if y_point_new[i] >= y_12:
                            t12max=round((i/y_point_new[i]*y_12)/3,2)
                            break
                    
                    Fmax=round(int(y_point[index]),1)
                    Tmax=round(int(x[index]-x0)/3,2)
                    Slope=round(Fmax/Tmax,2)
                    TR=round(t12max/Tmax,2)
                    with open('./model.pickle', 'rb') as f:
                        model = pickle.load(f)
                    score=round(model.predict([[Tmax,Slope,t12max,TR]])[0],1)
                    if score>100:
                        score=100
                    elif score<0:
                        score=0
                    #cv2.putText(frame,f'Tmax:{round((tmax-x0)/3,1)}s,Fmax:{round(0.95*(np.exp(-b*x0)*c+t),1)}' , (position_line3[0]+30,wz), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)
                    cv2.putText(frame_1,f'{score}', (wz[0]+40,wz[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
                    
                    plt.subplot(xianduan_len,1,xianduan_len-num)
                    plt.scatter(x,y_med,s=60)
                    plt.plot(x, y_point, '-p', color='red',
                        marker = 'o',
                        markersize=3, linewidth=5,
                        markerfacecolor='red',
                        markeredgecolor='red',
                        markeredgewidth=2)
                    plt.xlabel('帧数', fontsize=35)
                    plt.ylabel('荧光强度', fontsize=35)
                    plt.title(f'距离起点{(num)}cm的ICG曲线', fontsize=50)
                    plt.text(x=0.5*len(x),#文本x轴坐标 
                        y=0.5*Fmax, #文本y轴坐标
                        s=f'Score:{score},Tmax:{Tmax}s,Fmax:{Fmax},Slope:{Slope}\nT1/2max:{t12max}s,TR:{TR}\nx0:{round(x0,1)},b:{round(b,5)},c:{round(c,2)}\nd:{round(d,5)},e:{round(e,2)}', #文本内容
                        fontdict=dict(fontsize=35, color='r',family='MicroSoft YaHei',),#字体属性字典
                        #添加文字背景色
                        bbox={'facecolor': '#74C476', #填充色
                            'edgecolor':'b',#外框色
                            'alpha': 0.5, #框透明度
                            'pad': 8,#本文与框周围距离 
                            }
                        )
                    plt.axhline(y=y_point[index])
                    plt.axvline(x=x0-1)
                    plt.axvline(x=x[index])
            plt.savefig(dir+"/duodian_fenxi.jpg")
            plt.show()
          else:
            plt.rc("font", **font)
            plt.figure(figsize=(40,20*xianduan_len), dpi=80)
            fenduan=xianduan_len//4
            pool = ThreadPoolExecutor(max_workers=4)
            future1 = pool.submit(jisuan, zhixian[:fenduan,:],dirs_len)
            future2 = pool.submit(jisuan, zhixian[fenduan:fenduan*2,:],dirs_len)
            future3 = pool.submit(jisuan, zhixian[fenduan*2:fenduan*3,:],dirs_len)
            future4 = pool.submit(jisuan, zhixian[fenduan*3:,:],dirs_len)
            s5=np.concatenate((future1.result()[0],future2.result()[0],future3.result()[0],future4.result()[0]),axis=0)
            s6=np.concatenate((future1.result()[1],future2.result()[1],future3.result()[1],future4.result()[1]),axis=0)
            s7=np.concatenate((future1.result()[2],future2.result()[2],future3.result()[2],future4.result()[2]),axis=0)
            s8=np.concatenate((future1.result()[3],future2.result()[3],future3.result()[3],future4.result()[3]),axis=0)
            print("future1线程的状态:" + str(future1.done())) 
            x_scatter=np.arange(dirs_len)
            for num,wz in enumerate(zhixian):
                y_yuan=s5[num,:]
                y_yuce=s6[num,:]
                if s8[num]==0:
                    cv2.putText(frame_1,'0', (wz[0]+40,wz[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
                    plt.subplot(xianduan_len,1,xianduan_len-num)
                    plt.scatter(x_scatter,y_yuan,s=60)
                    plt.plot(x_scatter,y_yuce, '-p', color='grey',
                        marker = 'o',
                        markersize=3, linewidth=5,
                        markerfacecolor='red',
                        markeredgecolor='red',
                        markeredgewidth=2)
                    plt.xlabel('帧数', fontsize=35)
                    plt.ylabel('荧光强度', fontsize=35)
                    plt.title(f'距离起点{(num)}cm的ICG曲线', fontsize=50)
                    plt.text(x=0.5*len(x_scatter),#文本x轴坐标 
                        y=1, #文本y轴坐标
                        s=f'Score:{0}', #文本内容
                        fontdict=dict(fontsize=35, color='r',family='MicroSoft YaHei',),#字体属性字典
                        #添加文字背景色
                        bbox={'facecolor': '#74C476', #填充色
                            'edgecolor':'b',#外框色
                            'alpha': 0.5, #框透明度
                            'pad': 8,#本文与框周围距离 
                            }
                        )
                else:
                    cv2.putText(frame_1,f'{s7[num,0]}', (wz[0]+40,wz[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
                    
                    plt.subplot(xianduan_len,1,xianduan_len-num)
                    plt.scatter(x_scatter,y_yuan,s=60)
                    plt.plot(x_scatter,y_yuce, '-p', color='red',
                        marker = 'o',
                        markersize=3, linewidth=5,
                        markerfacecolor='red',
                        markeredgecolor='red',
                        markeredgewidth=2)
                    plt.xlabel('帧数', fontsize=35)
                    plt.ylabel('荧光强度', fontsize=35)
                    plt.title(f'距离起点{(num)}cm的ICG曲线', fontsize=50)
                    plt.text(x=0.5*len(x_scatter),#文本x轴坐标 
                        y=0.5*s7[num,2], #文本y轴坐标
                        s=f'Score:{s7[num,0]},Tmax:{s7[num,1]}s,Fmax:{s7[num,2]},Slope:{s7[num,3]}\nT1/2max:{s7[num,4]}s,TR:{s7[num,5]}\nx0:{round(s7[num,6],1)},b:{round(s7[num,7],5)},c:{round(s7[num,8],2)}\nd:{round(s7[num,9],5)},e:{round(s7[num,10],2)}', #文本内容
                        fontdict=dict(fontsize=35, color='r',family='MicroSoft YaHei',),#字体属性字典
                        #添加文字背景色
                        bbox={'facecolor': '#74C476', #填充色
                            'edgecolor':'b',#外框色
                            'alpha': 0.5, #框透明度
                            'pad': 8,#本文与框周围距离 
                            }
                        )
                    plt.axhline(y=y_yuce[int(s7[num,9])])
                    plt.axvline(x=s7[num,6]-1)
                    plt.axvline(x=x_scatter[int(s7[num,9])])
            plt.savefig(dir+"/duodian_fenxi.jpg")
            plt.show()
            pool.shutdown()
          jisuanjieshu=datetime.now()
          print('---计算结束---',str(jisuanjieshu),'---耗时---',(jisuanjieshu-kaishijisuan).seconds,'s')
          print('---总耗时---',(jisuanjieshu-kaishi).seconds,'s')
          
        
                    

  global chizi,dirs,frame_1
  chizi=chizichangdu
  dirs=os.listdir(peizhun_savepath)
  dirs.sort(key=lambda x: int(x.split('.')[0]))
  frame_1=cv2.imread(peizhun_savepath+"/0.jpg")
  h,w,c=frame_1.shape
  dirs_len=len(dirs)
  list1=np.zeros(shape=(h,w,c,dirs_len))
  for i in range(dirs_len):
    path=os.path.join(peizhun_savepath,dirs[i])
    img_BGR = cv2.imread(path)
    # 使用bgr转化hsv
    hsv = cv2.cvtColor(img_BGR,cv2.COLOR_BGR2HSV)
    list1[:,:,:,i]=hsv
  cv2.namedWindow('frame_1',cv2.WINDOW_NORMAL)
  cv2.setMouseCallback('frame_1', OnMouseAction)
  while True:
    cv2.imshow('frame_1',frame_1)
    if cv2.waitKey(1) & 0xFF == ord('p'):
        mask=np.zeros(shape=(h, w))
        h5=h//5
        w5=w//5
        h5w5=h5*w5
        point=np.zeros(shape=(h5w5,dirs_len))
        p=0
        for i in range(0,h5*5,5):
            for j in range(0,w5*5,5):
                for dir in range(dirs_len):
                    for t in range(5):
                        for z in range(5):
                            if list1[i+t,z+j,0,dir]>=35 and list1[i+t,z+j,0,dir]<=77:
                                p+=list1[i+t,z+j,1,dir]
                    p1=p/25
                    point[int((i/5)*w5+(j/5)),dir]=p1
                    p=0
        
        for i in range(h5w5):
            y_point=[]
            y_med = signal.medfilt(point[i,:], kernel_size=13)
            y_med=np.array(y_med)
            y_med=y_med.flatten()
            #y_med=y_med[y_med>0]
            if y_med[y_med>0].size == 0:
                num_h=i//w5
                num_w=i%w5
                mask[num_h,(num_w-1)*5:num_w*5]=0
                
            else:
                #y_med=np.insert(y_med,0,0)
                tidu=np.asarray([y_med[i]-y_med[i-1] for i in range(1,len(y_med))])
                tidu_index=np.where(tidu>0)[0][0]+1
                x=np.arange(len(y_med))
                x0,b,c,d_x,e=ransac1(x,y_med,0.8,400,3,0.8,tidu_index)
                #y_med1 = preprocess1.fit_transform(y_med)
                #y_med2 = preprocess1.fit_transform(y_med)
                #tmax=(np.log(0.01*(np.exp(-b*x0)+t/c))/-b)
                #thalfmax=(np.log(0.5*(np.exp(-b*x0)+t/c))/-b)
                #print(position_line3[0]+2,wz)
                for i in x:
                    y_point.append(func1(i,x0,b,c,d_x,e))
                index=np.where(y_point>=(max(y_point)*0.99))[0][0]
                
                y_point_new=y_point[tidu_index-1:index+1]
                
                y_12=int(y_point[index])/2
                
                for i in range(len(y_point_new)):
                    if y_point_new[i] >= y_12:
                        t12max=round((i/y_point_new[i]*y_12)/3,2)
                        break
                
                Fmax=round(int(y_point[index]),1)
                Tmax=round(int(x[index]-x0)/3,2)
                Slope=round(Fmax/Tmax,2)
                TR=round(t12max/Tmax,2)
                with open('./model.pickle', 'rb') as f:
                    model = pickle.load(f)
                score=round(model.predict([[Tmax,Slope,t12max,TR]])[0],1)
                if score>100:
                    score=100
                elif score<0:
                    score=0
                #cv2.putText(frame,f'Tmax:{round((tmax-x0)/3,1)}s,Fmax:{round(0.95*(np.exp(-b*x0)*c+t),1)}' , (position_line3[0]+30,wz), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)
                num_h=i//w5
                num_w=i%w5
                mask[num_h,(num_w-1)*5:num_w*5]=score
        
        color_array = frame
        for i in range(h):
            for j in range(w):
                if mask[i,j]<60 and mask[i,j]>0:
                    color_array[i,j]=[255-(mask[i,j]*3),0,0]
                if mask[i,j]>=60:
                    color_array[i,j]=[0,0,135+3*(mask[i,j]-60)]
        frame=cv2.addWeighted(frame,0.5,color_array,0.5,0)
        
                
            

                

    # 按 q 键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.imwrite(dir+"/biaojitu.jpg", frame_1)
        break