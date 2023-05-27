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

def shipin1(openpath,dir,chizichangdu):
  cap = cv2.VideoCapture(openpath)
  frameNum = 0
  quzhen_savepath=dir+'/'+'quzhen'
  folder=os.path.exists(quzhen_savepath)
  if not folder:
      os.makedirs(quzhen_savepath)
  while (cap.isOpened()):
      ret, frame = cap.read()
      frameNum = frameNum + 1
      if frameNum % 10 == 0:  # 调整帧数
          if ret:
              new_img = cv2.resize(frame,None,fx=0.3,fy=0.3,interpolation = cv2.INTER_LINEAR)
              cv2.imwrite(quzhen_savepath+'/'+str(frameNum//10) + ".jpg", new_img)  # 保存图片
          # cv2.namedWindow("resized", 0)  # 0可以改变窗口大小了
          # # cv2.resizeWindow("resized", 640, 480) # 设置固定大小不能该，上面和一起使用
          # cv2.imshow("resized", frame)  # 显示视频
          else:
              break
      if cv2.waitKey(1) & 0xFF == ord('q'):
          break
  cap.release
  cv2.destroyAllWindows()
  
  dirs1=os.listdir(quzhen_savepath+'/')
  dirs1.sort(key=lambda x: int(x.split('.')[0]))
  path1=quzhen_savepath+'/1.jpg'
  peizhun_savepath=dir+'/'+'peizhun'
  folder=os.path.exists(peizhun_savepath)
  if not folder:
      os.makedirs(peizhun_savepath)
  for i in range(len(dirs1)):
    path3=os.path.join(peizhun_savepath,dirs1[i])
    path2=os.path.join(quzhen_savepath,dirs1[i])
    peizhun(path1,path2,path3)

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
              print("超过最大拟合次数！")
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
                  print("更新结果:x0={}, b={}, c={},d={},e={}, 最佳内点比例={}"
                      .format(best_x0,best_b,best_c,best_d,best_e, Pro_current))
              if ((best_inliers_number / N) > ProT):
                  print("更新结果：终止，参数，内点比例=", best_x0,best_b,best_c,best_d,best_e,(best_inliers_number / N), "大于期望内点比例=", ProT)
                  break
              if i>0.9*Iter_Number:
                  print("i>Iter_Number")
                  x0_qz=math.ceil(best_x0)
                  y1=y[:x0_qz+1]
                  y2=y[x0_qz+1:]
                  y=np.append(y1,y2[y2>0])
                  N=len(y)
                  x=np.arange(N)
              if i> Iter_Number:
                  break
      return best_x0,best_b,best_c,best_d,best_e,best_inliers_number / N

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
          cv2.putText(frame_1,f"{chizi}mm" , (position_line[0]+2,position_line[1]+2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
      elif event == cv2.EVENT_LBUTTONDOWN:
          position_line2 = (x,y)
          cv2.circle(frame_1, position_line2, 2, (255,0,0), -1)
      elif event == cv2.EVENT_LBUTTONUP:
          z=1
          position_line3 = (x,y)
          cv2.line(frame_1, position_line2, position_line3, (255,0,0), 1)
          d1=math.sqrt((position_line3[0]-position_line2[0])**2+(position_line3[1]-position_line2[1])**2)
          zhixian_x=np.arange(position_line2[0],position_line3[0],(position_line3[0]-position_line2[0])/d1*5*(d/int(chizi)))
          zhixian_x=[int(zhixian_x[i]) for i in range(len(zhixian_x))]
          zhixian_y=np.arange(position_line2[1],position_line3[1],(position_line3[1]-position_line2[1])/d1*5*(d/int(chizi)))
          zhixian_y=[int(zhixian_y[i]) for i in range(len(zhixian_y))]
          zhixian=np.hstack((np.array(zhixian_x).reshape(-1,1),np.array(zhixian_y).reshape(-1,1)))
          print(zhixian)
          for j,i in enumerate(zhixian):
          #cv2.rectangle(frame, position_line2, (position_line3[0],i), (255,0,0), 1)
          # d1=math.sqrt((position_line2[0]-position_line3[0])**2+(position_line2[1]-position_line3[1])**2)
          # cv2.line(frame, position_line2, position_line3, (255,0,0), 2)
              cv2.circle(frame_1, center=i, radius=2,color=(255, 0, 0), thickness=-1)
              cv2.putText(frame_1,f"{j*5}mm" , (i[0]+4,i[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (255, 0, 0), 1)
          #cv2.putText(frame,f"{round((int(chizi)/d)*d1,1)}mm" , (position_line3[0]+4,position_line3[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)
          plt.rc("font", **font)
          plt.figure(figsize=(40,20*len(zhixian)), dpi=80)
          for num,wz in enumerate(zhixian):
              y_point=[]
              hl=int(wz[1]-2)
              hh=int(wz[1]+3)
              wl=int(wz[0]-2)
              wh=int(wz[0]+3)
              print(wl,wh)
              point=np.zeros(shape=(len(dirs)))
              p=0
              for i in range(len(dirs)):
                  for j in range(hl,hh):
                      for h in range(wl,wh):
                          if list1[j,h,0,i]>=100 and list1[j,h,0,i]<=124:
                              p+=list1[j,h,1,i]
                  p1=p/25
                  point[i]=p1
                  p=0
              print('2')
              
              y_med = signal.medfilt(point, kernel_size=13)
              y_med=np.array(y_med)
              y_med=y_med.flatten()
              #y_med=y_med[y_med>0]
              if y_med[y_med>0].size == 0:
                  #y_med=np.zeros((len(dirs)))
                  x=np.arange(len(dirs))
                  cv2.putText(frame_1,'0', (wz[0]+40,wz[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
                  plt.subplot(len(zhixian),1,len(zhixian)-num-1)
                  plt.scatter(x,y_med,s=60)
                  plt.plot(x, y_med, '-p', color='grey',
                      marker = 'o',
                      markersize=3, linewidth=5,
                      markerfacecolor='red',
                      markeredgecolor='red',
                      markeredgewidth=2)
                  plt.xlabel('帧数', fontsize=35)
                  plt.ylabel('荧光强度', fontsize=35)
                  plt.title(f'第{num+1}个点的ICG曲线', fontsize=50)
                  
                  z=0
              if z:
                  #y_med=np.insert(y_med,0,0)
                  tidu=np.asarray([y_med[i]-y_med[i-1] for i in range(1,len(y_med))])
                  tidu_index=np.where(tidu>0)[0][0]+1
                  x=np.arange(len(y_med))
                  x0,b,c,d_x,e,ratio1=ransac1(x,y_med,0.8,400,3,0.8,tidu_index)
                  
                  #y_med1 = preprocess1.fit_transform(y_med)
                  #y_med2 = preprocess1.fit_transform(y_med)
                  #tmax=(np.log(0.01*(np.exp(-b*x0)+t/c))/-b)
                  #thalfmax=(np.log(0.5*(np.exp(-b*x0)+t/c))/-b)
                  #print(position_line3[0]+2,wz)
                  for i in x:
                      y_point.append(func1(i,x0,b,c,d_x,e))
                  index=np.where(y_point>=(max(y_point)*0.99))[0][0]
                  print(index)
                  y_point_new=y_point[tidu_index-1:index+1]
                  print(y_point_new)
                  y_12=int(y_point[index])/2
                  print(y_12)
                  for i in range(len(y_point_new)):
                      if y_point_new[i] >= y_12:
                          t12max=round((i/y_point_new[i]*y_12)/3,2)
                          break
                  print('3')
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
                  cv2.putText(frame_1,f'{score}', (wz[0]+40,wz[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (255, 0, 0), 1)
                  
                  plt.subplot(len(zhixian),1,len(zhixian)-num)
                  plt.scatter(x,y_med,s=60)
                  plt.plot(x, y_point, '-p', color='red',
                      marker = 'o',
                      markersize=3, linewidth=5,
                      markerfacecolor='red',
                      markeredgecolor='red',
                      markeredgewidth=2)
                  plt.xlabel('帧数', fontsize=35)
                  plt.ylabel('荧光强度', fontsize=35)
                  plt.title(f'距离起点{5*(num+1)}mm的ICG曲线', fontsize=50)
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
                  #plt.savefig(f"F:/ygszfx/mfx/2whq{gs}.jpg")

  global chizi,dirs,frame_1
  chizi=chizichangdu
  dirs=os.listdir(peizhun_savepath)
  dirs.sort(key=lambda x: int(x.split('.')[0]))
  frame_1=cv2.imread(peizhun_savepath+"/1.jpg")
  h,w,c=frame_1.shape
  list1=np.zeros(shape=(h,w,c,len(dirs)))
  for i in range(len(dirs)):
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
        point=np.zeros(shape=(h5w5,len(dirs)))
        p=0
        for i in range(0,h5*5,5):
            for j in range(0,w5*5,5):
                for dir in range(len(dirs)):
                    for t in range(5):
                        for z in range(5):
                            if list1[i+t,z+j,0,dir]>=100 and list1[i+t,z+j,0,dir]<=124:
                                p+=list1[i+t,z+j,1,dir]
                    p1=p/25
                    point[int((i/5)*w5+(j/5)),dir]=p1
                    p=0
        print(point.shape)
        for i in range(h5w5):
            l=1
            y_point=[]
            y_med = signal.medfilt(point[i,:], kernel_size=13)
            y_med=np.array(y_med)
            y_med=y_med.flatten()
            #y_med=y_med[y_med>0]
            if y_med[y_med>0].size == 0:
                #y_med=np.zeros((len(dirs)))
                num_h=i//w5
                num_w=i%w5
                mask[num_h,(num_w-1)*5:num_w*5]=0
                l=0
            if l:
                #y_med=np.insert(y_med,0,0)
                tidu=np.asarray([y_med[i]-y_med[i-1] for i in range(1,len(y_med))])
                tidu_index=np.where(tidu>0)[0][0]+1
                x=np.arange(len(y_med))
                x0,b,c,d_x,e,ccy1,ratio1=ransac1(x,y_med,0.8,400,3,0.8,tidu_index)
                #y_med1 = preprocess1.fit_transform(y_med)
                #y_med2 = preprocess1.fit_transform(y_med)
                #tmax=(np.log(0.01*(np.exp(-b*x0)+t/c))/-b)
                #thalfmax=(np.log(0.5*(np.exp(-b*x0)+t/c))/-b)
                #print(position_line3[0]+2,wz)
                for i in x:
                    y_point.append(func1(i,x0,b,c,d_x,e))
                index=np.where(y_point>=(max(y_point)*0.99))[0][0]
                print(index)
                y_point_new=y_point[tidu_index-1:index+1]
                print(y_point_new)
                y_12=int(y_point[index])/2
                print(y_12)
                for i in range(len(y_point_new)):
                    if y_point_new[i] >= y_12:
                        t12max=round((i/y_point_new[i]*y_12)/3,2)
                        break
                print('3')
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
        print(mask)
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