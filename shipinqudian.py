import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import scipy.signal as signal
from scipy.optimize import curve_fit
import random
import math
import sys
import pickle

font = {'family' : 'MicroSoft YaHei',
        'weight' : 'bold',
        'size'   : 30}

def func1(x, x0,b,c,d,e):
    return np.piecewise(x, [x < x0, x >= x0], [lambda x:0, 
                                   lambda x:np.exp(-b*x0)*c-np.exp(-b*x)*c+np.exp(-d*x)*e])
def ransac1(x,y,rate,Iter_Number,DisT,ProT,qsd):
    chongcaiyang=0
    N=len(x)
    best_inliers_number =0
    i=0
    exception=0
    p0=qsd,0.1,1,0.1,1
    while True:
        x_2c= random.sample(list(x), int(N*rate))
        y_2c=y[x_2c]
        try:
            popt, pcov = curve_fit(func1, x_2c, y_2c,p0,maxfev=800000)
        except Exception:
            print("超过最大拟合次数！")
            exception=exception+1
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
                print("更新结果：x0={}, b={}, c={},d={},e={}, 最佳内点比例={}"
                    .format(best_x0,best_b,best_c,best_d,best_e, Pro_current))
            if ((best_inliers_number / N) > ProT):
                print("更新结果：终止，参数，内点比例=", best_x0,best_b,best_c,best_d,best_e,(best_inliers_number / N), "大于期望内点比例=", ProT)
                chongcaiyang=1
                break
            if i>Iter_Number:
                print("i>Iter_Number")
                x0_qz=math.ceil(best_x0)
                y1=y[:x0_qz+1]
                y2=y[x0_qz+1:]
                y=np.append(y1,y2[y2>0])
                N=len(y)
                x=np.arange(N)
            if i>1.5*Iter_Number:
                chongcaiyang=0
                break
            if exception>5:
                chongcaiyang=0
                break
    return best_x0,best_b,best_c,best_d,best_e,chongcaiyang,best_inliers_number / N

def OnMouseAction(event, x, y, flags, param):
    global frame, position_line,position_line1,position_line2,position_line3,position_line4,position_line5,d,d1,d2
    if event == cv2.EVENT_RBUTTONDOWN:
        position_line = (x,y)
        cv2.circle(frame, position_line, 2, (0,0,0), -1)
    # elif event == cv2.EVENT_MOUSEMOVE and (flags & cv2.EVENT_FLAG_RBUTTON):
    #     cv2.line(frame, position_line, (x,y), (0,0,0), 2)
    elif event == cv2.EVENT_RBUTTONUP:
        position_line1 = (x,y)
        d=math.sqrt((position_line[0]-position_line1[0])**2+(position_line[1]-position_line1[1])**2)
        cv2.line(frame, position_line, position_line1, (0,0,0), 2)
        cv2.putText(frame,f"{chizi}mm" , (position_line[0]+2,position_line[1]+2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    elif event == cv2.EVENT_MBUTTONDOWN:
        position_line2 = (x,y)
        cv2.circle(frame, position_line2, 2, (255,0,0), -1)
    elif event == cv2.EVENT_MBUTTONUP:
        z=1
        position_line3 = (x,y)
        cv2.line(frame, position_line2, position_line3, (255,0,0), 1)
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
            cv2.circle(frame, center=i, radius=2,color=(255, 0, 0), thickness=-1)
            cv2.putText(frame,f"{j*5}mm" , (i[0]+4,i[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)
        #cv2.putText(frame,f"{round((int(chizi)/d)*d1,1)}mm" , (position_line3[0]+4,position_line3[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)
        plt.rc("font", **font)
        plt.figure(figsize=(40,20*len(zhixian)), dpi=80)
        for num,wz in enumerate(zhixian):
            ccy1=1
            ccy2=1
            ccy3=1
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
                        if list1[j,h,0,i]>=35 and list1[j,h,0,i]<=77:
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
                cv2.putText(frame,'0', (wz[0]+40,wz[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
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
                x0,b,c,d_x,e,ccy1,ratio1=ransac1(x,y_med,0.8,1000,3,0.9,tidu_index)
                ratio2=0
                ratio3=0
                if ccy1==0:
                    print("chongcaiyang!!!!")
                    hl=int(wz[1]-2)
                    hh=int(wz[1]+3)
                    wl=int(wz[0]+1)
                    wh=int(wz[0]+6)
                    print(position_line2[0])
                    print(wl,wh)
                    point=np.zeros(shape=(len(dirs)))
                    p=0
                    for i in range(len(dirs)):
                        for j in range(hl,hh):
                            for h in range(wl,wh):
                                if list1[j,h,0,i]>=35 and list1[j,h,0,i]<=77:
                                    p+=list1[j,h,1,i]
                        p1=p/25
                        point[i]=p1
                        p=0
                    print('2')
                    y_med = signal.medfilt(point, kernel_size=13)
                    y_med=np.array(y_med)
                    y_med=y_med.flatten()
                    csz=np.mean(y_med[0:10])
                    x=np.arange(len(y_med))
                    x01,b1,c1,d_x1,e1,ccy2,ratio2=ransac1(x,y_med,0.8,1000,3,0.9,tidu_index)
                if ccy2==0:
                    print("Again chongcaiyang!!!!")
                    hl=int(wz[1]-2)
                    hh=int(wz[1]+3)
                    wl=int(wz[0]-5)
                    wh=int(wz[0])
                    print(wl,wh)
                    point=np.zeros(shape=(len(dirs)))
                    p=0
                    for i in range(len(dirs)):
                        for j in range(hl,hh):
                            for h in range(wl,wh):
                                if list1[j,h,0,i]>=35 and list1[j,h,0,i]<=77:
                                    p+=list1[j,h,1,i]
                        p1=p/25
                        point[i]=p1
                        p=0
                    print('2')
                    y_med = signal.medfilt(point, kernel_size=13)
                    y_med=np.array(y_med)
                    y_med=y_med.flatten()
                    csz=np.mean(y_med[0:10])
                    x=np.arange(len(y_med))
                    x02,b2,c2,d_x2,e2,ccy3,ratio3=ransac1(x,y_med,0.8,1000,3,0.9,tidu_index)
                if ratio3 > ratio2:
                    x01,b1,c1,d_x1,e1=x02,b2,c2,d_x2,e2
                    print("again success:chongcaiyang!!!!")
                if ratio2> ratio1:
                    x0,b,c,d_x,e=x01,b1,c1,d_x1,e1
                    print("success:chongcaiyang!!!!")
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
                #cv2.putText(frame,f'Tmax:{round((tmax-x0)/3,1)}s,Fmax:{round(0.95*(np.exp(-b*x0)*c+t),1)}' , (position_line3[0]+30,wz), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)
                cv2.putText(frame,f'{score}', (wz[0]+40,wz[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
                
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
                    s=f'Tmax:{Tmax}s,Fmax:{Fmax},Slope:{Slope}\nT1/2max:{t12max}s,TR:{TR}\nx0:{round(x0,1)},b:{round(b,5)},c:{round(c,2)}\nd:{round(d,5)},e:{round(e,2)}', #文本内容
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
        plt.savefig(f"F:/ygszfx/ceshi.jpg")
        plt.show()
                #plt.savefig(f"F:/ygszfx/mfx/2whq{gs}.jpg")


    elif event == cv2.EVENT_LBUTTONDOWN:                                          
        position_line4 = (x,y)
    elif event == cv2.EVENT_LBUTTONUP:
        l=1
        position_line5 = (x,y)
        d2=math.sqrt((position_line4[0]-position_line5[0])**2+(position_line4[1]-position_line5[1])**2)
        cv2.line(frame, position_line4, position_line5, (0,0,0), 2)
        cv2.putText(frame,f"{round(chizi/d*d2,1)}mm" , (position_line5[0]+2,position_line5[1]+2), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)
        ccy1=1
        ccy2=1
        ccy3=1
        y_point=[]
        hl=int(position_line5[1]-2)
        hh=int(position_line5[1]+3)
        wl=int(position_line5[0]-2)
        wh=int(position_line5[0]+3)
        print(wl,wh)
        point=np.zeros(shape=(len(dirs)))
        p=0
        for i in range(len(dirs)):
            for j in range(hl,hh):
                for h in range(wl,wh):
                    if list1[j,h,0,i]>=35 and list1[j,h,0,i]<=77:
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
            cv2.putText(frame,'0', (position_line5[0]+40,position_line5[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
            plt.rc("font", **font)
            plt.figure(figsize=(40,20), dpi=80)
            plt.scatter(x,y_med,s=60)
            plt.plot(x, y_med, '-p', color='grey',
                marker = 'o',
                markersize=3, linewidth=5,
                markerfacecolor='red',
                markeredgecolor='red',
                markeredgewidth=2)
            plt.xlabel('帧数', fontsize=35)
            plt.ylabel('荧光强度', fontsize=35)
            plt.title(f'距离起点{round(chizi/d*d2,1)}mm的ICG曲线', fontsize=50)
            plt.savefig(f"F:/ygszfx/ceshi.jpg")
            plt.show()
            l=0
        if l:
            #y_med=np.insert(y_med,0,0)
            tidu=np.asarray([y_med[i]-y_med[i-1] for i in range(1,len(y_med))])
            tidu_index=np.where(tidu>0)[0][0]+1
            x=np.arange(len(y_med))
            x0,b,c,d_x,e,ccy1,ratio1=ransac1(x,y_med,0.8,1000,3,0.9,tidu_index)
            ratio2=0
            ratio3=0
            if ccy1==0:
                print("chongcaiyang!!!!")
                hl=int(position_line5[1]-2)
                hh=int(position_line5[1]+3)
                wl=int(position_line5[0]+1)
                wh=int(position_line5[0]+6)
                print(position_line2[0])
                print(wl,wh)
                point=np.zeros(shape=(len(dirs)))
                p=0
                for i in range(len(dirs)):
                    for j in range(hl,hh):
                        for h in range(wl,wh):
                            if list1[j,h,0,i]>=35 and list1[j,h,0,i]<=77:
                                p+=list1[j,h,1,i]
                    p1=p/25
                    point[i]=p1
                    p=0
                print('2')
                y_med = signal.medfilt(point, kernel_size=13)
                y_med=np.array(y_med)
                y_med=y_med.flatten()
                x=np.arange(len(y_med))
                x01,b1,c1,d_x1,e1,ccy2,ratio2=ransac1(x,y_med,0.8,1000,3,0.9,tidu_index)
            if ccy2==0:
                print("Again chongcaiyang!!!!")
                hl=int(position_line5[1]-2)
                hh=int(position_line5[1]+3)
                wl=int(position_line5[0]-5)
                wh=int(position_line5[0])
                print(wl,wh)
                point=np.zeros(shape=(len(dirs)))
                p=0
                for i in range(len(dirs)):
                    for j in range(hl,hh):
                        for h in range(wl,wh):
                            if list1[j,h,0,i]>=35 and list1[j,h,0,i]<=77:
                                p+=list1[j,h,1,i]
                    p1=p/25
                    point[i]=p1
                    p=0
                print('2')
                y_med = signal.medfilt(point, kernel_size=13)
                y_med=np.array(y_med)
                y_med=y_med.flatten()
                x=np.arange(len(y_med))
                x02,b2,c2,d_x2,e2,ccy3,ratio3=ransac1(x,y_med,0.8,1000,3,0.9,tidu_index)
            if ratio3 > ratio2:
                x01,b1,c1,d_x1,e1=x02,b2,c2,d_x2,e2
                print("again success:chongcaiyang!!!!")
            if ratio2> ratio1:
                x0,b,c,d_x,e=x01,b1,c1,d_x1,e1
                print("success:chongcaiyang!!!!")
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
            #cv2.putText(frame,f'Tmax:{round((tmax-x0)/3,1)}s,Fmax:{round(0.95*(np.exp(-b*x0)*c+t),1)}' , (position_line3[0]+30,wz), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)
            cv2.putText(frame,f'{score}', (position_line5[0]+40,position_line5[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
            plt.rc("font", **font)
            plt.figure(figsize=(40,20), dpi=80)
            plt.scatter(x,y_med,s=60)
            plt.plot(x, y_point, '-p', color='red',
                marker = 'o',
                markersize=3, linewidth=5,
                markerfacecolor='red',
                markeredgecolor='red',
                markeredgewidth=2)
            plt.xlabel('帧数', fontsize=35)
            plt.ylabel('荧光强度', fontsize=35)
            plt.title(f'距离起点{round(chizi/d*d2,1)}mm的ICG曲线', fontsize=50)
            plt.text(x=0.5*len(x),#文本x轴坐标 
                y=0.5*Fmax, #文本y轴坐标
                s=f'Tmax:{Tmax}s,Fmax:{Fmax},Slope:{Slope}\nT1/2max:{t12max}s,TR:{TR}\nx0:{round(x0,1)},b:{round(b,5)},c:{round(c,2)}\nd:{round(d,5)},e:{round(e,2)}', #文本内容
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
            plt.savefig(f"F:/ygszfx/ceshi.jpg")
            plt.show()
                


 
    # elif event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_LBUTTON:      #按住左键拖曳不放开
    #     position2 = (x,y)
        
    # elif event == cv2.EVENT_LBUTTONUP:                                          #放开左键
    #     position2 = (x,y)  



if __name__ == '__main__':
    #chizi = sys.argv[1]
    chizi=100
    gs=0
    dirs=os.listdir("F:/ygszfx/lanxiumei/lxm/whq")
    dirs.sort(key=lambda x: int(x.split('.')[0]))
    frame=cv2.imread("F:/ygszfx/lanxiumei/lxm/whq/1.jpg")
    h,w,c=frame.shape
    list1=np.zeros(shape=(h,w,c,len(dirs)))
    for i in range(len(dirs)):
        path=os.path.join("F:/ygszfx/lanxiumei/lxm/whq/",dirs[i])
        img_BGR = cv2.imread(path)
        # 使用bgr转化hsv
        hsv = cv2.cvtColor(img_BGR,cv2.COLOR_BGR2HSV)
        list1[:,:,:,i]=hsv
    cv2.namedWindow('frame',cv2.WINDOW_NORMAL)
    cv2.setMouseCallback('frame', OnMouseAction)
    while True:
        cv2.imshow('frame',frame)
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
                                if list1[i+t,z+j,0,dir]>=35 and list1[i+t,z+j,0,dir]<=77:
                                    p+=list1[i+t,z+j,2,dir]
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
                    x0,b,c,d_x,e,ccy1,ratio1=ransac1(x,y_med,0.8,1000,3,0.9,tidu_index)
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
            cv2.imwrite("F:/ygszfx/lanxiumei/lxm/biaojiwhq2.jpg", frame)
            break
    
        #plt.savefig(f'F:/ygszfx/ycl/220330-h-1-qudou-peizhun1-icg-junzhi-resize1/{i}.jpg')
        