from peizhun1 import peizhun
import os


dirs=os.listdir("F:/ygszfx/lanxiumei/lxm/gwh/")
dirs.sort(key=lambda x: int(x.split('.')[0]))
path1="F:/ygszfx/lanxiumei/lxm/gwh/1.jpg"
for i in range(len(dirs)):
    path2=os.path.join("F:/ygszfx/lanxiumei/lxm/gwh/",dirs[i])
    peizhun(path1,path2,path2)

