import csv
import json
import os
n=2017
while n<=2019:
    path = "./json/"+str(n) #文件夹目录
    files= os.listdir(path) #得到文件夹下的所有文件名称
    for file in files:
        if file !=".json":
            with open("./json/"+str(n)+"/"+file,'r',encoding="GBK") as load_f:
                load_dict = json.load(load_f)
                typhoon=load_dict['typhoon']
                typhPointList=typhoon[8]

                # 判断路径是否存在
                path = './csv/'+str(n)+'/'
                isExists = os.path.exists(path)
                if not isExists:
                    # 如果不存在则创建目录
                    os.makedirs(path)
                with open("./csv/"+str(n)+"/"+file[:-4]+"csv", "w") as csvfile:
                    writer = csv.writer(csvfile)
                    fieldnames = ['num','time','lon','lat','press','maxSpeed','moveDir','moveSpeed']
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()
                    for i in range(len(typhPointList)):
                        writer.writerow({
                            'num':file[:-5],
                            'time': typhPointList[i][1],
                            'lon': typhPointList[i][4],
                            'lat':typhPointList[i][5],
                            'press':typhPointList[i][6],
                            'maxSpeed':typhPointList[i][7],
                            'moveDir':typhPointList[i][8],
                            'moveSpeed':typhPointList[i][9]
                        })
    n=n+1;