##给GPM的nc数据添加台风标
import os
from netCDF4 import Dataset
import numpy as np
import pymysql
def addTyphLabel():
    dir = "D:\\Precipitation\\data\\1h_GPM"
    #连接数据库
    conn = pymysql.connect(
        host='localhost',
        user='root',
        password='root',
        db='typhoon',
        charset='utf8',
           # autocommit=True,    # 如果插入数据，， 是否自动提交? 和conn.commit()功能一致。
    )
    #创建游标
    cur = conn.cursor()

    for year in os.listdir(dir):
        if(int(year)>=2017):
            yearDir = os.path.join(dir, year)
            for month in os.listdir(yearDir):
                monthDir = os.path.join(yearDir, month)
                for day in os.listdir(monthDir):
                    dayDir = os.path.join(monthDir, day)
                    for file in os.listdir(dayDir):

                        #获取GPM起止时间
                        date = file[0:8];
                        st_time = date+file[10:12]+"00"
                        end_time = date+file[14:16]+"00"
                        #数据库查询
                        try:
                            sqli = "select * FROM typhoontime where st_time<="+st_time+" and end_time>="+end_time;
                            result = cur.execute(sqli)
                            nc_file = Dataset(os.path.join(dayDir, file),'r+')
                            nc_file.createDimension('typhoon', 1)
                            nc_file.createVariable('typhoon', np.int, ('typhoon'))
                            nc_file.variables['typhoon'][:] = result
                            nc_file.close()
                            print(file + "   " + str(result));
                        except Exception as e:
                            print(e)