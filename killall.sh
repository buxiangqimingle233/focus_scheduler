ps aux|grep timeloop-mapper |grep -v grep |awk '{print $2}'|xargs kill -9
ps aux|grep focus |grep -v grep |awk '{print $2}'|xargs kill -9