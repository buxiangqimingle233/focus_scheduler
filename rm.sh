rm *.txt *.log *.yaml *.xml
cat /dev/null > nohup.out
for i in `find . -name "*.csv"`; do cat /dev/null >$i; done