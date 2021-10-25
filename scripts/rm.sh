cd ..
rm *.txt *.log *.yaml *.xml
cat /dev/null > nohup.out
cd focus-final-out
for i in `ls | grep csv`; do cat /dev/null >$i; done