set terminal postscript eps color "Arial_Bold, 32"
set output "exp.eps"

set size 2.85,1.1
set key box 
set key at screen 2.1,1.05
set key width 1.7
set key font "Arial, 24"
set key maxrows 1


set multiplot layout 1,4

set xlabel "NoC Wire Width (bit)" font "Arial_Bold, 28"
set ylabel "Avg Slowdown" offset 1.5,0 font "Arial_Bold, 28"
set xrange [200: 1795]
set yrange [1:4.5]
set grid

set xtics ("256" 256, "512" 512, "1024" 1024, "1280" 1280, "1536" 1536, "" 1792, "2048" 2048) font "Arial_Bold, 24" 
set ytics font "Arial_Bold, 24"

set size 0.8,1
set origin 0,0
set title "(a) Pipeline" 

plot "bert.dat" using 1:2 with linespoints linewidth 5 pointsize 2 title "METRO",\
     "bert.dat" using 1:3 with linespoints linewidth 5 pointsize 2 title "DOR",\
     "bert.dat" using 1:4 with linespoints linewidth 5 pointsize 2 title "XYYX",\
     "bert.dat" using 1:5 with linespoints linewidth 5 pointsize 2 title "ROMM",\
     "bert.dat" using 1:6 with linespoints linewidth 5 pointsize 2 title "MAD",\

set size 0.8,1
set origin 0.7,0
set title "(c) Hybrid-A" 
plot "resnets.dat" using 1:2 with linespoints linewidth 5 pointsize 2 title "METRO",\
     "resnets.dat" using 1:3 with linespoints linewidth 5 pointsize 2 title "DOR",\
     "resnets.dat" using 1:4 with linespoints linewidth 5 pointsize 2 title "XYYX",\
     "resnets.dat" using 1:5 with linespoints linewidth 5 pointsize 2 title "ROMM",\
     "resnets.dat" using 1:6 with linespoints linewidth 5 pointsize 2 title "MAD",\


set size 0.8,1
set origin 1.4,0
set title "(b) Hybrid-B" 
plot "mlperf.dat" using 1:2 with linespoints linewidth 5 pointsize 2 title "METRO",\
     "mlperf.dat" using 1:3 with linespoints linewidth 5 pointsize 2 title "DOR",\
     "mlperf.dat" using 1:4 with linespoints linewidth 5 pointsize 2 title "XYYX",\
     "mlperf.dat" using 1:5 with linespoints linewidth 5 pointsize 2 title "ROMM",\
     "mlperf.dat" using 1:6 with linespoints linewidth 5 pointsize 2 title "MAD",\

set size 0.8,1
set origin 2.1,0
set title "(c) Hybrid-C" 
plot "four.dat" using 1:2 with linespoints linewidth 5 pointsize 2 title "METRO",\
     "four.dat" using 1:3 with linespoints linewidth 5 pointsize 2 title "DOR",\
     "four.dat" using 1:4 with linespoints linewidth 5 pointsize 2 title "XYYX",\
     "four.dat" using 1:5 with linespoints linewidth 5 pointsize 2 title "ROMM",\
     "four.dat" using 1:6 with linespoints linewidth 5 pointsize 2 title "MAD",\


unset multiplot

set output '|ps2pdf -dEPSCrop exp.eps exp.pdf'
