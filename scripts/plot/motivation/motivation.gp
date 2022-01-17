set terminal postscript eps color "Arial_Bold, 20"

set size 1,0.75
set xlabel "NoC Wire Width (bit)" font "Arial_Bold, 20" offset 0,0.3
set ylabel "Maximum Delay (cycles)" font "Arial_Bold, 20" offset 2,0
set y2label "Bandwidth Waste (%)" font "Arial_Bold, 20" offset -4,0
set grid
# unset border
set key outside horizontal top center font "Arial_Bold, 16"

set logscale x
set xrange [100:2500]
set xtics ("128" 128, "256" 256, "512" 512, "1024" 1024, "2048" 2048)

set ytics nomirror offset 0.8,0
set yrange[0:120]
set y2range[0:100]
set y2tics 0,20,100 offset -1,0

set output "motivation.eps"

plot "packet_size_small.dat" using 1:2 with linespoints linecolor 1 linewidth 5 pointtype 5 pointsize 1.5 axis x1y1 title "8K-Byte_Delay" noenhanced,\
     "packet_size_small.dat" using 1:3 with linespoints linecolor 6 linewidth 5 pointtype 3 pointsize 2 axis x1y2 title "8K-Byte_Waste" noenhanced,\
     "packet_size_large.dat" using 1:2 with linespoints linecolor 4 linewidth 5 pointtype 5 pointsize 1.5 axis x1y1 title "16K-Byte_Delay" noenhanced,\
     "packet_size_large.dat" using 1:3 with linespoints linecolor 5 linewidth 5 pointtype 3 pointsize 2 axis x1y2 title "16K-Byte_Waste" noenhanced,\

set output '|ps2pdf -dEPSCrop motivation.eps motivation.pdf'
replot