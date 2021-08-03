set terminal postscript eps color "Arial, 28"
unset key
set size square
set cbrange[-1:10]

unset xtics
unset ytics

set palette rgbformulae 33, 13, 10

set output "mapping_visualization.eps"
plot "mapping_result.dat" with image pixels
set output '|ps2pdf -dEPSCrop mapping_visualization.eps mapping_visualization.pdf'
replot