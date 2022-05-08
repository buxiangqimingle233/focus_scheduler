set terminal postscript eps color "Arial, 28"
unset key
set size square
set cbrange[-1:10]

unset xtics
unset ytics

set palette rgbformulae 33, 13, 10

set output "./mapping_vis/mapping_visualization_Tetris.eps"
plot "./mapping_vis/mapping_result_Tetris.dat" with image pixels
set output '|ps2pdf -dEPSCrop ./mapping_vis/mapping_visualization_Tetris.eps ./mapping_vis/mapping_visualization_Tetris.pdf'
replot

set output "./mapping_vis/mapping_visualization_Zig-Zag.eps"
plot "./mapping_vis/mapping_result_Zig-Zag.dat" with image pixels
set output '|ps2pdf -dEPSCrop ./mapping_vis/mapping_visualization_Zig-Zag.eps ./mapping_vis/mapping_visualization_Zig-Zag.pdf'
replot

set output "./mapping_vis/mapping_visualization_Hilbert.eps"
plot "./mapping_vis/mapping_result_Hilbert.dat" with image pixels
set output '|ps2pdf -dEPSCrop ./mapping_vis/mapping_visualization_Hilbert.eps ./mapping_vis/mapping_visualization_Hilbert.pdf'
replot