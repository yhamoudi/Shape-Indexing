
reset

### pour pdf
 set term pdfcairo
 set output "courbe3.pdf" # le nom du fichier qui est engendre


stats 'courbe2' every ::::0 using 1 nooutput
NB_ALGOS = int(STATS_min)

#set title sprintf('Performances du descripteur 2 selon la distance utilisée')
set xlabel "Taille du descripteur (type 1 / type 2)"
set ylabel "Pourcentage de réussite sur le test set"


#set autoscale x

set style data linespoints

set pointsize 1   


plot for [i=2:(NB_ALGOS+1)] "<(sed '1d' courbe3)"  using i:xticlabels(1) title columnheader(i)
