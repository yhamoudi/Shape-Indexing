
reset

### pour pdf
 set term pdfcairo
 set output "courbe1.pdf" # le nom du fichier qui est engendre


stats 'courbe1' every ::::0 using 1 nooutput
NB_ALGOS = int(STATS_min)

#set title sprintf('Performances du descripteur 1 selon la distance utilisée')
set xlabel "Taille du descripteur"
set ylabel "Pourcentage de réussite sur le test set"


#set autoscale x

set style data linespoints

set pointsize 1   


plot for [i=2:(NB_ALGOS+1)] "<(sed '1d' courbe1)"  using i:xticlabels(1) title columnheader(i)
