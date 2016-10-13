library(gplots)
library(RColorBrewer)

args <- commandArgs(trailingOnly = TRUE)
# print(args[1])
# print(args[2])


#########################################################
### reading in data and transform it to matrix format
#########################################################

data <- read.csv(args[1], comment.char="#")
rnames <- data[,1]                            # assign labels in column 1 to "rnames"
mat_data <- data.matrix(data[,2:ncol(data)])  # transform column 2-5 into a matrix
rownames(mat_data) <- rnames                  # assign row names



#########################################################
### customizing and plotting heatmap
#########################################################

# creates a own color palette from red to green
my_palette <- colorRampPalette(c("red", "yellow", "green"))(n = 299)

# (optional) defines the color breaks manually for a "skewed" color transition
col_breaks = c(seq(-1,0,length=100),  # for red
  seq(0.001,0.7,length=100),              # for yellow
  seq(0.71,1,length=100))              # for green

# creates a 5 x 5 inch image
# png("h1_simple.png",
#   width = 5*300,        # 5 x 300 pixels
#   height = 5*300,
#   res = 300,            # 300 pixels per inch
#   pointsize = 8)        # smaller font size

# heatmap.2(t(mat_data),
#   cellnote = t(mat_data),  # same data set for cell labels
#   #notecol="black",      # change font color of cell labels to black
#   #density.info="none",  # turns off density plot inside color legend
#   trace="none",         # turns off trace lines inside the heat map
#   margins =c(12,9),     # widens margins around plot
#   col=my_palette,       # use on color palette defined earlier
#   breaks=col_breaks,    # enable color transition at specified limits
#   dendrogram="none",     # only draw a row dendrogram

if(length(args) < 3){
  offset <- -13
} else {
  offset <- as.integer(args[3])
}

if(length(args) < 4){
  color_map <- bluered
  colors = c(seq(0,0.5,length=20),seq(0.501,1,length=20)) #TODO change for certain graphs
} else {
  color_map <- colorRampPalette(c("white","red"))(n = 39)
  # colors = c(seq(0,0.05,length=20),seq(0.501,1,length=20)) #TODO change for certain graphs
  colors=NULL
}



png(file = args[2], units="in", width=11, height=8.5, res=300)

heatmap.2(t(mat_data),
	# cellnote = t(mat_data),
	Rowv=NULL,
	Colv="NA",
	srtCol=0,
	labRow = "",
	col = color_map,
  breaks = colors,
	scale="none",
	margins=c(3,0), # ("margin.Y", "margin.X")
	trace='none',
	symkey=FALSE,
	symbreaks=FALSE,
	dendrogram='none',
	offsetCol = offset,
	cexCol=0.9,
  key=FALSE, #change to turn on
  density.info=NULL,
  denscol="black",
  keysize=1,
  #( "bottom.margin", "left.margin", "top.margin", "left.margin" )
  key.par=list(mar=c(3.5,0,3,0)),
  # lmat -- added 2 lattice sections (5 and 6) for padding
  lmat=rbind(c(5, 4, 2), c(6, 1, 3)), lhei=c(2.5, 5), lwid=c(1, 10, 1)
  )



dev.off()
