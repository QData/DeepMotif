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


offset <- -13
offset <- -40
offset <- -10


colors=NULL


# color_map <- colorRampPalette(c("blue","red"))(n = 500)
# colors = c(seq(-1,0,length=100),seq(0.001,1,length=100)) # for yellow\

color_map <- bluered
# color_map <- colorRampPalette(c("white","red"))(n = 39)



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
