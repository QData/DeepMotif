library(gridGraphics)
library(grid)
library(gplots)
library(RColorBrewer)

grab_grob <- function(){
  grid.echo()
  grid.grab()
}



args <- commandArgs(trailingOnly = TRUE)
# print(args[1])
# print(args[2])


#########################################################
### reading in data and transform it to matrix format
#########################################################

data <- read.csv(args[1], comment.char="#")
rnames <- data[,1]                            # assign labels in column 1 to "rnames"
mat_data1 <- data.matrix(data[,2:ncol(data)])  # transform column 2-5 into a matrix
rownames(mat_data1) <- rnames


data <- read.csv(args[1], comment.char="#")
rnames <- data[,1]                            # assign labels in column 1 to "rnames"
mat_data2 <- data.matrix(data[,2:ncol(data)])  # transform column 2-5 into a matrix
rownames(mat_data2) <- rnames

data <- read.csv(args[1], comment.char="#")
rnames <- data[,1]                            # assign labels in column 1 to "rnames"
mat_data3 <- data.matrix(data[,2:ncol(data)])  # transform column 2-5 into a matrix
rownames(mat_data3) <- rnames


data <- read.csv(args[1], comment.char="#")
rnames <- data[,1]                            # assign labels in column 1 to "rnames"
mat_data4 <- data.matrix(data[,2:ncol(data)])  # transform column 2-5 into a matrix
rownames(mat_data4) <- rnames


arr <- list(mat_data1,mat_data2,mat_data3,mat_data4)


png(file = args[2], units="in", width=11, height=8.5, res=300)




color_map <- colorRampPalette(c("white","red"))(n = 39)
# colors = c(seq(0,0.05,length=20),seq(0.501,1,length=20)) #TODO change for certain graphs
colors=NULL

attach(mtcars)
par(mfrow=c(1,2))

heatmap.2(t(mat_data1),
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
  # dendrogram='none',
  density.info='histogram',
  denscol="black",
  keysize=1,
  #( "bottom.margin", "left.margin", "top.margin", "left.margin" )
  key.par=list(mar=c(3.5,0,3,0)),
  # lmat -- added 2 lattice sections (5 and 6) for padding
  lmat=rbind(c(5, 4, 2), c(6, 1, 3)), lhei=c(2.5, 5), lwid=c(1, 10, 1)
  )



heatmap.2(t(mat_data2),
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
  # dendrogram='none',
  density.info='histogram',
  denscol="black",
  keysize=1,
  #( "bottom.margin", "left.margin", "top.margin", "left.margin" )
  key.par=list(mar=c(3.5,0,3,0)),
  # lmat -- added 2 lattice sections (5 and 6) for padding
  lmat=rbind(c(5, 4, 2), c(6, 1, 3)), lhei=c(2.5, 5), lwid=c(1, 10, 1)
  )
