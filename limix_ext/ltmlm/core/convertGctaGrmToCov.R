# R script to read the GRM binary file
#bsub  -q priority -W 1:0  -o consoleTest.txt   R CMD BATCH '--args example.grm.bin example.ind exampleGCTA.cov' convertGctaGrmToCov.R

#read in file names
args=(commandArgs(T))
print(args)
BinFileName =  args[1]#"example.grm.bin"
indFile = args[2]# "example.ind"
indFile = read.table(indFile)
n = dim(indFile)[1]
outputGRM = args[3]#"exampleGCTA.cov"

GRM = matrix(0,nrow=n,ncol=n)
size = 4
BinFile = file(BinFileName, "rb");
GRM[ upper.tri(GRM,diag=T) ] = readBin(BinFile, n=n*(n+1)/2, what=numeric(0), size=size)
GRM = GRM + t(GRM) - diag(diag(GRM))
GRM = round(GRM, digits=10)
library(MASS)

write.matrix(GRM, outputGRM,",")
