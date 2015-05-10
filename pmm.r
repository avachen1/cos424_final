library(flexmix)

setwd('~/Documents/PrincetonJuniorSpring//COS424/HW/Final Project/')


class <- FLXPmultinom()
datatable <- read.table('train.txt')

formula = as.formula(paste("y ~ ", paste(colnames(datatable), collapse= "+")))

FLXMRglm(formula = formula , family='poisson')
model <- flexmix(formula = formula, data = datatable, k = 20, model = FLXMRglm(formula = formula, family='poisson'), )


class <- FLXPmultinom(~ AGE + ACADMOS + MINORDRG + LOGSPEND)


