
library(spc)

lamgrid<- seq(0.05, 0.3, length.out=101)

#cegrid<- seq(2.5, 4.0, length.out=101)
#cegrid<- seq(6.0, 12.0, length.out=101)
cegrid<- seq(5.0, 15.0, length.out=101)
 
z <- matrix(nrow=101, ncol=101)

j1 <- 0

for (lam in lamgrid) {
  j1 <- j1+1
  j2 <- 0
  for (ce in cegrid) {
    j2 <- j2+1
    z[j1,j2]<- mewma.arl(lam, ce, 3, delta=0)
  }
}

contour(lamgrid, cegrid, z, nlevels=5)

contour(lamgrid, cegrid, z, levels=c(100,200,300,400,500))


delta <- 0.1

j1 <- 0

for (lam in lamgrid) {
  j1 <- j1+1
  j2 <- 0
  for (ce in cegrid) {
    j2 <- j2+1
    z[j1,j2]<- mewma.arl(lam, ce, 3, delta=delta)
  }
}

contour(lamgrid, cegrid, z, nlevels=5)

print( summary( as.vector(z) ) )


delta <- 1

j1 <- 0

for (lam in lamgrid) {
  j1 <- j1+1
  j2 <- 0
  for (ce in cegrid) {
    j2 <- j2+1
    z[j1,j2]<- mewma.arl(lam, ce, 3, delta=delta)
  }
}

contour(lamgrid, cegrid, z, nlevels=5)

print( summary( as.vector(z) ) )

mewma.arl(0.05, 20, 3, delta=0, r=50)


rr <- 20:40
LL <- rep(NA, length(rr))
for ( i in 1:length(rr) ) LL[i] <- mewma.arl(0.05, 20, 3, delta=0.1, r=rr[i])

plot(rr, LL)

p <- 3

ARL0 <- 500

ARL1 <- Vectorize(function(x, r=30) {
  ce <- mewma.crit(x, ARL0, p, r=r)
  mewma.arl(x, ce, p, delta=delta, r=r)
})

delta <- 1

curve(ARL1, 0.05, 0.2)

lamOpt <- optimize(ARL1, c(0.05, 0.3), tol=1e-9)$minimum

abline(v=lamOpt, h=ARL1(lamOpt), lty=2, col="red")



delta <- 0.5

curve(ARL1, 0.05, 0.2)

lamOpt <- optimize(ARL1, c(0.05, 0.3), tol=1e-9)$minimum

abline(v=lamOpt, h=ARL1(lamOpt), lty=2, col="red")