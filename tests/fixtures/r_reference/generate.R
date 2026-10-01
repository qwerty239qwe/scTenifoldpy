# Reference outputs of the R packages for tests/test_r_parity_pipeline.py
#
# Regenerate after a new release of scTenifoldNet or scTenifoldKnk with
#
#   Rscript tests/fixtures/r_reference/generate.R
#
# run from the repository root, and commit the updated files. The script
# writes the input counts too, so the tests do not need R. versions.txt
# records the versions used; update the versions checked in
# test_fixture_versions (tests/test_r_parity_pipeline.py) to match.
#
# The data are the example of the R READMEs (100 genes x 2,000 cells).
# Y changes three genes of X, but not by copying other genes as the README
# does: exactly duplicated genes give tied edge weights on the quantile
# threshold, and whether those are kept depends on the last bit of the
# floating-point arithmetic.

suppressPackageStartupMessages({
  library(scTenifoldNet)
  library(scTenifoldKnk)
})

out_dir <- "tests/fixtures/r_reference"
if (!dir.exists(out_dir)) stop("run this script from the repository root")

write_gz <- function(x, name) {
  x <- as.matrix(x)
  values <- matrix(sprintf("%.17g", x), nrow = nrow(x))
  out <- cbind(rownames(x), values)
  colnames(out) <- c("", colnames(x))
  con <- gzfile(file.path(out_dir, name), "w")
  write.table(out, con, sep = ",", quote = FALSE, row.names = FALSE)
  close(con)
}

write_table_gz <- function(df, name) {
  num <- vapply(df, is.numeric, logical(1))
  df[num] <- lapply(df[num], sprintf, fmt = "%.17g")
  con <- gzfile(file.path(out_dir, name), "w")
  write.csv(df, con, row.names = FALSE, quote = FALSE)
  close(con)
}

# Input data
nCells <- 2000
nGenes <- 100
set.seed(1)
X <- matrix(rnbinom(n = nGenes * nCells, size = 20, prob = 0.98), ncol = nCells)
rownames(X) <- c(paste0("ng", 1:90), paste0("mt-", 1:10))
colnames(X) <- paste0("cell", seq_len(nCells))
set.seed(2)
Y <- X
Y[10, ] <- Y[50, ] + rpois(nCells, 1)
Y[2, ] <- Y[11, ] + rpois(nCells, 1)
Y[3, ] <- rnbinom(nCells, size = 20, prob = 0.95)
write.csv(X, gzfile(file.path(out_dir, "X_counts.csv.gz")), quote = FALSE)
write.csv(Y, gzfile(file.path(out_dir, "Y_counts.csv.gz")), quote = FALSE)

# QC
qcX <- scQC(X, minLibSize = 30)
writeLines(colnames(qcX), gzfile(file.path(out_dir, "X_qc_cells.txt.gz")))

# scTenifoldNet with its defaults except the library size filter
net <- scTenifoldNet(X = X, Y = Y, qc_minLibSize = 30, nCores = 1, seed = 1)
write_gz(net$tensorNetworks$X, "net_tensor_X.csv.gz")
write_gz(net$tensorNetworks$Y, "net_tensor_Y.csv.gz")
write_gz(net$manifoldAlignment, "net_manifold.csv.gz")
write_table_gz(net$diffRegulation, "net_dregulation.csv.gz")

# scTenifoldKnk with its defaults except the library size filter
knk <- scTenifoldKnk(countMatrix = X, gKO = "ng10", qc_minLibSize = 30, nCores = 1, seed = 1)
write_gz(knk$tensorNetworks$WT, "knk_tensor_WT.csv.gz")
write_gz(knk$manifoldAlignment, "knk_manifold.csv.gz")
write_table_gz(knk$diffRegulation, "knk_dregulation.csv.gz")

# The same with two genes knocked out together
knk2 <- scTenifoldKnk(countMatrix = X, gKO = c("ng10", "ng20"), qc_minLibSize = 30, nCores = 1, seed = 1)
write_gz(knk2$manifoldAlignment, "knk_multi_manifold.csv.gz")
write_table_gz(knk2$diffRegulation, "knk_multi_dregulation.csv.gz")

writeLines(c(
  R.version.string,
  paste("scTenifoldNet", packageVersion("scTenifoldNet")),
  paste("scTenifoldKnk", packageVersion("scTenifoldKnk")),
  paste("MASS", packageVersion("MASS"))
), file.path(out_dir, "versions.txt"))
