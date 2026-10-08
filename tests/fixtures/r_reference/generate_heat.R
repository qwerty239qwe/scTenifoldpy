# Reference outputs of the heat-kernel tools of scTenifoldKnk (>= 2.0.0) for
# tests/test_heat.py
#
#   Rscript tests/fixtures/r_reference/generate_heat.R
#
# run from the repository root, after generate.R (it reads X_counts.csv.gz).
# heat_versions.txt records the versions used.

suppressPackageStartupMessages(library(scTenifoldKnk))
stopifnot(packageVersion("scTenifoldKnk") >= "2.0.0")

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

X <- as.matrix(read.csv(file.path(out_dir, "X_counts.csv.gz"), row.names = 1, check.names = FALSE))
qcX <- scQC(X, minLibSize = 30)
WT <- as.matrix(read.csv(file.path(out_dir, "knk_tensor_WT.csv.gz"), row.names = 1, check.names = FALSE))
kos <- list("ng10", c("ng10", "ng20"))

write_gz(heatKernel(WT, t = 10), "heat_kernel_WT.csv.gz")
write_gz(hkManifoldAlignment(WT, qcX, gKO = kos, t = 10), "heat_hk_distances.csv.gz")
D1 <- knockoutDirection(qcX, gKO = kos, genes = rownames(WT), regressLibSize = TRUE)
D0 <- knockoutDirection(qcX, gKO = kos, genes = rownames(WT))
rownames(D0) <- paste0(rownames(D0), "_noregress")
write_gz(rbind(D1, D0), "heat_direction.csv.gz")

# Pipeline: manifold alignment with direction (defaults), heat alignment, transcriptome-wide
knk <- scTenifoldKnk(countMatrix = X, gKO = "ng10", qc_minLibSize = 30, nCores = 1, seed = 1)
write_table_gz(knk$diffRegulation, "knk_direction_dregulation.csv.gz")
knkHeat <- scTenifoldKnk(countMatrix = X, gKO = "ng10", qc_minLibSize = 30, nCores = 1, seed = 1,
                         ma_method = "heat")
write_table_gz(knkHeat$diffRegulation, "knk_heat_dregulation.csv.gz")
tw <- scTenifoldKnk(countMatrix = X, gKO = c("ng10", "ng20"), transcriptomeWide = TRUE,
                    qc_minLibSize = 30, nCores = 1, seed = 1)
write_gz(tw$perturbationDistances, "tw_distances.csv.gz")
write_gz(tw$perturbationDirections, "tw_directions.csv.gz")

writeLines(c(R.version.string, paste("scTenifoldKnk", packageVersion("scTenifoldKnk"))),
           file.path(out_dir, "heat_versions.txt"))
