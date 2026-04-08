#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(splatter)
  library(SingleCellExperiment)
  library(zellkonverter)
})

# -----------------------------
# Fixed pilot spec (revised)
# -----------------------------
set.seed(42)

n_cells <- 4800L
n_genes <- 2000L
k_groups <- 4L
batch_cells <- c(2400L, 2400L)  # two balanced batches for pilot
group_prob <- rep(1 / k_groups, k_groups)

# HPC storage policy: write active dataset artifacts to WORK, not HOME repo paths.
work_root <- Sys.getenv("WORK", unset = "")
if (!nzchar(work_root)) {
  stop("WORK environment variable is not set. On FAU HPC, set WORK and re-run.")
}
out_h5ad <- file.path(
  work_root,
  "sc_autoencoder_project",
  "data",
  "processed",
  "splatter_k04.h5ad"
)
dir.create(dirname(out_h5ad), recursive = TRUE, showWarnings = FALSE)

# -----------------------------
# Simulate with Splatter
# -----------------------------
params <- newSplatParams(batchCells = batch_cells, nGenes = n_genes)
params <- setParams(
  params,
  de.prob = 0.20,
  de.facLoc = 0.35,
  de.facScale = 0.25,
  batch.facLoc = 0.01,
  batch.facScale = 0.02,
  lib.loc = 10.0,
  lib.scale = 0.25,
  out.prob = 0.01,
  out.facLoc = 2.0,
  out.facScale = 0.30,
  bcv.common = 0.08,
  dropout.type = "none"
)

sce <- splatSimulate(
  params = params,
  method = "groups",
  group.prob = group_prob,
  verbose = FALSE
)

# -----------------------------
# Enforce deterministic names
# -----------------------------
rownames(sce) <- sprintf("gene_%05d", seq_len(nrow(sce)))
colnames(sce) <- sprintf("cell_%05d", seq_len(ncol(sce)))

# -----------------------------
# Required obs fields
# -----------------------------
# cell_type (categorical, exactly 4 groups)
if (!("Group" %in% colnames(colData(sce)))) {
  stop("Splatter output missing colData$Group; cannot build required obs$cell_type")
}
cell_type <- factor(as.character(colData(sce)$Group))

# batch (categorical)
if ("Batch" %in% colnames(colData(sce))) {
  batch <- factor(as.character(colData(sce)$Batch))
} else {
  # fallback (should rarely happen)
  batch <- factor(rep(c("batch0", "batch1"), length.out = ncol(sce)))
}

colData(sce)$cell_type <- cell_type
colData(sce)$batch <- batch

# -----------------------------
# Matrix contract
# -----------------------------
# Raw counts for NB/ZINB/scVI in layers["counts"]
counts_mat <- as.matrix(counts(sce))
counts_mat[counts_mat < 0] <- 0
counts_mat <- round(counts_mat)

# Pilot-safe X content:
# use log1p library-size normalized expression
libsize <- colSums(counts_mat)
libsize[libsize == 0] <- 1
x_mat <- log1p(t(t(counts_mat) / libsize) * 1e4)

assay(sce, "counts") <- counts_mat
assay(sce, "X") <- x_mat

# Optional metadata (goes to uns-equivalent metadata)
metadata(sce)$simulation_source <- "splatter"
metadata(sce)$simulation_k <- 4L
metadata(sce)$simulation_seed <- 42L
metadata(sce)$simulation_note <- "phase3_pilot_k04_revised"

# -----------------------------
# Required validation checks before export
# -----------------------------
if (nlevels(colData(sce)$cell_type) != 4L) {
  stop(sprintf("Expected exactly 4 cell types, got %d", nlevels(colData(sce)$cell_type)))
}
if (any(table(colData(sce)$cell_type) == 0)) {
  stop("At least one cell_type level has zero cells")
}
if (!is.factor(colData(sce)$cell_type)) stop("cell_type must be factor")
if (!is.factor(colData(sce)$batch)) stop("batch must be factor")

if (!all(counts_mat >= 0)) stop("counts contains negative values")
if (!all(abs(counts_mat - round(counts_mat)) < 1e-8)) stop("counts not integer-like")

if (anyDuplicated(rownames(sce)) > 0) stop("Duplicate gene names detected")
if (anyNA(rownames(sce))) stop("NA gene names detected")
if (!all(nchar(rownames(sce)) > 0)) stop("Empty gene names detected")

if (!("counts" %in% assayNames(sce))) stop("Missing assay 'counts'")
if (!("X" %in% assayNames(sce))) stop("Missing assay 'X'")
if (!all(dim(assay(sce, "counts")) == dim(assay(sce, "X")))) {
  stop("Assays 'counts' and 'X' have mismatched dimensions")
}

# Put X first for broad writeH5AD compatibility
an <- assayNames(sce)
assays(sce) <- assays(sce)[c("X", setdiff(an, "X"))]

# Remove non-serializable metadata entries (notably Splatter's Params object)
# that can break AnnData export via zellkonverter.
simple_meta <- list()
for (k in names(metadata(sce))) {
  v <- metadata(sce)[[k]]
  if (is.null(v) || is.atomic(v) || is.character(v) || is.numeric(v) || is.logical(v)) {
    simple_meta[[k]] <- v
  }
}
metadata(sce) <- simple_meta

# -----------------------------
# Export to .h5ad
# -----------------------------
write_fun <- zellkonverter::writeH5AD
if ("X_name" %in% names(formals(write_fun))) {
  write_fun(sce, out_h5ad, X_name = "X")
} else {
  write_fun(sce, out_h5ad)
}

cat("Wrote:", out_h5ad, "\n")
cat("Done. Validate with repo inspector before pilot run.\n")
