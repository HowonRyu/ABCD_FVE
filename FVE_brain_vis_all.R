################################################################################
# Unified Brain Visualization Pipeline
################################################################################

library(R.matlab)
library(rgl)
library(htmltools)
library(ggplot2)
library(webshot2)
library(grid)
library(gridExtra)
library(png)
library(readr)
library(gifti)
wd = "."  # Set to your FVE project directory if not running from it
setwd(wd)
source("FVE_brain_vis_util.R")

# Load icsurfs
surfs             <- readMat("data/SurfeView_surfaces.mat")
icsurfs_coords_lh <- surfs$surf.lh.pial[[1]][1:10242, ]
icsurfs_coords_rh <- surfs$surf.rh.pial[[1]][1:10242, ]
icsurfs_faces_lh  <- surfs$icsurfs[1,6][[1]][[1]][2][[1]]
icsurfs_faces_rh  <- surfs$icsurfs[1,6][[1]][[1]][2][[1]] + 10242
coords            <- rbind(icsurfs_coords_lh, icsurfs_coords_rh)
faces             <- rbind(icsurfs_faces_lh,  icsurfs_faces_rh)

# Load FreeSurfer surfaces
surf_lh      <- read_gifti("data_out/fs_LR.10k.L.pial.surf.gii")
surf_rh      <- read_gifti("data_out/fs_LR.10k.R.pial.surf.gii")
coords_lh_fs <- surf_lh$data$pointset
coords_rh_fs <- surf_rh$data$pointset
faces_lh_fs  <- surf_lh$data$triangle + 1
faces_rh_fs  <- surf_rh$data$triangle + 1 + 10242
coords_fs    <- rbind(coords_lh_fs, coords_rh_fs)
faces_fs     <- rbind(faces_lh_fs,  faces_rh_fs)

# Config
B                   <- 50
N_VERTICES_PER_HEMI <- 10242
N_VERTICES          <- N_VERTICES_PER_HEMI * 2
COLOR_GRADIENT      <- c("grey70", "grey60", "darkred", "red", "orange", "yellow2", "yellow")
COLUMN_WIDTHS       <- c(1.5, 2, 2, 2, 2, 1.2)

################################################################################
# norm_method: "minmax", "quantile", or "normal"
norm_method <- "minmax"
################################################################################

all_display_names <- list(
  "LASSO"                  = "LASSO",
  "LASSO_partial"          = "LASSO Partial",
  "LASSO_partial_tsa"      = "LASSO TSA Partial",
  "Ridge"                  = "Ridge",
  "Ridge_partial"          = "Ridge Partial",
  "Ridge_partial_tsa"      = "Ridge TSA Partial",
  "RF"                     = "RF",
  "RF_partial"             = "RF Partial",
  "RF_partial_tsa"         = "RF TSA Partial",
  "mapping_regular"        = "Mean per ROI",
  "mapping_partial"        = "Mean per ROI Partial",
  "mapping_partial_tsa"    = "Mean per ROI TSA Partial",
  "mapping_fs_regular"     = "Mean per ROI (FS)",
  "mapping_fs_partial"     = "Mean per ROI Partial (FS)",
  "mapping_fs_partial_tsa" = "Mean per ROI TSA Partial (FS)",
  "PCA_2_regular"          = "PCA (2 PCs)",
  "PCA_2_partial"          = "PCA (2 PCs) Partial",
  "PCA_2_partial_tsa"      = "PCA (2 PCs) TSA Partial",
  "PCA_102_regular"        = "PCA (102 PCs)",
  "PCA_102_partial"        = "PCA (102 PCs) Partial",
  "PCA_102_partial_tsa"    = "PCA (102 PCs) TSA Partial",
  "PCA_205_regular"        = "PCA (205 PCs)",
  "PCA_205_partial"        = "PCA (205 PCs) Partial",
  "PCA_205_partial_tsa"    = "PCA (205 PCs) TSA Partial",
  "PCA_1024_regular"       = "PCA (1024 PCs)",
  "PCA_1024_partial"       = "PCA (1024 PCs) Partial",
  "PCA_1024_partial_tsa"   = "PCA (1024 PCs) TSA Partial"
)



################################################################################
# Distribution plots
################################################################################

plot_coefficient_distributions("Full",        1, comparison_models, norm_method)
plot_coefficient_distributions("Partial",     2, comparison_models, norm_method)
plot_coefficient_distributions("TSA Partial", 3, comparison_models, norm_method)



################################################################################
# Brain visualization per model family
################################################################################

visualize_brain_models(
  model_family        = "LASSO",
  model_names         = c("LASSO", "LASSO_partial", "LASSO_partial_tsa"),
  model_display_names = all_display_names,
  output_dir          = "LR_output",
  table_filename      = "LASSO_brain_views_table.png",
  get_vals_function   = get_lr_vals,
  norm_method         = norm_method
)

visualize_brain_models(
  model_family        = "Ridge",
  model_names         = c("Ridge", "Ridge_partial", "Ridge_partial_tsa"),
  model_display_names = all_display_names,
  output_dir          = "LR_output",
  table_filename      = "Ridge_brain_views_table.png",
  get_vals_function   = get_lr_vals,
  norm_method         = norm_method
)

visualize_brain_models(
  model_family        = "Random Forest",
  model_names         = c("RF", "RF_partial", "RF_partial_tsa"),
  model_display_names = all_display_names,
  output_dir          = "rf_output",
  table_filename      = "RF_brain_views_table.png",
  get_vals_function   = get_rf_vals,
  norm_method         = norm_method
)

visualize_brain_models(
  model_family        = "Mean per ROI (FreeSurfer)",
  model_names         = c("mapping_fs_regular", "mapping_fs_partial", "mapping_fs_partial_tsa"),
  model_display_names = all_display_names,
  output_dir          = "MeanROI_output",
  table_filename      = "MeanPerROI_FS_brain_views_table.png",
  get_vals_function   = get_mapping_fs_vals,
  use_freesurfer      = TRUE,
  norm_method         = norm_method
)

#visualize_brain_models(
#  model_family        = "PCA (2 PCs)",
#  model_names         = c("PCA_2_regular", "PCA_2_partial", "PCA_2_partial_tsa"),
#  model_display_names = all_display_names,
#  output_dir          = "PCA_output",
#  table_filename      = "PCA_2_brain_views_table.png",
#  get_vals_function   = get_pca_vals,
#  norm_method         = norm_method
#)

#visualize_brain_models(
#  model_family        = "PCA (102 PCs)",
#  model_names         = c("PCA_102_regular", "PCA_102_partial", "PCA_102_partial_tsa"),
#  model_display_names = all_display_names,
#  output_dir          = "PCA_output",
#  table_filename      = "PCA_102_brain_views_table.png",
#  get_vals_function   = get_pca_vals,
#  norm_method         = norm_method
#)

# visualize_brain_models(
#   model_family        = "PCA (205 PCs)",
#   model_names         = c("PCA_205_regular", "PCA_205_partial", "PCA_205_partial_tsa"),
#   model_display_names = all_display_names,
#   output_dir          = "PCA_output",
#   table_filename      = "PCA_205_brain_views_table.png",
#   get_vals_function   = get_pca_vals,
#   norm_method         = norm_method
# )

#visualize_brain_models(
#  model_family        = "PCA (1024 PCs)",
#  model_names         = c("PCA_1024_regular", "PCA_1024_partial", "PCA_1024_partial_tsa"),
#  model_display_names = all_display_names,
#  output_dir          = "PCA_output",
#  table_filename      = "PCA_1024_brain_views_table.png",
#  get_vals_function   = get_pca_vals,
#  norm_method         = norm_method
#)

################################################################################
# Cross-model comparison
################################################################################

comparison_models <- list(
  "Mean per ROI" = list(
    model_names       = c("mapping_fs_regular", "mapping_fs_partial", "mapping_fs_partial_tsa"),
    output_dir        = "MeanROI_output",
    get_vals_function = get_mapping_fs_vals
  ),
  "LASSO" = list(
    model_names       = c("LASSO", "LASSO_partial", "LASSO_partial_tsa"),
    output_dir        = "LR_output",
    get_vals_function = get_lr_vals
  ),
  "Ridge" = list(
    model_names       = c("Ridge", "Ridge_partial", "Ridge_partial_tsa"),
    output_dir        = "LR_output",
    get_vals_function = get_lr_vals
  ), 
  "Random Forest" = list(
    model_names       = c("RF", "RF_partial", "RF_partial_tsa"),
    output_dir        = "rf_output",
    get_vals_function = get_rf_vals
  )
)

create_cross_model_comparison("Full",        1, comparison_models, norm_method)
create_cross_model_comparison("Partial",     2, comparison_models, norm_method)
create_cross_model_comparison("TSA Partial", 3, comparison_models, norm_method)


################################################################################
# Bootstrap correlation analysis 
################################################################################

#corr_results_full    <- compute_bootstrap_correlations(comparison_models, formulation_index = 1)
#corr_results_partial <- compute_bootstrap_correlations(comparison_models, formulation_index = 2)
#corr_results_tsa     <- compute_bootstrap_correlations(comparison_models, formulation_index = 3)

plot_bootstrap_correlations(corr_results_full,    norm_method, formulation_type = "Full")
plot_bootstrap_correlations(corr_results_partial, norm_method, formulation_type = "Partial")
plot_bootstrap_correlations(corr_results_tsa,     norm_method, formulation_type = "TSA Partial")


