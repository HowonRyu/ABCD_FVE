wd <- "."  # Set to your FVE project directory if not running from it



################################################################################
################################# LR coefficient map ###########################
################################################################################
load(file.path(wd, "LR_output/LR_all_models_boot100.Rdata")) # all the models
all_models


##### save the important features out of 20484 vertices
library(glmnet)

B_selected <- 50
model_types <- c('LASSO', 'LASSO_partial', 'LASSO_partial_tsa',
                 'Ridge', 'Ridge_partial', 'Ridge_partial_tsa')

# Extract all coefficients
all_coefficients <- list()

for (model_type in model_types) {
  cat(paste0("Processing ", model_type, "...\n"))
  
  coef_list <- list()
  
  for (b in 1:B_selected) {
    model_name <- paste0(model_type, "_b", b)
    
    if (model_name %in% names(all_models)) {
      selected_model = all_models[[model_name]]
      
      coefs <- as.matrix(selected_model$beta)
      coef_list[[paste0("b", b)]] <- coefs
      
      if (b %% 10 == 0) {
        cat(paste0("  Processed ", b, "/", B, " iterations\n"))
      }
    }
  }
  
  all_coefficients[[model_type]] <- coef_list
}



# Save coefficients as CSV files
for (model_type in model_types) {
  cat(paste0("Saving ", model_type, " coefficients...\n"))
  
  coef_data <- all_coefficients[[model_type]]
  
  # Convert list to matrix
  coef_matrix <- do.call(cbind, coef_data)
  colnames(coef_matrix) <- paste0("b", 1:B_selected)
  
  # Create feature names
  if (model_type %in% c('LASSO', 'Ridge')) {
    # Regular models: 20484 vertices + age + sex
    feature_names <- c(paste0(0:10241, "_l"),paste0(0:10241, "_r"), "age", "sex")
  } else {
    # Partial models: 20484 vertices only
    feature_names <- c(paste0(0:10241, "_l"),paste0(0:10241, "_r"))
  }

  # Create dataframe
  coef_df <- data.frame(feature = feature_names, coef_matrix)
  
  # Save as CSV
  write.csv(coef_df, 
            file = paste0(wd, "/LR_output/coefficients_", model_type, "_boot", B, ".csv"),
            row.names = FALSE)
}









#### LASSO/Ridge Coefficient 
all_models$LASSO_b1$beta   #age 0.22 sex 1.72
all_models$Ridge_b1$beta  #age 0.06 sex 0.316

all_models$LASSO_partial_b1$beta   #age 0.22 sex 1.6
all_models$Ridge_partial_b1$beta  #age 0.05 sex 0.282



#### LASSO/Ridge Coefficient map visualization
library(R.matlab)
library(rgl)
surfs <- readMat("data/SurfeView_surfaces.mat")



# Vertices
icsurfs_coords_lh <- surfs$surf.lh.pial[[1]][1:10242, ]
icsurfs_coords_rh <- surfs$surf.rh.pial[[1]][1:10242, ]


# faces
icsurfs_faces_lh <- surfs$icsurfs[1,6][[1]][[1]][2][[1]]
icsurfs_faces_rh <- surfs$icsurfs[1,6][[1]][[1]][2][[1]] + 10242



# Coords and faces by combining left and right
coords = rbind(icsurfs_coords_lh, icsurfs_coords_rh)
faces  = rbind(icsurfs_faces_lh, icsurfs_faces_rh)

dim(coords)
dim(faces)




# mean across bootstrap samples
use_mean = FALSE

if (use_mean == TRUE) {
  ridge_coefs <- sapply(1:100, function(b) {
    as.numeric(all_models[[paste0("Ridge_b", b)]]$beta)
  })
  ridge_mean <- rowMeans(ridge_coefs)
  
  lasso_coefs <- sapply(1:100, function(b) {
    as.numeric(all_models[[paste0("LASSO_b", b)]]$beta)
  })
  lasso_mean <- rowMeans(lasso_coefs)
  
  # if more than 50% 0, want clean 0
  is_zero <- (lasso_coefs == 0)
  zero_majority <- rowSums(is_zero) > (ncol(lasso_coefs) / 2)
  lasso_mean[zero_majority] <- 0
} else {
  ridge_mean = all_models$Ridge_partial_b1$beta
  lasso_mean = all_models$LASSO_partial_b1$beta
}



# Visualization
library(rgl)
library(fields)
library(htmltools)

model = "LASSO"
vectors = list()
vectors[['LASSO']] = as.vector(lasso_mean)[1:20484]
vectors[['Ridge']] = as.vector(ridge_mean)[1:20484]


vals = vectors[[model]]
plot(1:20484, vals, type="l", xlab="vertex", ylab="coefficient", main=model)


# Get data range
vmin <- min(vals)
vmax <- max(vals)
max_abs <- max(abs(vmin), abs(vmax))


(thresh <- quantile(abs(vals), 0.99))
col_fun <- colorRampPalette(c("blue", "white", "red"))(1000)
vals_clamped <- pmin(pmax(vals, -thresh), thresh)
breaks <- seq(-thresh, thresh, length.out = length(col_fun) + 1)
colors <- col_fun[cut(vals_clamped, breaks = breaks, include.lowest = TRUE, labels = FALSE)]


# Creat shade3d
open3d()
shade3d(
  tmesh3d(vertices = t(coords), indices = t(faces), homogeneous = FALSE),
  color = colors,
  meshColor = "vertices"
)
rgl_obj <- rglwidget(width = 600, height = 600)

# --- Generate colorbar (with visible ticks) ---
dir.create("LR_output", showWarnings = FALSE)
colorbar_file <- paste0("LR_output/", model, "_coeffs.png")
png(colorbar_file, width = 200, height = 700, bg = "transparent", res = 200)

# Margins: bottom, left, top, right
# Top margin increased to 6 for "Coefficient"
# Right margin (6) ensures tick labels are visible
par(mar = c(3, 4, 6, 6), mgp = c(2, 0.8, 0))

plot.new()
fields::image.plot(
  legend.only = TRUE,
  zlim = c(-thresh, thresh),
  col = col_fun,
  horizontal = FALSE,
  legend.width = 1.6,
  legend.mar = 5,   # space between bar and tick labels
  axis.args = list(
    at = seq(-thresh, thresh, length.out = 5),
    labels = sprintf("%.2f", seq(-thresh, thresh, length.out = 5)),
    cex.axis = 1
  ),
  legend.args = list(
    text = "beta",
    side = 3,
    line = 2.5,     # move the title farther away from the colorbar
    cex = 1.1
  )
)

dev.off()

# Combine with 3D plot in Viewer
browsable(
  tagList(
    tags$div(
      style = "display:flex; flex-direction:row; align-items:center; gap:8px;",
      rgl_obj,
      tags$img(src = colorbar_file, style = "height:450px;")
    )
  )
)



################################################################################
################################# PCA result ###################################
################################################################################

library(tidyverse)
setwd(wd)
reg_train_data_org_lin = read_csv("data_out/lin_reg_train_data.csv") #9080 rows

par(mfrow=c(1,1))
plot(reg_train_data_org_lin$total_surface_area, train_pca$x[,1],
     xlab="TSA", ylab="PC1")
plot(as.numeric(reg_train_data_org_lin$interview_age), train_pca$x[,1],
     xlab="age", ylab="PC1")



####### attempt 1: projected PCs
load(file.path(wd, "PCA_output/result_tbl_boot_50_1013.Rdata"))

result_tbl_boot



# why is test R2 higher than train?
par(mfrow=c(3,1))
plot(result_dict_boot$partial_test_r2, ylim = c(0,0.5), type="l", col="black", lty=4,
     main="regular", ylab="R2")
lines(result_dict_boot$partial_train_r2, col="black")
legend("top", legend=c("train", "test"), col=c("black", "black"), lty=c(1, 4))

plot(result_dict_boot$reg_test_r2, ylim = c(0,0.5), type="l", col="red", lty=4,
     main="partial", ylab="R2")
lines(result_dict_boot$reg_train_r2, col="red")
legend("top", legend=c("train", "test"), col=c("red", "red"), lty=c(1, 4))

plot(result_dict_boot$partial_tsa_test_r2, ylim = c(0,0.5), type="l", col="blue", lty=4,
     main="partial_tsa", ylab="R2")
lines(result_dict_boot$partial_tsa_train_r2, col="blue")
legend("top", legend=c("train", "test"), col=c("blue", "blue"), lty=c(1, 4))




## individual look
load(file.path(wd, "PCA_output/PCA_partial_train_result.Rdata"))
load(file.path(wd, "PCA_output/PCA_partial_test_result.Rdata"))
reg_data_org_partial = read_csv("data_out/reg_train_data_partial.csv")
reg_test_data_org_partial = read_csv("data_out/reg_test_data_partial.csv")


reg_train_data_partial = reg_train_data_org_partial
reg_test_data_partial = reg_test_data_org_partial

par(mfrow=c(1,1))
plot(train_pca_p$x[,1], train_pca_p$x[,2], xlim=c(-350,300), ylim=c(-150,150),
     xlab="PC1", ylab="PC2", main="partial_train")


####### attempt 2 separate PCs
load(file.path(wd, "PCA_output/result_tbl_boot_alt_50_1013.Rdata"))
result_tbl_boot


################################################################################
################################### Result summary plot ########################
################################################################################
df <- tribble(
  ~Model, ~Formulation, ~Variables, ~MeanFVE, ~SD,
  "TSA Linear Regression", "full", "TSA", 0.1370, 0.0188,
  "TSA Linear Regression", "partial", "TSA", 0.0745, 0.0140,
  "TSA Linear Regression", "partial TSA", "TSA", 0, 0,
  
  "TSA Quadratic Regression", "full", "TSA + TSA²", 0.1397, 0.0192,
  "TSA Quadratic Regression", "partial", "TSA + TSA²", 0.0772, 0.0140,
  "TSA Quadratic Regression", "partial TSA", "TSA + TSA²", 0, 0,
  
  "Mean per ROI Regression", "full", "63 ROIs", 0.1433, 0.0161,
  "Mean per ROI Regression", "partial", "63 ROIs", 0.0796, 0.0163,
  "Mean per ROI Regression", "partial TSA", "63 ROIs", 0.0103, 0.0093,
  
  "PCR (2 PCs)", "full", "PCs", 0.0558, 0.0109,
  "PCR (2 PCs)", "partial", "PCs", 0.0685, 0.0132,
  "PCR (2 PCs)", "partial TSA", "PCs", -0.0013, 0.0012,
  
  "PCR (102 PCs)", "full", "PCs", 0.0865, 0.0162,
  "PCR (102 PCs)", "partial", "PCs", 0.0964, 0.0179,
  "PCR (102 PCs)", "partial TSA", "PCs", 0.0250, 0.0136,
  
  "PCR (205 PCs)", "full", "PCs", 0.0860, 0.0194,
  "PCR (205 PCs)", "partial", "PCs", 0.0924, 0.0187,
  "PCR (205 PCs)", "partial TSA", "PCs", 0.0211, 0.0163,
  
  "PCR (1024 PCs)", "full", "PCs", 0.0611, 0.0274,
  "PCR (1024 PCs)", "partial", "PCs", -0.0108, 0.0276,
  "PCR (1024 PCs)", "partial TSA", "PCs", -0.0925, 0.0298,
  
  "LASSO", "full", "20,484 vertices", 0.1589, 0.0193,
  "LASSO", "partial", "20,484 vertices", 0.0934, 0.0195,
  "LASSO", "partial TSA", "20,484 vertices", -0.0101, 0.0195,
  
  "Ridge", "full", "20,484 vertices", 0.1272, 0.0162,
  "Ridge", "partial", "20,484 vertices", 0.1006, 0.0178,
  "Ridge", "partial TSA", "20,484 vertices", 0.0169, 0.0162,
  
  "Random Forest", "full", "20,484 vertices", 0.1281, 0.0087,
  "Random Forest", "partial", "20,484 vertices", 0.0920, 0.0079,
  "Random Forest", "partial TSA", "20,484 vertices", 0.0204, 0.0055,
  
  "GCN", "full", "63 ROIs", 0.0569, 0.1285,
  "GCN", "partial", "63 ROIs", 0.0326, 0.0391,
  "GCN", "partial TSA", "63 ROIs", -0.0096, 0.0107
)

formulation_order <- c(
  "full",
  "partial",
  "partial TSA"
)

model_order <- c(
  "TSA Linear Regression",
  "TSA Quadratic Regression",
  "Mean per ROI Regression",
  "PCR (2 PCs)",
  "PCR (102 PCs)",
  "PCR (205 PCs)",
  "PCR (1024 PCs)",
  "LASSO",
  "Ridge",
  "Random Forest",
  "GCN"
)

df <- df %>%
  mutate(
    Formulation = factor(Formulation, 
                         levels = formulation_order,
                         labels = c("full", "partial", "TSA partial")),
    Model = factor(Model, levels = model_order)
  )

plot_sub = ggplot(
  df,
  aes(
    x = Formulation,
    y = MeanFVE,
    ymin = MeanFVE - 2*SD,
    ymax = MeanFVE + 2*SD
  )
) +
  geom_pointrange(
    linewidth = 0.8,
    size = 0.1  
  ) +
  geom_errorbar(
    linewidth = 0.8,
    width = 0.1
  ) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "gray50") +
  facet_wrap(~ Model, nrow = 3, ncol = 4) +
  labs(
    y = "Fraction of Variance Explained (Mean +/- 2 SD)",
    x = ""
  ) +
  theme_bw(base_size = 12) +
  theme(
    strip.text = element_text(face = "bold", size = 10),      
    axis.text.x = element_text(angle = 0, hjust = 0.5, size = 10),  
    axis.text.y = element_text(size = 12),                   
    axis.title.y = element_text(size = 14),  
    #axis.title.x = element_text(size = 14),   
    panel.grid.major.x = element_blank(),
    plot.margin = margin(t = 5, r = 5, b = 10, l = 5, unit = "pt")
  )
plot_sub

ggsave("manuscript_plots/summary_fig.png", plot = plot_sub, units="in", width = 10, height = 6.5, dpi = 600)


