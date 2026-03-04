#wd <- "/niddk-data-central/mae_hr/FVE"
wd <- "~/Projects/FVE"

#################################   PINV    ###################################   

load("~/Projects/FVE/run_output_local/_method1_dict_ds.RData")
method1_dict


load("~/Projects/FVE/run_output_local/PINV_method2_dict_ds.RData")
method2_dict

load("~/Projects/FVE/run_output_local/PINV_result_tbl_boot_ds.Rdata")
result_tbl_boot

range(reg_test_data)
range(reg_train_data %>% select(-nihtbx_cryst_uncorrected, -sex_2, -interview_age))


  
range(lin_reg_train_data$nihtbx_cryst_uncorrected)

################################# LR result ################################# 
load("~/Projects/FVE/LR_output/new/LR_result_tbl_boot100.Rdata") # table

result_tbl_boot


load("~/Projects/FVE/LR_output/new/LR_result_dict_boot100.Rdata")

result_dict_boot$Ridge_partial_tsa_test_r2

################################# LR coefficient map ################################# 

load("~/Projects/FVE/LR_output/LR_all_models_boot100.Rdata") # all the models
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

##################################V2


################################# PCA result ################################# 
library(tidyverse)
setwd("~/Projects/FVE")
reg_train_data_org_lin = read_csv("data_out/lin_reg_train_data.csv") #9080 rows

par(mfrow=c(1,1))
plot(reg_train_data_org_lin$total_surface_area, train_pca$x[,1],
     xlab="TSA", ylab="PC1")
plot(as.numeric(reg_train_data_org_lin$interview_age), train_pca$x[,1],
     xlab="age", ylab="PC1")



####### attempt 1: projected PCs
load("~/Projects/FVE/PCA_output/result_tbl_boot_50_1013.Rdata")

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
load("~/Projects/FVE/PCA_output/PCA_partial_train_result.Rdata")
load("~/Projects/FVE/PCA_output/PCA_partial_test_result.Rdata")
reg_data_org_partial = read_csv("data_out/reg_train_data_partial.csv")
reg_test_data_org_partial = read_csv("data_out/reg_test_data_partial.csv")


reg_train_data_partial = reg_train_data_org_partial
reg_test_data_partial = reg_test_data_org_partial

par(mfrow=c(1,1))
plot(train_pca_p$x[,1], train_pca_p$x[,2], xlim=c(-350,300), ylim=c(-150,150),
     xlab="PC1", ylab="PC2", main="partial_train")


####### attempt 2 separate PCs
load("~/Projects/FVE/PCA_output/result_tbl_boot_alt_50_1013.Rdata")
result_tbl_boot



############################################### Result summary plot
library(tidyverse)

df <- tribble(
  ~Model, ~Formulation, ~Variables, ~MeanFVE, ~SD,
  "Linear Regression", "full", "TSA", 0.1370, 0.0188,
  "Linear Regression", "partial", "TSA", 0.0745, 0.0140,
  "Linear Regression", "partial TSA", "TSA", 0, 0,
  
  "Mean per ROI + Lin. Regression", "full", "63 ROIs", 0.1453, 0.0168,
  "Mean per ROI + Lin. Regression", "partial", "63 ROIs", 0.0837, 0.0159,
  "Mean per ROI + Lin. Regression", "partial TSA", "63 ROIs", 0.0103, 0.0094,
  
  "PCA1 + Lin. Regression", "full", "PCs", 0.0561, 0.0108,
  "PCA1 + Lin. Regression", "partial", "PCs", 0.0687, 0.0131,
  "PCA1 + Lin. Regression", "partial TSA", "PCs", -0.0010, 0.0011,
  
  "PCA2 + Lin. Regression", "full", "PCs", 0.0558, 0.0109,
  "PCA2 + Lin. Regression", "partial", "PCs", 0.0685, 0.0132,
  "PCA2 + Lin. Regression", "partial TSA", "PCs", -0.0013, 0.0012,
  
  "PCA3 + Lin. Regression", "full", "PCs", 0.0865, 0.0162,
  "PCA3 + Lin. Regression", "partial", "PCs", 0.0964, 0.0179,
  "PCA3 + Lin. Regression", "partial TSA", "PCs", 0.0250, 0.0136,
  
  "PCA4 + Lin. Regression", "full", "PCs", 0.0860, 0.0194,
  "PCA4 + Lin. Regression", "partial", "PCs", 0.0924, 0.0187,
  "PCA4 + Lin. Regression", "partial TSA", "PCs", 0.0211, 0.0163,
  
  "PCA5 + Lin. Regression", "full", "PCs", 0.0611, 0.0274,
  "PCA5 + Lin. Regression", "partial", "PCs", -0.0108, 0.0276,
  "PCA5 + Lin. Regression", "partial TSA", "PCs", -0.0925, 0.0298,
  
  "LASSO", "full", "20,484 vertices", 0.1589, 0.0193,
  "LASSO", "partial", "20,484 vertices", 0.0934, 0.0195,
  "LASSO", "partial TSA", "20,484 vertices", -0.0101, 0.0195,
  
  "Ridge", "full", "20,484 vertices", 0.1272, 0.0162,
  "Ridge", "partial", "20,484 vertices", 0.1006, 0.0178,
  "Ridge", "partial TSA", "20,484 vertices", 0.0169, 0.0162,
  
  "Random Forest", "full", "20,484 vertices", 0.1315, 0.0101,
  "Random Forest", "partial", "20,484 vertices", 0.0940, 0.0078,
  "Random Forest", "partial TSA", "20,484 vertices", 0.0218, 0.0054,
  
  "GCN1", "full", "63 ROIs", 0.0569, 0.1285,
  "GCN1", "partial", "63 ROIs", 0.0326, 0.0391,
  "GCN1", "partial TSA", "63 ROIs", -0.0096, 0.0107
)

formulation_order <- c(
  "full",
  "partial",
  "partial TSA"
)

model_order <- c(
  "Linear Regression",
  "Mean per ROI + Lin. Regression",
  "PCA1 + Lin. Regression",
  "PCA2 + Lin. Regression",
  "PCA3 + Lin. Regression",
  "PCA4 + Lin. Regression",
  "PCA5 + Lin. Regression",
  "LASSO",
  "Ridge",
  "Random Forest",
  "GCN1"
)

df <- df %>%
  mutate(
    Formulation = factor(Formulation, 
                         levels = formulation_order,
                         labels = c("regular", "partial", "TSA partial")),
    Model = factor(Model, levels = model_order)
  )

ggplot(
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
  facet_wrap(~ Model, scales = "free_x") +
  labs(
    y = "Fraction of Variance Explained (Mean +/- 2 SD)",
    x = "Model formulation type"
  ) +
  theme_bw(base_size = 12) +
  theme(
    strip.text = element_text(face = "bold", size = 10),      
    axis.text.x = element_text(angle = 0, hjust = 0.5, size = 9),  
    axis.text.y = element_text(size = 9),                   
    axis.title = element_text(size = 8),                   
    panel.grid.major.x = element_blank()
  )









