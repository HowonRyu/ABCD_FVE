######################################################## 
# Generate HTML files with different views and capture screenshots
########################################################

library(R.matlab)
library(rgl)
library(htmltools)
library(webshot2)
library(grid)
library(gridExtra)
library(png)

# Load surface data
surfs <- readMat("data/SurfeView_surfaces.mat")

# Vertices
icsurfs_coords_lh <- surfs$surf.lh.pial[[1]][1:10242, ]
icsurfs_coords_rh <- surfs$surf.rh.pial[[1]][1:10242, ]

# Faces
icsurfs_faces_lh <- surfs$icsurfs[1,6][[1]][[1]][2][[1]]
icsurfs_faces_rh <- surfs$icsurfs[1,6][[1]][[1]][2][[1]] + 10242

# Coords and faces by combining left and right
coords <- rbind(icsurfs_coords_lh, icsurfs_coords_rh)
faces  <- rbind(icsurfs_faces_lh, icsurfs_faces_rh)

# Configuration
B <- 50
N_VERTICES_PER_HEMI <- 10242
N_VERTICES <- N_VERTICES_PER_HEMI * 2
N_COVARIATES <- 2
N_FEATURES_REGULAR <- N_VERTICES + N_COVARIATES
N_FEATURES_PARTIAL <- N_VERTICES

# Model configurations
model_configs <- list(
  "LASSO" = list(type = "regular", n_features = N_FEATURES_REGULAR),
  "LASSO_partial" = list(type = "partial", n_features = N_FEATURES_PARTIAL),
  "LASSO_partial_tsa" = list(type = "partial", n_features = N_FEATURES_PARTIAL),
  "Ridge" = list(type = "regular", n_features = N_FEATURES_REGULAR),
  "Ridge_partial" = list(type = "partial", n_features = N_FEATURES_PARTIAL),
  "Ridge_partial_tsa" = list(type = "partial", n_features = N_FEATURES_PARTIAL)
)

model_display_names <- list(
  "LASSO" = "LASSO",
  "LASSO_partial" = "LASSO Partial",
  "LASSO_partial_tsa" = "LASSO TSA Partial",
  "Ridge" = "Ridge",
  "Ridge_partial" = "Ridge Partial",
  "Ridge_partial_tsa" = "Ridge TSA Partial",
  "RF" = "RF",
  "RF_partial" = "RF Partial",
  "RF_partial_tsa" = "RF TSA Partial"
)

# Function to create HTML with specific view angle
create_brain_html <- function(vals, output_file, title, view_angle, ceiling_quantile = 0.90) {
  
  # Create colormap
  col_fun <- colorRampPalette(c("white", "darkgreen"))(1000)
  
  vmin <- 0
  p_ceiling <- quantile(vals, ceiling_quantile)
  vmax <- max(vals)
  
  # Create breaks and colors
  breaks <- seq(vmin, p_ceiling, length.out = length(col_fun) + 1)
  colors <- col_fun[pmin(cut(vals, breaks = breaks, include.lowest = TRUE, labels = FALSE), length(col_fun))]
  
  # Create 3D visualization
  open3d()
  shade3d(
    tmesh3d(vertices = t(coords), indices = t(faces), homogeneous = FALSE),
    color = colors,
    meshColor = "vertices"
  )
  
  # Set view angle
  if (view_angle == "front") {
    view3d(theta = 0, phi = 0, zoom = 0.7)
  } else if (view_angle == "left") {
    view3d(theta = -90, phi = 0, zoom = 0.7)
  } else if (view_angle == "right") {
    view3d(theta = 90, phi = 0, zoom = 0.7)
  } else if (view_angle == "back") {
    view3d(theta = 180, phi = 0, zoom = 0.7)
  }
  
  rgl_obj <- rglwidget(width = 600, height = 600)
  
  # Create HTML without title for cleaner screenshot
  html_output <- browsable(
    tagList(
      tags$div(
        style = "display:flex; flex-direction:row; align-items:center; justify-content:center;",
        rgl_obj
      )
    )
  )
  
  # Save to HTML file
  save_html(html_output, file = output_file)
  
  close3d()
}

######################################################## 
# Process LASSO models
########################################################

lr_output_dir <- "LR_output"
lasso_models <- c("LASSO", "LASSO_partial", "LASSO_partial_tsa")

# Store vals for colorbar
lasso_vals_list <- list()

for (model_type in lasso_models) {
  
  coef_file <- file.path(lr_output_dir, paste0("coefficients_", model_type, "_boot", B, ".csv"))
  
  if (!file.exists(coef_file)) {
    next
  }
  
  coef_data <- read.csv(coef_file)
  config <- model_configs[[model_type]]
  
  # Exclude age and sex for regular models
  if (config$type == "regular") {
    coef_data <- coef_data[1:N_VERTICES, ]
  }
  
  # Calculate mean absolute coefficient
  coef_cols <- coef_data[, grep("^b[0-9]+$", colnames(coef_data))]
  mean_coef_array <- rowMeans(abs(coef_cols))
  
  # Split into hemispheres
  lh_coefs <- mean_coef_array[1:N_VERTICES_PER_HEMI]
  rh_coefs <- mean_coef_array[(N_VERTICES_PER_HEMI + 1):N_VERTICES]
  vals <- c(lh_coefs, rh_coefs)
  
  # Store vals for colorbar
  lasso_vals_list[[model_type]] <- vals
  
  # Create HTML for each view
  for (view in c("front", "left", "right", "back")) {
    html_file <- file.path(lr_output_dir, "vis_output", 
                           paste0(model_type, "_brain_", view, ".html"))
    create_brain_html(vals, html_file, 
                      paste(model_display_names[[model_type]], "-", toupper(view)), 
                      view)
    
    # Capture screenshot
    png_file <- file.path(lr_output_dir, "vis_output", 
                          paste0(model_type, "_brain_", view, ".png"))
    webshot(html_file, png_file, vwidth = 800, vheight = 800, delay = 3)
  }
}

######################################################## 
# Process Ridge models
########################################################

ridge_models <- c("Ridge", "Ridge_partial", "Ridge_partial_tsa")

# Store vals for colorbar
ridge_vals_list <- list()

for (model_type in ridge_models) {
  
  coef_file <- file.path(lr_output_dir, paste0("coefficients_", model_type, "_boot", B, ".csv"))
  
  if (!file.exists(coef_file)) {
    next
  }
  
  coef_data <- read.csv(coef_file)
  config <- model_configs[[model_type]]
  
  # Exclude age and sex for regular models
  if (config$type == "regular") {
    coef_data <- coef_data[1:N_VERTICES, ]
  }
  
  # Calculate mean absolute coefficient
  coef_cols <- coef_data[, grep("^b[0-9]+$", colnames(coef_data))]
  mean_coef_array <- rowMeans(abs(coef_cols))
  
  # Split into hemispheres
  lh_coefs <- mean_coef_array[1:N_VERTICES_PER_HEMI]
  rh_coefs <- mean_coef_array[(N_VERTICES_PER_HEMI + 1):N_VERTICES]
  vals <- c(lh_coefs, rh_coefs)
  
  # Store vals for colorbar
  ridge_vals_list[[model_type]] <- vals
  
  # Create HTML for each view
  for (view in c("front", "left", "right", "back")) {
    html_file <- file.path(lr_output_dir, "vis_output", 
                           paste0(model_type, "_brain_", view, ".html"))
    create_brain_html(vals, html_file, 
                      paste(model_display_names[[model_type]], "-", toupper(view)), 
                      view)
    
    # Capture screenshot
    png_file <- file.path(lr_output_dir, "vis_output", 
                          paste0(model_type, "_brain_", view, ".png"))
    webshot(html_file, png_file, vwidth = 800, vheight = 800, delay = 3)
  }
}

######################################################## 
# Process RF models
########################################################

rf_output_dir <- "rf_output"

rf_models <- list(
  list(file = "rf_importance.csv", name = "RF"),
  list(file = "rf_partial_importance.csv", name = "RF_partial"),
  list(file = "rf_partial_tsa_importance.csv", name = "RF_partial_tsa")
)

# Store vals for colorbar
rf_vals_list <- list()

for (model in rf_models) {
  
  feature_importance_file <- file.path(rf_output_dir, model$file)
  
  if (!file.exists(feature_importance_file)) {
    next
  }
  
  # Load data
  feature_data <- read.csv(feature_importance_file)
  feature_data <- feature_data[, -1]
  
  # Exclude age and sex if present
  if (ncol(feature_data) == N_VERTICES + 2) {
    feature_data <- feature_data[, 1:N_VERTICES]
  }
  
  # Calculate mean importance
  mean_importance <- colMeans(feature_data)
  vals <- mean_importance
  
  # Store vals for colorbar
  rf_vals_list[[model$name]] <- vals
  
  # Create HTML for each view
  for (view in c("front", "left", "right", "back")) {
    html_file <- file.path(rf_output_dir, "vis_output", 
                           paste0(model$name, "_brain_", view, ".html"))
    create_brain_html(vals, html_file, 
                      paste(model_display_names[[model$name]], "-", toupper(view)), 
                      view)
    
    # Capture screenshot
    png_file <- file.path(rf_output_dir, "vis_output", 
                          paste0(model$name, "_brain_", view, ".png"))
    webshot(html_file, png_file, vwidth = 800, vheight = 800, delay = 3)
  }
}

######################################################## 
# Create combined plot tables
########################################################

create_plot_table <- function(model_family, model_names, output_dir, save_path, vals_list) {
  
  # Create a list to store grobs
  plot_list <- list()
  
  # Add column headers
  plot_list[[1]] <- textGrob("Model", gp = gpar(fontsize = 14, fontface = "bold"))
  plot_list[[2]] <- textGrob("Front", gp = gpar(fontsize = 14, fontface = "bold"))
  plot_list[[3]] <- textGrob("Left", gp = gpar(fontsize = 14, fontface = "bold"))
  plot_list[[4]] <- textGrob("Right", gp = gpar(fontsize = 14, fontface = "bold"))
  plot_list[[5]] <- textGrob("Bottom", gp = gpar(fontsize = 14, fontface = "bold"))
  plot_list[[6]] <- textGrob("Colorbar", gp = gpar(fontsize = 14, fontface = "bold"))
  
  # Add rows
  for (i in 1:length(model_names)) {
    model_name <- model_names[i]
    display_name <- model_display_names[[model_name]]
    
    # Model name
    plot_list[[length(plot_list) + 1]] <- textGrob(display_name, gp = gpar(fontsize = 12))
    
    # Load and add images
    for (view in c("front", "left", "right", "back")) {
      img_path <- file.path(output_dir, "vis_output", paste0(model_name, "_brain_", view, ".png"))
      
      cat("Looking for image:", img_path, "- Exists:", file.exists(img_path), "\n")
      
      if (file.exists(img_path)) {
        img <- readPNG(img_path)
        
        # Rotate left and right images 90 degrees
        if (view == "left") {
          # Rotate 90 degrees clockwise (left)
          img <- aperm(img, c(2, 1, 3))
          img <- img[nrow(img):1, , ]
        } else if (view == "right") {
          # Rotate 90 degrees counter-clockwise (right)
          img <- aperm(img, c(2, 1, 3))
          img <- img[, ncol(img):1, ]
        }
        
        plot_list[[length(plot_list) + 1]] <- rasterGrob(img, interpolate = TRUE)
      } else {
        plot_list[[length(plot_list) + 1]] <- textGrob("N/A")
      }
    }
    
    # Add colorbar for this row
    vals <- vals_list[[model_name]]
    if (!is.null(vals)) {
      vmin <- min(vals)
      vmax <- max(vals)
      p75 <- quantile(vals, 0.75)
      p50 <- quantile(vals, 0.50)
      p25 <- quantile(vals, 0.25)
      
      # Create colorbar grob (smaller)
      gradient_colors <- colorRampPalette(c("white", "darkgreen"))(100)
      y_coords <- seq(0.1, 0.9, length.out = 100)
      
      colorbar_plot <- gTree(children = gList(
        rectGrob(x = 0.15, y = y_coords, width = 0.05, height = 0.8/100,
                 gp = gpar(fill = gradient_colors, col = NA)),
        textGrob(sprintf("%.4f", vmax), x = 0.22, y = 0.90, just = "left", gp = gpar(fontsize = 7)),
        textGrob(sprintf("%.4f", p75), x = 0.22, y = 0.75, just = "left", gp = gpar(fontsize = 6)),
        textGrob(sprintf("%.4f", p50), x = 0.22, y = 0.50, just = "left", gp = gpar(fontsize = 6)),
        textGrob(sprintf("%.4f", p25), x = 0.22, y = 0.25, just = "left", gp = gpar(fontsize = 6)),
        textGrob(sprintf("%.4f", vmin), x = 0.22, y = 0.10, just = "left", gp = gpar(fontsize = 7))
      ))
      
      plot_list[[length(plot_list) + 1]] <- colorbar_plot
    } else {
      plot_list[[length(plot_list) + 1]] <- textGrob("N/A")
    }
  }
  
  # Arrange in grid
  png(save_path, width = 2800, height = 850 * length(model_names), res = 150)
  grid.arrange(grobs = plot_list, ncol = 6, 
               widths = c(1, 2, 2, 2, 2, 1.2),
               top = textGrob(paste(model_family, "Models - Brain Visualizations"), 
                              gp = gpar(fontsize = 16, fontface = "bold")))
  dev.off()
}

# Create LASSO table
create_plot_table("LASSO", 
                  c("LASSO", "LASSO_partial", "LASSO_partial_tsa"), 
                  lr_output_dir,
                  file.path(lr_output_dir, "vis_output", "LASSO_brain_views_table.png"),
                  lasso_vals_list)

# Create Ridge table
create_plot_table("Ridge", 
                  c("Ridge", "Ridge_partial", "Ridge_partial_tsa"), 
                  lr_output_dir,
                  file.path(lr_output_dir, "vis_output", "Ridge_brain_views_table.png"),
                  ridge_vals_list)

# Create RF table
create_plot_table("Random Forest", 
                  c("RF", "RF_partial", "RF_partial_tsa"), 
                  rf_output_dir,
                  file.path(rf_output_dir, "vis_output", "RF_brain_views_table.png"),
                  rf_vals_list)
