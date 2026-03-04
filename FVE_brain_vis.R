######################################################## L/R Feature importance visualization

#### LASSO/Ridge Coefficient map visualization
library(R.matlab)
library(rgl)
library(fields)
library(htmltools)
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


# Configuration
B <- 50
output_dir <- "LR_output"
model_types <- c("LASSO", "LASSO_partial", "LASSO_partial_tsa",
                 "Ridge", "Ridge_partial", "Ridge_partial_tsa")

# Display names for titles
model_display_names <- list(
  "LASSO" = "LASSO",
  "LASSO_partial" = "LASSO partial",
  "LASSO_partial_tsa" = "LASSO TSA partial",
  "Ridge" = "Ridge",
  "Ridge_partial" = "Ridge partial",
  "Ridge_partial_tsa" = "Ridge TSA partial"
)

for (model_type in model_types) {
  
  cat("\n==================================================\n")
  cat("Processing:", model_type, "\n")
  cat("==================================================\n")
  
  # Read the feature importance CSV for this model
  feature_data <- read.csv(file.path(output_dir, "vis_output", 
                                     paste0("feature_importance_", model_type, "_boot", B, ".csv")))
  
  # Extract brain vertices only
  brain_features <- feature_data[feature_data$feature_type == "vertex", ]
  
  # Create count arrays for left and right hemispheres
  lh_counts <- rep(0, 10242)
  rh_counts <- rep(0, 10242)
  
  for (i in 1:nrow(brain_features)) {
    vertex_num <- brain_features$vertex_num[i] + 1  # R is 1-indexed
    count <- brain_features$count[i]
    
    if (brain_features$hemisphere[i] == "left") {
      lh_counts[vertex_num] <- count
    } else if (brain_features$hemisphere[i] == "right") {
      rh_counts[vertex_num] <- count
    }
  }
  
  # Combine left and right hemisphere counts
  vals <- c(lh_counts, rh_counts)
  
  cat("Total vertices:", length(vals), "\n")
  cat("Vertices selected at least once:", sum(vals > 0), "\n")
  cat("Max selection count:", max(vals), "\n")
  
  # Create colormap (white to red)
  col_fun <- colorRampPalette(c("white", "red"))(1000)
  
  # Map values to colors
  vmin <- 0
  vmax <- B
  breaks <- seq(vmin, vmax, length.out = length(col_fun) + 1)
  colors <- col_fun[cut(vals, breaks = breaks, include.lowest = TRUE, labels = FALSE)]
  
  # Create 3D visualization
  open3d()
  shade3d(
    tmesh3d(vertices = t(coords), indices = t(faces), homogeneous = FALSE),
    color = colors,
    meshColor = "vertices"
  )
  
  # Get display name for title
  display_name <- model_display_names[[model_type]]
  
  rgl_obj <- rglwidget(width = 600, height = 600)
  
  # Create HTML with title
  html_output <- browsable(
    tagList(
      tags$h2(paste(display_name, "- Top 10% vertices selection"), 
              style = "text-align:center; margin-bottom:20px;"),
      tags$div(
        style = "display:flex; flex-direction:row; align-items:center; justify-content:center; gap:8px;",
        rgl_obj
      )
    )
  )
  
  # Save to HTML file
  html_file <- file.path(output_dir, paste0(model_type, "_brain_heatmap.html"))
  save_html(html_output, file = html_file)
  
  cat("Saved HTML:", html_file, "\n")
  
  # Print summary statistics
  cat("Model:", display_name, "\n")
  cat("Total vertices: 20,484\n")
  cat("Vertices always selected:", sum(vals == B), "\n")
  cat("Mean (non-zero):", mean(vals[vals > 0]), "\n")
  
  # Close the current rgl device
  close3d()
}




