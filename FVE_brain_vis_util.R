normalize_coeffs <- function(vals, method = "minmax") {
  if (method == "raw")      return(vals)
  if (method == "abs")      return(abs(vals))
  if (sd(vals, na.rm = TRUE) == 0) return(rep(0, length(vals)))
  if (method == "minmax")   return((vals - min(vals, na.rm = TRUE)) / (max(vals, na.rm = TRUE) - min(vals, na.rm = TRUE)))
  if (method == "quantile") return(ecdf(vals)(vals))
  if (method == "normal")   return((vals - mean(vals, na.rm = TRUE)) / sd(vals, na.rm = TRUE))
}

get_raw_mean <- function(model_name, output_dir, getter_type) {
  if (getter_type == "lr") {
    N_VERTICES_PER_HEMI <- 10242
    model_file <- file.path(output_dir, paste0("LR_all_models_boot", B, ".Rdata"))
    if (!file.exists(model_file)) return(NULL)
    env <- new.env()
    load(model_file, envir = env)
    all_models <- env$all_models
    first_model_key <- paste0(model_name, "_b1")
    if (!(first_model_key %in% names(all_models))) return(NULL)
    all_coef_names  <- rownames(all_models[[first_model_key]]$beta)
    exclude_pattern <- "^\\(Intercept\\)|^age$|^sex$|^interview_age$"
    vertex_indices  <- which(!grepl(exclude_pattern, all_coef_names, ignore.case = TRUE))
    N_VERTICES      <- length(vertex_indices)
    coef_matrix <- matrix(NA, nrow = N_VERTICES, ncol = B)
    for (b in 1:B) {
      model_key <- paste0(model_name, "_b", b)
      if (model_key %in% names(all_models))
        coef_matrix[, b] <- all_models[[model_key]]$beta[vertex_indices]
    }
    return(rowMeans(coef_matrix, na.rm = TRUE))

  } else if (getter_type == "rf") {
    N_VERTICES <- 20484
    rf_files <- list(
      "RF"             = "rf_importance.csv",
      "RF_partial"     = "rf_partial_importance.csv",
      "RF_partial_tsa" = "rf_partial_tsa_importance.csv"
    )
    feature_file <- file.path(output_dir, rf_files[[model_name]])
    if (!file.exists(feature_file)) return(NULL)
    feature_data <- read.csv(feature_file)[, -1]
    if (ncol(feature_data) == N_VERTICES + 2) feature_data <- feature_data[, 1:N_VERTICES]
    vals <- colMeans(feature_data)
    if (all(is.na(vals))) return(NULL)
    return(vals)

  } else if (getter_type == "mapping_fs") {
    model_name_clean <- sub("mapping_fs_", "", model_name)
    coef_file <- file.path(output_dir,
                           paste0("mean_coef_mapping_freesurfer_", model_name_clean, "_boot", B, ".csv"))
    if (!file.exists(coef_file)) return(NULL)
    return(read.csv(coef_file)$mean_coef)

  } else if (getter_type == "mapping") {
    model_name_clean <- sub("mapping_", "", model_name)
    coef_file <- file.path(output_dir,
                           paste0("mean_coef_mapping_surfview_", model_name_clean, "_boot", B, ".csv"))
    if (!file.exists(coef_file)) return(NULL)
    return(read.csv(coef_file)$mean_coef)

  } else if (getter_type == "pca") {
    parts      <- strsplit(model_name, "_")[[1]]
    num_pcs    <- as.integer(parts[2])
    model_type <- paste(parts[3:length(parts)], collapse = "_")
    M    <- 50
    SEED <- 1013
    N_VERTICES <- 20484
    if (model_type == "regular") {
      coef_file <- file.path(output_dir, paste0("PCA_reg_org_coeffs_",         num_pcs, "_", M, "_", SEED, ".csv"))
    } else if (model_type == "partial") {
      coef_file <- file.path(output_dir, paste0("PCA_partial_org_coeffs_",     num_pcs, "_", M, "_", SEED, ".csv"))
    } else if (model_type == "partial_tsa") {
      coef_file <- file.path(output_dir, paste0("PCA_partial_tsa_org_coeffs_", num_pcs, "_", M, "_", SEED, ".csv"))
    }
    if (!file.exists(coef_file)) return(NULL)
    coef_data <- read.csv(coef_file)
    if (model_type == "regular") coef_data <- coef_data[1:N_VERTICES, ]
    coef_cols <- coef_data[, grep("^V[0-9]+$", colnames(coef_data))]
    if (ncol(coef_cols) == 0) return(NULL)
    return(rowMeans(coef_cols, na.rm = TRUE))
  }
}

apply_norm <- function(raw, norm_method, n_vertices_per_hemi = NULL) {
  if (is.null(raw)) return(NULL)
  if (norm_method == "raw") {
    vals <- raw
  } else {
    vals <- normalize_coeffs(abs(raw), method = norm_method)
  }
  if (!is.null(n_vertices_per_hemi)) {
    return(c(vals[1:n_vertices_per_hemi],
             vals[(n_vertices_per_hemi + 1):length(vals)]))
  }
  return(vals)
}

get_lr_vals <- function(model_name, output_dir, norm_method = "minmax") {
  raw <- get_raw_mean(model_name, output_dir, "lr")
  apply_norm(raw, norm_method, n_vertices_per_hemi = 10242)
}

get_rf_vals <- function(model_name, output_dir, norm_method = "minmax") {
  raw <- get_raw_mean(model_name, output_dir, "rf")
  apply_norm(raw, norm_method)
}

get_mapping_vals <- function(model_name, output_dir, norm_method = "minmax") {
  raw <- get_raw_mean(model_name, output_dir, "mapping")
  apply_norm(raw, norm_method)
}

get_mapping_fs_vals <- function(model_name, output_dir, norm_method = "minmax") {
  raw <- get_raw_mean(model_name, output_dir, "mapping_fs")
  apply_norm(raw, norm_method)
}

get_pca_vals <- function(model_name, output_dir, norm_method = "minmax") {
  raw <- get_raw_mean(model_name, output_dir, "pca")
  apply_norm(raw, norm_method, n_vertices_per_hemi = 10242)
}

################################################################################
# Brain visualization
################################################################################

visualize_brain_models <- function(
    model_family,
    model_names,
    model_display_names,
    output_dir,
    table_filename,
    get_vals_function,
    use_freesurfer = FALSE,
    norm_method = "minmax") {

  if (use_freesurfer) {
    surf_coords <- coords_fs
    surf_faces  <- faces_fs
  } else {
    surf_coords <- coords
    surf_faces  <- faces
  }

  colorbar_range <- if (norm_method == "normal") c(-1.5, 1.5) else NULL

  create_brain_html <- function(vals, output_file, title, view_angle, ceiling_quantile = 0.90) {
    col_fun <- colorRampPalette(COLOR_GRADIENT)(1000)

    if (!is.null(colorbar_range)) {
      vmin   <- colorbar_range[1]
      vmax   <- colorbar_range[2]
      breaks <- seq(vmin, vmax, length.out = length(col_fun) + 1)
      colors <- col_fun[pmin(pmax(cut(vals, breaks = breaks, include.lowest = TRUE,
                                      labels = FALSE), 1), length(col_fun))]
    } else {
      vmin      <- 0
      p_ceiling <- quantile(vals, ceiling_quantile, na.rm = TRUE)
      breaks    <- seq(vmin, p_ceiling, length.out = length(col_fun) + 1)
      colors    <- col_fun[pmin(cut(vals, breaks = breaks, include.lowest = TRUE,
                                    labels = FALSE), length(col_fun))]
    }

    open3d()
    shade3d(
      tmesh3d(vertices = t(surf_coords), indices = t(surf_faces), homogeneous = FALSE),
      color     = colors,
      meshColor = "vertices"
    )
    if (view_angle == "front") {
      view3d(theta = 0,   phi = 0, zoom = 0.7)
    } else if (view_angle == "left") {
      view3d(theta = -90, phi = 0, zoom = 0.7)
    } else if (view_angle == "right") {
      view3d(theta = 90,  phi = 0, zoom = 0.7)
    } else if (view_angle == "back") {
      view3d(theta = 180, phi = 0, zoom = 0.7)
    }
    rgl_obj     <- rglwidget(width = 600, height = 600)
    html_output <- browsable(
      tagList(tags$div(
        style = "display:flex; flex-direction:row; align-items:center; justify-content:center;",
        rgl_obj
      ))
    )
    save_html(html_output, file = output_file)
    close3d()
  }

  vals_list <- list()

  for (model_name in model_names) {
    vals                    <- get_vals_function(model_name, output_dir, norm_method = norm_method)
    vals_list[[model_name]] <- vals

    for (view in c("front", "left", "right", "back")) {
      html_file <- file.path(output_dir, "vis_output",
                             paste0(model_name, "_brain_", view, ".html"))
      create_brain_html(vals, html_file,
                        paste(model_display_names[[model_name]], "-", toupper(view)),
                        view)
      png_file <- file.path(output_dir, "vis_output",
                            paste0(model_name, "_brain_", view, ".png"))
      webshot(html_file, png_file, vwidth = 800, vheight = 800, delay = 3)
    }
  }

  create_plot_table(model_family, model_names, model_display_names,
                    output_dir, table_filename, vals_list, norm_method)
}

create_plot_table <- function(model_family, model_names, model_display_names,
                              output_dir, table_filename, vals_list, norm_method = "minmax") {
  plot_list <- list()

  plot_list[[1]] <- textGrob("Model",  gp = gpar(fontsize = 16, fontface = "bold"))
  plot_list[[2]] <- textGrob("Front",  gp = gpar(fontsize = 16, fontface = "bold"))
  plot_list[[3]] <- textGrob("Left",   gp = gpar(fontsize = 16, fontface = "bold"))
  plot_list[[4]] <- textGrob("Right",  gp = gpar(fontsize = 16, fontface = "bold"))
  plot_list[[5]] <- textGrob("Bottom", gp = gpar(fontsize = 16, fontface = "bold"))
  plot_list[[6]] <- textGrob("",       gp = gpar(fontsize = 16, fontface = "bold"))

  generic_display_names <- c("Full", "Partial", "TSA Partial")

  for (i in seq_along(model_names)) {
    model_name   <- model_names[i]
    display_name <- generic_display_names[i]

    plot_list[[length(plot_list) + 1]] <- textGrob(display_name, gp = gpar(fontsize = 14))

    for (view in c("front", "left", "right", "back")) {
      img_path <- file.path(output_dir, "vis_output",
                            paste0(model_name, "_brain_", view, ".png"))
      if (file.exists(img_path)) {
        img <- readPNG(img_path)
        if (view == "left") {
          img <- aperm(img, c(2, 1, 3))
          img <- img[nrow(img):1, , ]
        } else if (view == "right") {
          img <- aperm(img, c(2, 1, 3))
          img <- img[, ncol(img):1, ]
        }
        plot_list[[length(plot_list) + 1]] <- rasterGrob(img, interpolate = TRUE)
      } else {
        plot_list[[length(plot_list) + 1]] <- textGrob("N/A")
      }
    }

    vals <- vals_list[[model_name]]
    if (!is.null(vals)) {
      vmin <- if (norm_method == "normal") -1.5 else min(vals, na.rm = TRUE)
      vmax <- if (norm_method == "normal")  1.5 else max(vals, na.rm = TRUE)
      gradient_colors <- rev(colorRampPalette(COLOR_GRADIENT)(100))
      colorbar_plot   <- gTree(children = gList(
        rasterGrob(matrix(gradient_colors, ncol = 1),
                   x = 0.15, y = 0.5, width = 0.05, height = 0.8, interpolate = TRUE),
        textGrob(sprintf("%.2f", vmax), x = 0.22, y = 0.90, just = "left", gp = gpar(fontsize = 9)),
        textGrob(sprintf("%.2f", vmin), x = 0.22, y = 0.10, just = "left", gp = gpar(fontsize = 9))
      ))
      plot_list[[length(plot_list) + 1]] <- colorbar_plot
    } else {
      plot_list[[length(plot_list) + 1]] <- textGrob("N/A")
    }
  }

  manuscript_dir <- file.path("manuscript_plots", norm_method)
  if (!dir.exists(manuscript_dir)) dir.create(manuscript_dir, recursive = TRUE)

  base_name <- sub("\\.png$", paste0("_", norm_method, ".png"), table_filename)
  save_path <- file.path(manuscript_dir, base_name)

  png(save_path, width = 2800, height = 850 * length(model_names), res = 300)
  grid.arrange(grobs = plot_list, ncol = 6, widths = COLUMN_WIDTHS)
  dev.off()
}

################################################################################
# Cross-model comparison
################################################################################

create_cross_model_comparison <- function(formulation_type, formulation_index,
                                          comparison_models, norm_method = "minmax") {
  model_order       <- c("Mean per ROI", "LASSO", "Ridge", "Random Forest")
  col_widths_images <- COLUMN_WIDTHS[2:6]

  header_row <- arrangeGrob(
    textGrob("Front",  gp = gpar(fontsize = 16, fontface = "bold")),
    textGrob("Left",   gp = gpar(fontsize = 16, fontface = "bold")),
    textGrob("Right",  gp = gpar(fontsize = 16, fontface = "bold")),
    textGrob("Bottom", gp = gpar(fontsize = 16, fontface = "bold")),
    textGrob("",       gp = gpar(fontsize = 16, fontface = "bold")),
    ncol = 5, widths = col_widths_images
  )

  row_list <- list(header_row)

  for (model_label in model_order) {
    model_info        <- comparison_models[[model_label]]
    model_name        <- model_info$model_names[formulation_index]
    output_dir        <- model_info$output_dir
    get_vals_function <- model_info$get_vals_function

    cat("\nProcessing", model_label, "-", formulation_type, "\n")

    vals_standardized <- get_vals_function(model_name, output_dir, norm_method = norm_method)

    title_row <- arrangeGrob(
      textGrob(model_label, gp = gpar(fontsize = 15, fontface = "bold"),
               just = "left", x = unit(0.01, "npc")),
      ncol = 1
    )
    row_list[[length(row_list) + 1]] <- title_row

    if (is.null(vals_standardized)) {
      cat("  Data not found, skipping\n")
      row_list[[length(row_list) + 1]] <- arrangeGrob(
        grobs = replicate(5, textGrob("N/A"), simplify = FALSE),
        ncol = 5, widths = col_widths_images
      )
      next
    }

    img_grobs <- list()
    for (view in c("front", "left", "right", "back")) {
      img_path <- file.path(output_dir, "vis_output",
                            paste0(model_name, "_brain_", view, ".png"))
      if (file.exists(img_path)) {
        img <- readPNG(img_path)
        if (view == "left") {
          img <- aperm(img, c(2, 1, 3))
          img <- img[nrow(img):1, , ]
        } else if (view == "right") {
          img <- aperm(img, c(2, 1, 3))
          img <- img[, ncol(img):1, ]
        }
        img_grobs[[length(img_grobs) + 1]] <- rasterGrob(img, interpolate = TRUE)
      } else {
        img_grobs[[length(img_grobs) + 1]] <- textGrob("N/A")
      }
    }

    vmin <- if (norm_method == "normal") -1.5 else min(vals_standardized, na.rm = TRUE)
    vmax <- if (norm_method == "normal")  1.5 else max(vals_standardized, na.rm = TRUE)
    gradient_colors <- rev(colorRampPalette(COLOR_GRADIENT)(100))
    colorbar_grob   <- gTree(children = gList(
      rasterGrob(matrix(gradient_colors, ncol = 1),
                 x = 0.15, y = 0.5, width = 0.05, height = 0.8, interpolate = TRUE),
      textGrob(sprintf("%.2f", vmax), x = 0.22, y = 0.90, just = "left", gp = gpar(fontsize = 9)),
      textGrob(sprintf("%.2f", vmin), x = 0.22, y = 0.10, just = "left", gp = gpar(fontsize = 9))
    ))

    row_list[[length(row_list) + 1]] <- arrangeGrob(
      grobs = c(img_grobs, list(colorbar_grob)),
      ncol = 5, widths = col_widths_images
    )
  }

  n_models   <- length(model_order)
  final_plot <- arrangeGrob(grobs = row_list, ncol = 1,
                             heights = c(0.4, rep(c(0.3, 2.5), n_models)))

  manuscript_dir <- file.path("manuscript_plots", norm_method)
  if (!dir.exists(manuscript_dir)) dir.create(manuscript_dir, recursive = TRUE)

  filename  <- paste0("Cross_Model_Comparison_", gsub(" ", "_", formulation_type),
                      "_", norm_method, ".png")
  save_path <- file.path(manuscript_dir, filename)

  png(save_path, width = 2800 * (5/6),
      height = 300 + (850 + 120) * n_models + 150, res = 350)
  main_title <- textGrob(formulation_type, gp = gpar(fontsize = 36, fontface = "bold"))
  grid.arrange(main_title, final_plot, ncol = 1, heights = c(0.05, 1))
  dev.off()

  cat("\nSaved:", save_path, "\n")
}

################################################################################
# Distribution plots
################################################################################

plot_coefficient_distributions <- function(formulation_type, formulation_index,
                                            comparison_models, norm_method = "minmax") {
  model_order <- c("Mean per ROI", "LASSO", "Ridge", "Random Forest")

  getter_types <- list(
    "Mean per ROI"  = "mapping_fs",
    "LASSO"         = "lr",
    "Ridge"         = "lr",
    "Random Forest" = "rf"
  )

  grob_list      <- list()
  grob_list[[1]] <- textGrob("Raw",
                              gp = gpar(fontsize = 20, fontface = "bold"))
  grob_list[[2]] <- textGrob("Standardized Absolute",
                              gp = gpar(fontsize = 20, fontface = "bold"))

  for (model_label in model_order) {
    model_info  <- comparison_models[[model_label]]
    model_name  <- model_info$model_names[formulation_index]
    output_dir  <- model_info$output_dir
    getter_type <- getter_types[[model_label]]

    raw <- get_raw_mean(model_name, output_dir, getter_type)

    if (is.null(raw)) {
      for (j in 1:2) grob_list[[length(grob_list) + 1]] <- textGrob("N/A")
      next
    }

    if (getter_type == "mapping_fs") raw <- unique(raw)
    raw_vals <- raw
    std_vals  <- normalize_coeffs(abs(raw), method = norm_method)

    p_raw <- ggplot(data.frame(x = raw_vals), aes(x = x)) +
      geom_histogram(bins = 80, fill = "steelblue", color = NA, alpha = 0.8) +
      labs(title = model_label, x = NULL, y = "Count") +
      theme_minimal(base_size = 16) +
      theme(plot.title = element_text(face = "bold", size = 18))

    p_std <- ggplot(data.frame(x = std_vals), aes(x = x)) +
      geom_histogram(bins = 80, fill = "darkred", color = NA, alpha = 0.8) +
      labs(title = model_label, x = NULL, y = "Count") +
      scale_x_continuous(breaks = seq(0, 1, by = 0.2)) +
      coord_cartesian(xlim = c(-0.1, 1.1)) +
      theme_minimal(base_size = 16) +
      theme(plot.title = element_text(face = "bold", size = 18))

    grob_list[[length(grob_list) + 1]] <- ggplotGrob(p_raw)
    grob_list[[length(grob_list) + 1]] <- ggplotGrob(p_std)
  }

  manuscript_dir <- file.path("manuscript_plots", norm_method)
  if (!dir.exists(manuscript_dir)) dir.create(manuscript_dir, recursive = TRUE)

  filename  <- paste0("Coef_Distribution_", gsub(" ", "_", formulation_type),
                      "_", norm_method, ".png")
  save_path <- file.path(manuscript_dir, filename)

  n_models <- length(model_order)
  png(save_path, width = 2800 * (5/6),
      height = 300 + (500 + 120) * n_models + 150, res = 350)
  main_title <- textGrob(formulation_type, gp = gpar(fontsize = 36, fontface = "bold"))
  plot_body  <- arrangeGrob(grobs = grob_list, ncol = 2,
                             heights = c(0.12, rep(1, n_models)))
  grid.arrange(main_title, plot_body, ncol = 1, heights = c(0.05, 1))
  dev.off()

  cat("Saved:", save_path, "\n")
}

################################################################################
# Bootstrap correlation analysis
################################################################################

compute_bootstrap_correlations <- function(comparison_models, formulation_index = 1) {
  model_order <- c("Mean per ROI", "LASSO", "Ridge", "Random Forest")
  n_models    <- length(model_order)

  getter_types <- list(
    "Mean per ROI"  = "mapping_fs",
    "LASSO"         = "lr",
    "Ridge"         = "lr",
    "Random Forest" = "rf"
  )

  get_boot_vec <- function(model_label, model_name, output_dir, b) {
    getter_type <- getter_types[[model_label]]

    if (getter_type == "lr") {
      env <- new.env()
      load(file.path(output_dir, paste0("LR_all_models_boot", B, ".Rdata")), envir = env)
      all_models      <- env$all_models
      model_key       <- paste0(model_name, "_b", b)
      if (!model_key %in% names(all_models)) return(NULL)
      first_model     <- all_models[[paste0(model_name, "_b1")]]
      all_coef_names  <- rownames(first_model$beta)
      exclude_pattern <- "^\\(Intercept\\)|^age$|^sex$|^interview_age$"
      vertex_indices  <- which(!grepl(exclude_pattern, all_coef_names, ignore.case = TRUE))
      return(all_models[[model_key]]$beta[vertex_indices])

    } else if (getter_type == "rf") {
      rf_files <- list(
        "RF"             = "rf_importance.csv",
        "RF_partial"     = "rf_partial_importance.csv",
        "RF_partial_tsa" = "rf_partial_tsa_importance.csv"
      )
      feature_file <- file.path(output_dir, rf_files[[model_name]])
      if (!file.exists(feature_file)) return(NULL)
      feature_data <- read.csv(feature_file)[, -1]
      N_VERTICES   <- 20484
      if (ncol(feature_data) == N_VERTICES + 2) feature_data <- feature_data[, 1:N_VERTICES]
      if (b > nrow(feature_data)) return(NULL)
      return(as.numeric(feature_data[b, ]))

    } else if (getter_type == "mapping_fs") {
      model_name_clean <- sub("mapping_fs_", "", model_name)
      coef_file <- file.path(output_dir,
                             paste0("mean_coef_mapping_freesurfer_", model_name_clean, "_boot", B, ".csv"))
      if (!file.exists(coef_file)) return(NULL)
      coef_data <- read.csv(coef_file)
      if (paste0("b", b) %in% colnames(coef_data)) {
        raw_vec <- coef_data[[paste0("b", b)]]
      } else {
        raw_vec <- coef_data$mean_coef
      }
      expanded <- raw_vec
      for (val in unique(raw_vec)) {
        idx <- which(raw_vec == val)
        expanded[idx] <- val / length(idx)
      }
      return(expanded)
    }
  }

  corr_array <- array(NA, dim = c(n_models, n_models, B),
                      dimnames = list(model_order, model_order, paste0("b", 1:B)))

  for (b in 1:B) {
    cat("Bootstrap", b, "\n")
    vecs <- list()
    for (model_label in model_order) {
      model_info          <- comparison_models[[model_label]]
      vecs[[model_label]] <- get_boot_vec(model_label,
                                          model_info$model_names[formulation_index],
                                          model_info$output_dir, b)
    }
    if (any(sapply(vecs, is.null))) next
    corr_array[, , b] <- cor(do.call(cbind, vecs), use = "pairwise.complete.obs")
  }

  corr_mean <- apply(corr_array, c(1, 2), mean, na.rm = TRUE)
  corr_var  <- apply(corr_array, c(1, 2), var,  na.rm = TRUE)

  cat("\nMean correlation matrix:\n")
  print(round(corr_mean, 4))
  cat("\nVariance of correlations:\n")
  print(round(corr_var,  4))

  list(mean = corr_mean, var = corr_var, array = corr_array)
}

plot_bootstrap_correlations <- function(corr_results, norm_method = "minmax",
                                        formulation_type = "Full") {
  model_order <- c("Mean per ROI", "LASSO", "Ridge", "Random Forest")
  corr_mean   <- corr_results$mean
  corr_var    <- corr_results$var
  corr_array  <- corr_results$array

  pairs     <- combn(model_order, 2, simplify = FALSE)
  pair_data <- do.call(rbind, lapply(pairs, function(p) {
    data.frame(
      pair  = paste(p[1], "&", p[2]),
      value = corr_array[p[1], p[2], ],
      stringsAsFactors = FALSE
    )
  }))

  corr_mean_long         <- reshape2::melt(corr_mean)
  corr_var_long          <- reshape2::melt(corr_var)
  corr_combined_long     <- corr_mean_long
  corr_combined_long$var <- corr_var_long$value

  p_heat <- ggplot(corr_combined_long, aes(x = Var1, y = Var2, fill = value)) +
    geom_tile(color = "white") +
    geom_text(aes(label = sprintf("%.3f\n(±%.3f)", value, sqrt(var))),
              size = 4, lineheight = 0.9) +
    scale_fill_gradient2(low = "steelblue", mid = "white", high = "darkred",
                         midpoint = 0, limits = c(-1, 1), name = "r") +
    labs(title = "(A) Coefficient Correlation (+/- SD)", x = NULL, y = NULL) +
    theme_minimal(base_size = 14) +
    theme(axis.text.x = element_text(angle = 30, hjust = 1, size = 12),
          axis.text.y = element_text(size = 12),
          plot.title  = element_text(face = "bold", size = 16))

  p_dist <- ggplot(pair_data, aes(x = pair, y = value, fill = pair)) +
    geom_boxplot(outlier.size = 1.5, alpha = 0.8) +
    geom_hline(yintercept = 0, linetype = "dashed", color = "grey50") +
    scale_fill_brewer(palette = "Set2") +
    coord_cartesian(ylim = c(-0.06, 0.35)) +
    labs(title = "(B) Coefficient Correlation Distributions",
         x = NULL, y = "Pearson correlation coefficient") +
    theme_minimal(base_size = 14) +
    theme(axis.text.x     = element_text(angle = 30, hjust = 1, size = 11),
          plot.title      = element_text(face = "bold", size = 16),
          legend.position = "none")

  manuscript_dir <- file.path("manuscript_plots", norm_method)
  if (!dir.exists(manuscript_dir)) dir.create(manuscript_dir, recursive = TRUE)

  write.csv(corr_mean, file.path(manuscript_dir, paste0("bootstrap_corr_mean_", formulation_type, "_", norm_method, ".csv")))
  write.csv(corr_var,  file.path(manuscript_dir, paste0("bootstrap_corr_var_",  formulation_type, "_", norm_method, ".csv")))

  combined <- arrangeGrob(p_heat, p_dist, ncol = 2, widths = c(1, 1.5))
  titled   <- arrangeGrob(
    textGrob(formulation_type, gp = gpar(fontsize = 26, fontface = "bold"), x = 0, just = "left"),
    combined,
    ncol = 1, heights = c(0.08, 1)
  )
  ggsave(file.path(manuscript_dir, paste0("bootstrap_corr_combined_", formulation_type, "_", norm_method, ".png")),
         titled, width = 17, height = 5.5, dpi = 350)

  cat("Saved correlation plots to", manuscript_dir, "\n")
  
  
}
