#install.packages(c("dplyr", "readr"))
#install.packages("vctrs")
library(dplyr)
library(readr)
library(rsample)
#library(tidyverse)

my_R2 = function(true, pred) {
  ss_res <- sum((true - pred)^2)
  ss_tot <- sum((true - mean(true))^2)
  r_squared <- 1 - (ss_res / ss_tot)
  return(r_squared)
}

wd="."  # Set to your FVE project directory if not running from it

######################### brain_mapping #########################
# For bootstrapping
reg_data_org_all = readr::read_csv(file.path(wd, "data_out", "brain_mapping_mean_regular.csv")) %>% select(-'...1')
reg_data_org_partial = readr::read_csv(file.path(wd, "data_out", "brain_mapping_mean_partial.csv"))   %>% select(-'...1')
reg_data_org_partial_tsa = readr::read_csv(file.path(wd, "data_out", "brain_mapping_mean_partial_tsa.csv"))  %>% select(-'...1')

dim(reg_data_org_all)
dim(reg_data_org_partial)



# For standard split
reg_train_data_org = readr::read_csv(file.path(wd, "data_out", "brain_mapping_mean_regular_train.csv"))  %>% select(-'...1')
reg_test_data_org  = readr::read_csv(file.path(wd, "data_out", "brain_mapping_mean_regular_test.csv"))  %>% select(-'...1')
reg_train_data_org_partial = readr::read_csv(file.path(wd, "data_out", "brain_mapping_mean_partial_train.csv"))  %>% select(-'...1')
reg_test_data_org_partial = readr::read_csv(file.path(wd, "data_out", "brain_mapping_mean_partial_test.csv"))  %>% select(-'...1')
reg_train_data_org_partial_tsa = readr::read_csv(file.path(wd, "data_out", "brain_mapping_mean_partial_tsa_train.csv"))  %>% select(-'...1')
reg_test_data_org_partial_tsa = readr::read_csv(file.path(wd, "data_out", "brain_mapping_mean_partial_tsa_test.csv"))  %>% select(-'...1')


dim(reg_train_data_org)
dim(reg_test_data_org)
dim(reg_train_data_org_partial)
dim(reg_test_data_org_partial)



################################################################################


mapping_reg <- function(train_data, test_data) {
  train_model = lm(nihtbx_cryst_uncorrected ~ . , data=train_data)
  sum_train_model = summary(train_model)
  test_true_y = test_data$nihtbx_cryst_uncorrected
  test_pred_y = predict(train_model, newdata = test_data)
  test_r2 = my_R2(true=test_true_y, pred=test_pred_y)
  train_r2 = sum_train_model$r.squared
  
  return(c(train_r2, test_r2, train_model))
}





##########################  Var estimation Bootstrap  ########################## 

# Bootstrap sampling
B = 50
set.seed(1013)


if (B !=1) {
  boot_samples <-bootstraps(reg_data_org_all, times = B)
  boot_samples_p <-bootstraps(reg_data_org_partial, times = B)
  boot_samples_p_tsa <- bootstraps(reg_data_org_partial_tsa, times = B)
  
  test_ratio = 0.2
  n = nrow(reg_data_org_all)
  n1 = as.integer(n*(1-test_ratio))
  n2 = n-n1
}


all_models = list()

reg_test_r2 = c()
partial_test_r2 = c()
partial_tsa_test_r2 = c()
reg_train_r2 = c()
partial_train_r2 = c()
partial_tsa_train_r2 = c()


# Loop through each bootstrap resample
for (b in 1:B) {
  print(paste0("B=",b,"/", B," start at ",Sys.time()))
  
  
  #regular
  if (B==1) {
    print("B=1, using standard test/train split")
    reg_train_data = reg_train_data_org
    reg_test_data = reg_test_data_org
  } else {
    split <- boot_samples$splits[[b]]
    reg_train_data <- analysis(split)      # In-bag
    oob_samples  <- assessment(split)
    reg_test_data <- oob_samples %>% slice_sample(n = n2, replace = TRUE)
  }
  

  r2 = mapping_reg(reg_train_data, reg_test_data) 
  reg_train_r2 = c(reg_train_r2, r2[[1]])
  reg_test_r2 = c(reg_test_r2, r2[[2]])
  train_mapping = r2[[3]]

  if (B==1) {
    #save(train_pca, file=paste0(wd, "/PCA_output/PCA_train_result.Rdata"))
    #save(test_pca, file=paste0(wd, "/PCA_output/PCA_test_result.Rdata"))
  } else {
    all_models[[paste0("mapping_train", b)]] = train_mapping
  }
  
  
  # partial
  if (B==1) {
    print("B=1, using standard test/train split")
    reg_train_data_partial = reg_train_data_org_partial
    reg_test_data_partial = reg_test_data_org_partial
  } else {
    split_p <- boot_samples_p$splits[[b]]
    reg_train_data_partial <- analysis(split_p)      # In-bag
    oob_samples_p  <- assessment(split_p)
    reg_test_data_partial <- oob_samples_p %>% slice_sample(n = n2, replace = TRUE)
  }
  
  
  r2_partial = mapping_reg(reg_train_data_partial, reg_test_data_partial) 
  partial_train_r2 = c(partial_train_r2, r2_partial[[1]])
  partial_test_r2 = c(partial_test_r2, r2_partial[[2]])  
  train_mapping_p = r2_partial[[3]]
  
  if (B==1) {
    #save(train_pca_p, file=paste0(wd, "/PCA_output/PCA_partial_train_result.Rdata"))
    #save(test_pca_p, file=paste0(wd, "/PCA_output/PCA_partial_test_result.Rdata"))
  } else {
    all_models[[paste0("mapping_p_train", b)]] = train_mapping_p

  }
  
  # partial_tsa
  if (B==1) {
    print("B=1, using standard test/train split")
    reg_train_data_partial_tsa = reg_train_data_org_partial_tsa
    reg_test_data_partial_tsa = reg_test_data_org_partial_tsa
  } else {
    split_p_tsa <- boot_samples_p_tsa$splits[[b]]
    reg_train_data_partial_tsa <- analysis(split_p_tsa)      # In-bag
    oob_samples_p_tsa  <- assessment(split_p_tsa)
    reg_test_data_partial_tsa <- oob_samples_p_tsa %>% slice_sample(n = n2, replace = TRUE)
  }
  
  
  
  r2_partial_tsa = mapping_reg(reg_train_data_partial_tsa, reg_test_data_partial_tsa) 
  partial_tsa_train_r2 = c(partial_tsa_train_r2, r2_partial_tsa[[1]])
  partial_tsa_test_r2 = c(partial_tsa_test_r2, r2_partial_tsa[[2]])  
  train_mapping_p_tsa = r2_partial_tsa[[3]]
  
  if (B==1) {
    #save(train_pca_p_tsa, file=paste0(wd, "/PCA_output/PCA_partial_tsa_train_result.Rdata"))
    #save(test_pca_p_tsa, file=paste0(wd, "/PCA_output/PCA_partial_tsa_test_result.Rdata"))
  } else {
    all_models[[paste0("mapping_p_tsa_train", b)]] = train_mapping_p_tsa
  }
  
}


result_dict_boot <- list(reg_test_r2 = reg_test_r2, partial_test_r2 = partial_test_r2, partial_tsa_test_r2 = partial_tsa_test_r2,
                         reg_train_r2 = reg_train_r2, partial_train_r2 = partial_train_r2, partial_tsa_train_r2 = partial_tsa_train_r2)

result_tbl_boot <- data.frame(
  variable = names(result_dict_boot),
  mean = sapply(result_dict_boot, mean),
  var_boot   = sapply(result_dict_boot, var),   
  sd_boot =   sapply(result_dict_boot, sd)
)
save(result_tbl_boot, file=paste0(wd, "/PCA_output/mapping_result_tbl_boot_", B, ".Rdata"))


if (B!=1) {
  save(all_models, file=paste0(wd, "/PCA_output/mapping_result_all_models_", B, ".Rdata"))
  
} 


################################################################################
###################   Brain mapping coefficients    ############################   
################################################################################
library(readr)

# Configuration
wd <- "."  # Set to your FVE project directory if not running from it
output_dir <- file.path(wd, "PCA_output")
B <- 50

# Load brain mapping data to get region column names
brain_mapping_mean <- read_csv(file.path(wd, "data_out/brain_mapping_mean_regular.csv"))
exclude_cols <- c("interview_age", "sex_2", "nihtbx_cryst_uncorrected", "...1")
region_cols <- setdiff(names(brain_mapping_mean), exclude_cols)


# Load R models
load(file.path(wd, "PCA_output", paste0("mapping_result_all_models_", B, ".Rdata")))


# Model configurations
model_configs <- list(
  list(name = "regular", prefix = "mapping_train"),
  list(name = "partial", prefix = "mapping_p_train"),
  list(name = "partial_tsa", prefix = "mapping_p_tsa_train")
)

# Process each model type
for (i in 1:length(model_configs)) {
  config = model_configs[[i]]
  
  model_name <- config$name
  model_prefix <- config$prefix
  
  cat("\n", strrep("=", 60), "\n")
  cat("Processing", model_name, "models\n")
  cat(strrep("=", 60), "\n")
  
  # Storage for coefficients across bootstraps
  all_bootstrap_coefs <- list()
  
  for (b in 1:B) {
    model_key <- paste0(model_prefix, b)
    coefs <- all_models[[model_key]]
    coef_names <- names(coefs)
    
    # Extract region coefficients (exclude intercept and covariates)
    region_coefs <- coefs[!coef_names %in% c("(Intercept)", "interview_age", "sex_2")]
    
    all_bootstrap_coefs[[b]] <- region_coefs
    
    if (b == 1) {
      cat("Sample regions:", head(names(region_coefs), 5), "\n")
      cat("Total regions in model:", length(region_coefs), "\n")
    }
  }
  
  # Calculate mean coefficients across bootstraps
  all_region_names <- unique(unlist(lapply(all_bootstrap_coefs, names)))
  
  mean_region_coefs <- numeric(length(all_region_names))
  names(mean_region_coefs) <- all_region_names
  
  for (region_name in all_region_names) {
    values <- sapply(all_bootstrap_coefs, function(x) {
      if (region_name %in% names(x)) x[region_name] else 0
    })
    mean_region_coefs[region_name] <- mean(values, na.rm = TRUE)
  }
  
  
  # Create dataframe with region coefficients
  region_coef_df <- data.frame(
    region_name = names(mean_region_coefs),
    mean_coef = as.numeric(mean_region_coefs),
    stringsAsFactors = FALSE
  )
  
  # Save region coefficients to CSV
  csv_file <- file.path(output_dir, 
                        paste0("brain_mapping_mean_coeff_", model_name, "_boot", B, ".csv"))
  write_csv(region_coef_df, csv_file)
}

################################################################################
########################   label -> vertices    ################################   
################################################################################

library(readr)
library(gifti)

# Configuration
wd <- "."  # Set to your FVE project directory if not running from it
output_dir <- file.path(wd, "PCA_output")
B <- 50

# Load FreeSurfer labels
label_L <- read_gifti(file.path(wd, "data_out/abcd_labels.L.10k.label.gii"))
label_R <- read_gifti(file.path(wd, "data_out/abcd_labels.R.10k.label.gii"))

data_L <- label_L$data$labels
data_R <- label_R$data$labels

# Get unique labels
unique_labels <- sort(unique(c(data_L, data_R)))



# Model types
model_types <- c("regular", "partial", "partial_tsa")

# Process each model type
for (model_name in model_types) {
  
  cat("\n", strrep("=", 60), "\n")
  cat("Processing", model_name, "\n")
  cat(strrep("=", 60), "\n")
  
  # Load region coefficients
  coef_file <- file.path(wd, "PCA_output", 
                         paste0("brain_mapping_mean_coeff_", model_name, "_boot", B, ".csv"))
  
  if (!file.exists(coef_file)) {
    cat("File not found:", coef_file, "\n")
    next
  }
  
  region_coefs <- read_csv(coef_file)
  
  cat("Loaded", nrow(region_coefs), "region coefficients\n")
  
  # Initialize vertex coefficient arrays
  vertex_coefs_left <- numeric(10242)
  vertex_coefs_right <- numeric(10242)
  
  # Map each label to its coefficient
  for (i in seq_along(unique_labels)) {
    label_val <- unique_labels[i]
    
    # Find corresponding region name
    # Assuming region names are like "label_X" where X is the label value
    region_name <- paste0("label_", label_val)
    
    # Check if this region exists in coefficients
    region_row <- region_coefs[region_coefs$region_name == region_name, ]
    
    if (nrow(region_row) > 0) {
      coef_value <- region_row$mean_coef[1]
      
      # Assign coefficient to all vertices with this label
      # Left hemisphere
      mask_left <- data_L == label_val
      vertex_coefs_left[mask_left] <- coef_value
      
      # Right hemisphere
      mask_right <- data_R == label_val
      vertex_coefs_right[mask_right] <- coef_value
    }
  }
  
  # Combine left and right hemispheres
  vertex_coefs <- c(vertex_coefs_left, vertex_coefs_right)
  
  cat("Non-zero vertices:", sum(vertex_coefs != 0), "\n")
  cat("Min:", min(vertex_coefs), ", Max:", max(vertex_coefs), "\n")
  
  # Create dataframe for visualization
  vertex_coef_df <- data.frame(
    vertex_id = 0:20483,
    hemisphere = c(rep("left", 10242), rep("right", 10242)),
    vertex_num = rep(0:10241, 2),
    mean_coef = vertex_coefs
  )
  
  # Save to CSV
  csv_file <- file.path(output_dir,
                        paste0("mean_coef_mapping_", model_name, "_boot", B, ".csv"))
  write_csv(vertex_coef_df, csv_file)
  cat("Saved:", csv_file, "\n")
}

################################################################################
########################   FS -> SurfeView (using saved mapping)   #############   
################################################################################

library(readr)

# Configuration
wd <- "."  # Set to your FVE project directory if not running from it
output_dir <- file.path(wd, "PCA_output")
B <- 50

# Load the saved mapping from Python
mapping <- read_csv(file.path(wd, "data_out/orig_to_fs_mapping.csv"))

mapping_left <- mapping[mapping$hemisphere == "left", ]
mapping_right <- mapping[mapping$hemisphere == "right", ]



# Create inverse mapping: FreeSurfer index -> Original index
# orig_to_fs: mapping_left$fs_idx[i] is the FS index for original vertex i
# We need fs_to_orig: for each FS vertex, which original vertex does it come from?

fs_to_orig_left <- integer(10242)
fs_to_orig_right <- integer(10242)

for (i in 1:nrow(mapping_left)) {
  orig_idx <- mapping_left$orig_idx[i]  # 0-indexed in CSV
  fs_idx <- mapping_left$fs_idx[i]      # 0-indexed in CSV
  
  # Convert to 1-indexed for R
  fs_to_orig_left[fs_idx + 1] <- orig_idx + 1
}

for (i in 1:nrow(mapping_right)) {
  orig_idx <- mapping_right$orig_idx[i]
  fs_idx <- mapping_right$fs_idx[i]
  
  # Convert to 1-indexed for R
  fs_to_orig_right[fs_idx + 1] <- orig_idx + 1
}



# Model types
model_types <- c("regular", "partial", "partial_tsa")

# Process each model type
for (model_name in model_types) {
  
  cat("\n", strrep("=", 60), "\n")
  cat("Processing", model_name, "\n")
  cat(strrep("=", 60), "\n")
  
  # Load FreeSurfer vertex coefficients
  fs_coef_file <- file.path(output_dir, 
                            paste0("mean_coef_mapping_", model_name, "_boot", B, ".csv"))
  
  if (!file.exists(fs_coef_file)) {
    cat("File not found:", fs_coef_file, "\n")
    next
  }
  
  fs_coef_data <- read.csv(fs_coef_file)
  
  # Extract FreeSurfer coefficients (20484 vertices)
  # These are in FreeSurfer vertex order (0-indexed in vertex_num, but row is 1-indexed)
  fs_coefs_left <- fs_coef_data$mean_coef[1:10242]
  fs_coefs_right <- fs_coef_data$mean_coef[10243:20484]
  
  cat("Loaded FreeSurfer coefficients\n")
  cat("  Left non-zero:", sum(fs_coefs_left != 0), "\n")
  cat("  Right non-zero:", sum(fs_coefs_right != 0), "\n")
  
  # Map back to original SurfeView ordering
  orig_coefs_left <- numeric(10242)
  orig_coefs_right <- numeric(10242)
  
  for (fs_idx in 1:10242) {
    orig_idx <- fs_to_orig_left[fs_idx]
    orig_coefs_left[orig_idx] <- fs_coefs_left[fs_idx]
  }
  
  for (fs_idx in 1:10242) {
    orig_idx <- fs_to_orig_right[fs_idx]
    orig_coefs_right[orig_idx] <- fs_coefs_right[fs_idx]
  }
  
  # Combine left and right
  orig_coefs <- c(orig_coefs_left, orig_coefs_right)
  
  cat("Mapped to SurfeView ordering\n")
  cat("  Non-zero vertices:", sum(orig_coefs != 0), "\n")
  cat("  Min:", min(orig_coefs), ", Max:", max(orig_coefs), "\n")
  
  # Create dataframe in SurfeView ordering
  vertex_coef_df <- data.frame(
    vertex_id = 0:20483,
    hemisphere = c(rep("left", 10242), rep("right", 10242)),
    vertex_num = rep(0:10241, 2),
    mean_coef = orig_coefs
  )
  
  # Save to CSV
  csv_file <- file.path(output_dir,
                        paste0("mean_coef_mapping_surfview_", model_name, "_boot", B, ".csv"))
  write_csv(vertex_coef_df, csv_file)
}

