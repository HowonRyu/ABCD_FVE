library(dplyr)
#library(tidyverse)
#.libPaths("/data/howon/FVE/Rlibs")

my_R2 = function(true, pred) {
  ss_res <- sum((true - pred)^2)
  ss_tot <- sum((true - mean(true))^2)
  r_squared <- 1 - (ss_res / ss_tot)
  return(r_squared)
}

#wd="/niddk-data-central/mae_hr/FVE"
wd="~/Projects/FVE"

######################### PCA #########################
# For bootstrapping
reg_data_org_all = readr::read_csv(file.path(wd, "data_out", "FVE_dat.csv")) 
reg_data_org_partial = readr::read_csv(file.path(wd, "data_out", "FVE_dat_partial.csv")) 
reg_data_org_partial_tsa = readr::read_csv(file.path(wd, "data_out", "FVE_dat_partial_tsa.csv"))

dim(reg_data_org_all)
dim(reg_data_org_partial)


# For standard split
reg_train_data_org = readr::read_csv(file.path(wd, "data_out", "reg_train_data.csv"))
reg_test_data_org  = readr::read_csv(file.path(wd, "data_out", "reg_test_data.csv"))
reg_train_data_org_partial = readr::read_csv(file.path(wd, "data_out", "reg_train_data_partial.csv"))
reg_test_data_org_partial = readr::read_csv(file.path(wd, "data_out", "reg_test_data_partial.csv"))
reg_train_data_org_partial_tsa = readr::read_csv(file.path(wd, "data_out", "reg_train_data_partial_tsa.csv"))
reg_test_data_org_partial_tsa = readr::read_csv(file.path(wd, "data_out", "reg_test_data_partial_tsa.csv"))


dim(reg_train_data_org)
dim(reg_test_data_org)
dim(reg_train_data_org_partial)
dim(reg_test_data_org_partial)

################################################################################


scale_and_pca = function(df) {
  X_df = df %>% select(-nihtbx_cryst_uncorrected)
  means <- sapply(X_df, mean)
  sds   <- sapply(X_df, sd)
  X_df_scaled <- as_tibble(scale(X_df, center = means, scale = sds))
  pca_output = prcomp(X_df_scaled)
  return(pca_output)
} 


##########################  Var estimation Bootstrap  ########################## 

# Bootstrap sampling
B = 1
set.seed(1013)

if (B !=1) {
  boot_samples <- bootstraps(reg_data_org_all, times = B)
  boot_samples_p <- bootstraps(reg_data_org_partial, times = B)
  boot_samples_p_tsa <- bootstraps(reg_data_org_partial_tsa, times = B)
  
  test_ratio = 0.2
  n = nrow(reg_data_org_all)
  n1 = as.integer(n*(1-test_ratio))
  n2 = n-n1
}


all_models = list()

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
  
  X_var_mat_train =  reg_train_data %>% select(-nihtbx_cryst_uncorrected)
  X_var_mat_test = reg_test_data %>% select(-nihtbx_cryst_uncorrected)
  
  train_pca = scale_and_pca(X_var_mat_train)
  test_pca = scale_and_pca(X_var_mat_test)
  
  if (B==1) {
    save(train_pca, file=paste0(wd, "/LR_output/PCA_train_result.Rdata"))
    save(test_pca, file=paste0(wd, "/LR_output/PCA_test_result.Rdata"))
  } else {
    all_models[[paste0("PCA_train", b)]] = train_pca
    all_models[[paste0("PCA_test", b)]] = test_pca
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
  
  X_var_mat_train_p =  reg_train_data_partial %>% select(-nihtbx_cryst_uncorrected)
  X_var_mat_test_p =  reg_test_data_partial %>% select(-nihtbx_cryst_uncorrected)

  train_pca_p = scale_and_pca(X_var_mat_train_p)
  test_pca_p = scale_and_pca(X_var_mat_test_p)
  
  if (B==1) {
    save(train_pca_p, file=paste0(wd, "/LR_output/PCA_partial_train_result.Rdata"))
    save(test_pca_p, file=paste0(wd, "/LR_output/PCA_partial_test_result.Rdata"))
  } else {
    all_models[[paste0("PCA_p_train", b)]] = train_pca_p
    all_models[[paste0("PCA_p_test", b)]] = test_pca_p
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
  
  X_var_mat_train_p_tsa = reg_train_data_partial_tsa %>% select(-nihtbx_cryst_uncorrected)
  X_var_mat_test_p_tsa = reg_test_data_partial_tsa %>% select(-nihtbx_cryst_uncorrected)
  
  
  train_pca_p_tsa = scale_and_pca(X_var_mat_train_p_tsa)
  test_pca_p_tsa = scale_and_pca(X_var_mat_test_p_tsa)
  
  if (B==1) {
    save(train_pca_p_tsa, file=paste0(wd, "/LR_output/PCA_partial_tsa_train_result.Rdata"))
    save(test_pca_p_tsa, file=paste0(wd, "/LR_output/PCA_partial_tsa_test_result.Rdata"))
  } else {
    all_models[[paste0("PCA_p_tsa_train", b)]] = train_pca_p_tsa
    all_models[[paste0("PCA_p_tsa_test", b)]] = test_pca_p_tsa
  }
  
}

if (B!=1) {
  save(all_models, file=paste0(wd, "/LR_output/PCA_all_models_", B, ".Rdata"))
} 

  



