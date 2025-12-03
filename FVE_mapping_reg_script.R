install.packages(c("dplyr", "readr"))
library(dplyr)
library(readr)
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
reg_data_org_all = readr::read_csv(file.path(wd, "data_out", "brain_mapping_mean_regular.csv")) 
reg_data_org_partial = readr::read_csv(file.path(wd, "data_out", "brain_mapping_mean_partial.csv")) 
reg_data_org_partial_tsa = readr::read_csv(file.path(wd, "data_out", "brain_mapping_mean_partial_tsa.csv"))

dim(reg_data_org_all)
dim(reg_data_org_partial)



# For standard split
reg_train_data_org = readr::read_csv(file.path(wd, "data_out", "brain_mapping_mean_regular_train.csv"))
reg_test_data_org  = readr::read_csv(file.path(wd, "data_out", "brain_mapping_mean_regular_test.csv"))
reg_train_data_org_partial = readr::read_csv(file.path(wd, "data_out", "brain_mapping_mean_partial_train.csv"))
reg_test_data_org_partial = readr::read_csv(file.path(wd, "data_out", "brain_mapping_mean_partial_test.csv"))
reg_train_data_org_partial_tsa = readr::read_csv(file.path(wd, "data_out", "brain_mapping_mean_partial_tsa_train.csv"))
reg_test_data_org_partial_tsa = readr::read_csv(file.path(wd, "data_out", "brain_mapping_mean_partial_tsa_test.csv"))


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
  
  return(c(train_r2, test_r2))
}





##########################  Var estimation Bootstrap  ########################## 

# Bootstrap sampling
B = 5
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
  reg_train_r2 = c(reg_train_r2, r2[1])
  reg_test_r2 = c(reg_test_r2, r2[2])  
  
  if (B==1) {
    #save(train_pca, file=paste0(wd, "/PCA_output/PCA_train_result.Rdata"))
    #save(test_pca, file=paste0(wd, "/PCA_output/PCA_test_result.Rdata"))
  } else {
    #all_models[[paste0("PCA_train", b)]] = train_pca
    #all_models[[paste0("PCA_test", b)]] = test_pca
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
  partial_train_r2 = c(partial_train_r2, r2_partial[1])
  partial_test_r2 = c(partial_test_r2, r2_partial[2])  
  
  
  if (B==1) {
    #save(train_pca_p, file=paste0(wd, "/PCA_output/PCA_partial_train_result.Rdata"))
    #save(test_pca_p, file=paste0(wd, "/PCA_output/PCA_partial_test_result.Rdata"))
  } else {
    #all_models[[paste0("PCA_p_train", b)]] = train_pca_p
    #all_models[[paste0("PCA_p_test", b)]] = test_pca_p
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
  partial_tsa_train_r2 = c(partial_tsa_train_r2, r2_partial_tsa[1])
  partial_tsa_test_r2 = c(partial_tsa_test_r2, r2_partial_tsa[2])  
  
  
  if (B==1) {
    #save(train_pca_p_tsa, file=paste0(wd, "/PCA_output/PCA_partial_tsa_train_result.Rdata"))
    #save(test_pca_p_tsa, file=paste0(wd, "/PCA_output/PCA_partial_tsa_test_result.Rdata"))
  } else {
    #all_models[[paste0("PCA_p_tsa_train", b)]] = train_pca_p_tsa
    #all_models[[paste0("PCA_p_tsa_test", b)]] = test_pca_p_tsa
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
  #save(all_models, file=paste0(wd, "/PCA_output/PCA_all_models_", B, ".Rdata"))
  
} 





