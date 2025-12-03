install.packages(c("dplyr", "readr", "rsample"))
library(dplyr)
library(readr)
library(rsample)
#library(tidyverse)
#.libPaths("/data/howon/FVE/Rlibs")

my_R2 = function(true, pred) {
  ss_res <- sum((true - pred)^2)
  ss_tot <- sum((true - mean(true))^2)
  r_squared <- 1 - (ss_res / ss_tot)
  return(r_squared)
}

wd="/niddk-data-central/mae_hr/FVE"
#wd="~/Projects/FVE"

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
  #X_df_cleaned = X_df[complete.cases(X_df),]
  X_df_scaled = X_df %>% 
    mutate(across(everything(), ~ if (sd(.) == 0) {0} else {( . - mean(.) ) / sd(.)} )  )
  pca_output = prcomp(X_df_scaled)
  return(pca_output)
} 



proc_pca_output <- function(train_df, test_df, pca_train_output, pca_test_output, n_pcs, name) {
  #plotting
  cumulative_ve = cumsum(pca_train_output$sdev^2 / sum(pca_train_output$sdev^2))
  png(filename=paste0(wd, "/PCA_output/", name, "_fve_plot.png"))
  plot(cumulative_ve, main=paste0("(train) Prop var. explained at PC", n_pcs,
                                  " = ", round(cumulative_ve[n_pcs],3)),
       ylab="cumulative prop var explained",
       xlab="PC")
  dev.off()
  
  
  # get FVE from the linear model
  pcs_train = as.data.frame(pca_train_output$x[, 1:n_pcs])
  train_eigenvects = pca_train_output$rotation[, 1:n_pcs]
  X_test_df = test_df %>% select(-nihtbx_cryst_uncorrected)

  print(dim(train_eigenvects))
  print(dim(as.matrix(X_test_df)))
  
  pcs_test = as.data.frame((as.matrix(X_test_df) %*% train_eigenvects))
  

  pcs_train$nihtbx_cryst_uncorrected = train_df$nihtbx_cryst_uncorrected
  pcs_test$nihtbx_cryst_uncorrected = test_df$nihtbx_cryst_uncorrected

  pca_train_lin = lm(nihtbx_cryst_uncorrected ~ ., data = pcs_train)

  pca_test_lin = lm(nihtbx_cryst_uncorrected ~ ., data = pcs_test)
  
  train_rsq = summary(pca_train_lin)$r.squared
  test_rsq = summary(pca_test_lin)$r.squared
  
  #test_pred_y = predict(pca_train_lin, pcs_test)
  #outcome_y_test = pcs_test$nihtbx_cryst_uncorrected
  #test_rsq = my_R2(outcome_y_test, test_pred_y)
  return(c(train_rsq, test_rsq))
}




##########################  Var estimation Bootstrap  ########################## 

# Bootstrap sampling
B = 50
set.seed(1013)
n_pcs = 100

if (B !=1) {
  boot_samples <- rsample::bootstraps(reg_data_org_all, times = B)
  boot_samples_p <- rsample::bootstraps(reg_data_org_partial, times = B)
  boot_samples_p_tsa <- rsample::bootstraps(reg_data_org_partial_tsa, times = B)
  
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
  
  
  
  print(paste0("regular data PCA start at ", Sys.time()))
  train_pca = scale_and_pca(reg_train_data)
  test_pca = scale_and_pca(reg_test_data)
  print(paste0("regular data PCA done at ", Sys.time()))
  
  
  regular_rsq = proc_pca_output(reg_train_data, reg_test_data, train_pca, test_pca, n_pcs, "regular")

  reg_train_r2 = c(reg_train_r2, regular_rsq[1])
  reg_test_r2 = c(reg_test_r2, regular_rsq[2])
  
  
  if (B==1) {
    save(train_pca, file=paste0(wd, "/PCA_output/PCA_train_result.Rdata"))
    save(test_pca, file=paste0(wd, "/PCA_output/PCA_test_result.Rdata"))
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
  
  
  print(paste0("partial data PCA start at ",Sys.time()))
  train_pca_p = scale_and_pca(reg_train_data_partial)
  test_pca_p = scale_and_pca(reg_test_data_partial)
  print(paste0("partial data PCA done at ",Sys.time()))
  
  partial_rsq = proc_pca_output(reg_train_data_partial, reg_test_data_partial, train_pca_p, test_pca_p, n_pcs, "partial")

  partial_train_r2 = c(partial_train_r2, partial_rsq[1])
  partial_test_r2 = c(partial_test_r2, partial_rsq[2])
  
  
  if (B==1) {
    save(train_pca_p, file=paste0(wd, "/PCA_output/PCA_partial_train_result.Rdata"))
    save(test_pca_p, file=paste0(wd, "/PCA_output/PCA_partial_test_result.Rdata"))
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
  
  
  
  print(paste0("partial_tsa data PCA start at ",Sys.time()))
  train_pca_p_tsa = scale_and_pca(reg_train_data_partial_tsa)
  test_pca_p_tsa = scale_and_pca(reg_test_data_partial_tsa)
  print(paste0("partial_tsa data PCA done at ",Sys.time()))
  
  partial_tsa_rsq = proc_pca_output(reg_train_data_partial_tsa, reg_test_data_partial_tsa, train_pca_p_tsa, test_pca_p_tsa, n_pcs, "partial_tsa")

  partial_tsa_train_r2 = c(partial_tsa_train_r2, partial_tsa_rsq[1])
  partial_tsa_test_r2 = c(partial_tsa_test_r2, partial_tsa_rsq[2])
  
  if (B==1) {
    save(train_pca_p_tsa, file=paste0(wd, "/PCA_output/PCA_partial_tsa_train_result.Rdata"))
    save(test_pca_p_tsa, file=paste0(wd, "/PCA_output/PCA_partial_tsa_test_result.Rdata"))
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
save(result_tbl_boot, file=paste0(wd, "/PCA_output/result_tbl_boot_", B, ".Rdata"))


if (B!=1) {
  #save(all_models, file=paste0(wd, "/PCA_output/PCA_all_models_", B, ".Rdata"))

} 





