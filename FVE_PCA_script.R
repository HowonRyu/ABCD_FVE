#install.packages(c("dplyr", "rsample"))
library(dplyr)
#library(readr)
library(rsample)
#library(tidyverse)

my_R2 = function(true, pred) {
  ss_res <- sum((true - pred)^2)
  ss_tot <- sum((true - mean(true))^2)
  r_squared <- 1 - (ss_res / ss_tot)
  return(r_squared)
}


######################### PCA #########################
# For bootstrapping
reg_data_org_all = read.csv(file.path(wd, "data_out", "FVE_dat.csv")) 
reg_data_org_partial = read.csv(file.path(wd, "data_out", "FVE_dat_partial.csv")) 
reg_data_org_partial_tsa = read.csv(file.path(wd, "data_out", "FVE_dat_partial_tsa.csv"))

dim(reg_data_org_all)
dim(reg_data_org_partial)


# For standard split
reg_train_data_org = read.csv(file.path(wd, "data_out", "reg_train_data.csv"))
reg_test_data_org  = read.csv(file.path(wd, "data_out", "reg_test_data.csv"))
reg_train_data_org_partial = read.csv(file.path(wd, "data_out", "reg_train_data_partial.csv"))
reg_test_data_org_partial = read.csv(file.path(wd, "data_out", "reg_test_data_partial.csv"))
reg_train_data_org_partial_tsa = read.csv(file.path(wd, "data_out", "reg_train_data_partial_tsa.csv"))
reg_test_data_org_partial_tsa = read.csv(file.path(wd, "data_out", "reg_test_data_partial_tsa.csv"))


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

scale_x = function(X_df) {
  X_df_scaled = X_df %>% mutate(across(everything(), ~ if (sd(.) == 0) {0} else {( . - mean(.) ) / sd(.)} )  )
  return(X_df_scaled)
}



proc_pca_output <- function(train_df, test_df, pca_train_output, pca_test_output, n_pcs, name, plotting=FALSE) {
  #plotting
  if (plotting == TRUE) {
  cumulative_ve = cumsum(pca_train_output$sdev^2 / sum(pca_train_output$sdev^2))
  png(filename=paste0(wd, "/PCA_output/", name, "_fve_plot.png"))
  plot(cumulative_ve, main=paste0("(train) Prop var. explained at PC", n_pcs,
                                  " = ", round(cumulative_ve[n_pcs],3)),
       ylab="cumulative prop var explained",
       xlab="PC")
  dev.off()
  }

  # train PCs
  if (n_pcs == 1) {
    pcs_train <- data.frame(PC1 = pca_train_output$x[, 1])
    train_eigenvects <- pca_train_output$rotation[, 1, drop = FALSE]
  } else {
    pcs_train <- as.data.frame(pca_train_output$x[, 1:n_pcs, drop = FALSE])
    colnames(pcs_train) <- paste0("PC", seq_len(n_pcs))
    if (n_pcs == 1) {
      dim(pca_train_output$rotation)
      train_eigenvects <- pca_train_output$rotation[, 1]
      
    } else {
      train_eigenvects <- pca_train_output$rotation[, 1:n_pcs]
    }

  }

  # test PCs
  X_test_df <- test_df %>% select(-nihtbx_cryst_uncorrected)
  X_test_df_scaled <- scale_x(X_test_df)

  pcs_test <- as.data.frame(as.matrix(X_test_df_scaled) %*% train_eigenvects)
  colnames(pcs_test) <- colnames(pcs_train)

  # add outcome
  pcs_train$nihtbx_cryst_uncorrected <- train_df$nihtbx_cryst_uncorrected
  pcs_test$nihtbx_cryst_uncorrected <- test_df$nihtbx_cryst_uncorrected

  # linear regression
  pca_train_lin = lm(nihtbx_cryst_uncorrected ~ ., data = pcs_train)
  pca_test_pred = predict(pca_train_lin, newdata=pcs_test)
  train_rsq = summary(pca_train_lin)$r.squared
  test_rsq = my_R2(true=pcs_test$nihtbx_cryst_uncorrected, pred=pca_test_pred)

  # store original coefficients
  if (n_pcs == 1) {
    dim(pca_train_output$rotation)
    length(coef(pca_train_lin))
    length(coef(pca_train_lin)[-1])
    train_coeffs = pca_train_output$rotation[, 1] %*% coef(pca_train_lin)[-1]
  } else {
      train_coeffs = data.frame(pca_train_output$rotation[, 1:n_pcs] %*% coef(pca_train_lin)[-1])
  }

  
  return(c(train_rsq, test_rsq, train_coeffs))
}




##########################  Var estimation Bootstrap  ########################## 

# Bootstrap sampling
B = 50
seed = 1013
set.seed(seed)
n_pcs = 1   #counter so far: n_pcs=2, 
nickname = paste0(n_pcs, "_", B, "_", seed)


if (B == 1) {
  plotting = TRUE
} else {
  plotting = FALSE
}
if (B !=1) {
  boot_samples <- rsample::bootstraps(reg_data_org_all, times = B)
  boot_samples_p <- rsample::bootstraps(reg_data_org_partial, times = B)
  boot_samples_p_tsa <- rsample::bootstraps(reg_data_org_partial_tsa, times = B)
  
  test_ratio = 0.2
  n = nrow(reg_data_org_all)
  n1 = as.integer(n*(1-test_ratio))
  n2 = n-n1
}


all_coeffs = list()

reg_test_r2 = c()
partial_test_r2 = c()
partial_tsa_test_r2 = c()
reg_train_r2 = c()
partial_train_r2 = c()
partial_tsa_train_r2 = c()
reg_coeff_tbl = matrix(nrow=B, ncol=20486)
partial_coeff_tbl = matrix(nrow=B, ncol=20484)
partial_tsa_coeff_tbl = matrix(nrow=B, ncol=20484)

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
  
  
  regular_rsq = proc_pca_output(reg_train_data, reg_test_data, train_pca, test_pca, n_pcs, "regular", plotting)
  reg_train_r2 = c(reg_train_r2, regular_rsq[[1]])
  reg_test_r2 = c(reg_test_r2, regular_rsq[[2]])
  reg_coeff_tbl[b,] <- regular_rsq[[3]]
  
  if (B==1) {
    save(train_pca, file=paste0(wd, "/PCA_output/PCA_train_result_", nickname, ".Rdata"))
    save(test_pca, file=paste0(wd, "/PCA_output/PCA_test_result_", nickname, ".Rdata"))
  } else {}

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
  
  partial_rsq = proc_pca_output(reg_train_data_partial, reg_test_data_partial, train_pca_p, test_pca_p, n_pcs, "partial", plotting)

  partial_train_r2 = c(partial_train_r2, partial_rsq[[1]])
  partial_test_r2 = c(partial_test_r2, partial_rsq[[2]])
  partial_coeff_tbl[b,] <- partial_rsq[[3]]
  
  if (B==1) {
    save(train_pca_p, file=paste0(wd, "/PCA_output/PCA_partial_train_result_", nickname, ".Rdata"))
    save(test_pca_p, file=paste0(wd, "/PCA_output/PCA_partial_test_result_", nickname, ".Rdata"))
  } else {
    #all_models[[paste0("PCA_p_train_", nickname)]] = train_pca_p
    #all_models[[paste0("PCA_p_test_", nickname)]] = test_pca_p
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
  
  partial_tsa_rsq = proc_pca_output(reg_train_data_partial_tsa, reg_test_data_partial_tsa, train_pca_p_tsa, test_pca_p_tsa, n_pcs, "partial_tsa", plotting)

  partial_tsa_train_r2 = c(partial_tsa_train_r2, partial_tsa_rsq[[1]])
  partial_tsa_test_r2 = c(partial_tsa_test_r2, partial_tsa_rsq[[2]])
  partial_tsa_coeff_tbl[b,] <- partial_tsa_rsq[[3]]
  
  if (B==1) {
    save(train_pca_p_tsa, file=paste0(wd, "/PCA_output/PCA_partial_tsa_train_result_", nickname, ".Rdata"))
    save(test_pca_p_tsa, file=paste0(wd, "/PCA_output/PCA_partial_tsa_test_result_", nickname, ".Rdata"))
  } else {
    #all_models[[paste0("PCA_p_tsa_train_", nickname)]] = train_pca_p_tsa
    #all_models[[paste0("PCA_p_tsa_test_", nickname)]] = test_pca_p_tsa
  }
  
}


result_dict_boot <- list(reg_test_r2 = reg_test_r2, partial_test_r2 = partial_test_r2, partial_tsa_test_r2 = partial_tsa_test_r2,
reg_train_r2 = reg_train_r2, partial_train_r2 = partial_train_r2, partial_tsa_train_r2 = partial_tsa_train_r2)

save(result_dict_boot, file=paste0(wd, "/PCA_output/result_dict_boot_", nickname, ".Rdata")  )

result_tbl_boot <- data.frame(
  variable = names(result_dict_boot),
  mean = sapply(result_dict_boot, mean),
  var_boot   = sapply(result_dict_boot, var),   
  sd_boot =   sapply(result_dict_boot, sd)
)
save(result_tbl_boot, file=paste0(wd, "/PCA_output/result_tbl_boot_", nickname, ".Rdata")  )



if (B!=1) {
write.csv(reg_coeff_tbl, file = paste0(wd, "/PCA_output/PCA_reg_org_coeffs_", nickname, ".csv"), row.names = FALSE)
write.csv(partial_coeff_tbl, file = paste0(wd, "/PCA_output/PCA_partial_org_coeffs_", nickname, ".csv"), row.names = FALSE)
write.csv(partial_tsa_coeff_tbl, file = paste0(wd, "/PCA_output/PCA_partial_tsa_org_coeffs_", nickname, ".csv"), row.names = FALSE)
} 
