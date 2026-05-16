library(dplyr)
library(GGally)
library(broom)
library(glmnet)
library(rsample)


my_R2 = function(true, pred) {
  ss_res <- sum((true - pred)^2)
  ss_tot <- sum((true - mean(true))^2)
  r_squared <- 1 - (ss_res / ss_tot)
  return(r_squared)
}

setwd(".")  # Set to your FVE project directory if not running from it
################################################################################
################### CV for var(error) estimation    ############################
################################################################################

# Data load
reg_train_data_org = read_csv("data_out/lin_reg_train_data.csv") #9083 rows
reg_test_data_org = read_csv("data_out/lin_reg_test_data.csv") #2271 



reg_data_org = rbind(reg_train_data_org, reg_test_data_org)


# set numbers for cross validation
run_again=TRUE




###############  Var estimation method 1 (Nadeau et al.) ############### 

method1_dict = list()
J_list = seq(5, 105, 10)
if (run_again==TRUE) {
  for (J_index in seq_along(J_list)) {
    J = J_list[[J_index]] 
    test_ratio = 0.2
    n = nrow(reg_data_org)
    n1 = as.integer(n*(1-test_ratio))
    n2 = n-n1
    
    
    print(paste0("-------------------J=",J, "-------------------"))
    
    # initialize loop
    covar_test_r2 = c()
    surface_test_r2 = c()
    all_test_r2 = c()
    all_quad_test_r2 = c()
    resid_test_r2 = c()
    resid2_test_r2 = c()
    
    # loop through
    for (j in 1:J) {
      print(paste0("split=",j,"/",J, " start at ", Sys.time()))
      # define train and test
      train_idx = sample(n, n1)         # Randomly pick n1 indices
      reg_train_data = reg_data_org[train_idx, ]     # Training set
      reg_test_data = reg_data_org[-train_idx, ]    
      
      ### regular analysis
      model_covar <- lm(nihtbx_cryst_uncorrected ~ sex + interview_age , data=reg_train_data)
      model_surface <- lm(nihtbx_cryst_uncorrected ~ total_surface_area, data=reg_train_data)
      model_all <- lm(nihtbx_cryst_uncorrected ~sex +  interview_age + total_surface_area , data=reg_train_data)
      model_all_quad <- lm(nihtbx_cryst_uncorrected ~sex +  interview_age + total_surface_area +I(total_surface_area^2), data=reg_train_data)
      
      ### partial analysis
      reg_train_data_partial = reg_train_data
      reg_train_data_partial$total_surface_area =  lm(total_surface_area ~ sex + interview_age, data=reg_train_data)$resid
      reg_train_data_partial$nihtbx_cryst_uncorrected = lm(nihtbx_cryst_uncorrected ~ sex + interview_age, data=reg_train_data)$resid
      
      # (y~ age+Sex) ~ (tsa ~ age+sex)
      model_partial1 <- lm(nihtbx_cryst_uncorrected ~ total_surface_area, data=reg_train_data_partial)
      # (y~ age+Sex) ~ (tsa ~ age+sex) + (tsa^2 ~ age+sex)
      model_partial2 <- lm(nihtbx_cryst_uncorrected ~ total_surface_area + I(total_surface_area^2), data=reg_train_data_partial)
      
      
      
      ## Test data
      reg_test_data_partial = reg_test_data
      reg_test_data_partial$total_surface_area =  lm(total_surface_area ~ sex + interview_age, data=reg_test_data)$resid
      reg_test_data_partial$nihtbx_cryst_uncorrected = lm(nihtbx_cryst_uncorrected ~ sex + interview_age, data=reg_test_data)$resid
      
      test_y_true = reg_test_data$nihtbx_cryst_uncorrected
      test_y_true_partial = reg_test_data_partial$nihtbx_cryst_uncorrected
      
      covar_test_pred = predict(model_covar, newdata = reg_test_data)
      surface_test_pred = predict(model_surface, newdata = reg_test_data)
      all_test_pred = predict(model_all, newdata = reg_test_data)
      all_quad_test_pred = predict(model_all_quad, newdata = reg_test_data)
      resid_test_pred = predict(model_partial1, newdata = reg_test_data_partial)
      resid2_test_pred = predict(model_partial2, newdata = reg_test_data_partial)
      
      
      # Calculate R-squared on test data
      covar_test_r2 = c(covar_test_r2, my_R2(true=test_y_true, pred=covar_test_pred))
      surface_test_r2 = c(surface_test_r2, my_R2(true=test_y_true, pred=surface_test_pred))
      all_test_r2 = c(all_test_r2, my_R2(true=test_y_true, pred=all_test_pred))
      all_quad_test_r2 = c(all_quad_test_r2, my_R2(true=test_y_true, pred=all_quad_test_pred))
      resid_test_r2 = c(resid_test_r2, my_R2(true=test_y_true_partial, pred=resid_test_pred))
      resid2_test_r2 = c(resid2_test_r2, my_R2(true=test_y_true_partial, pred=resid2_test_pred))
    }
    
    # save results
    result_dict <- list(
      covar_test_r2 = covar_test_r2, surface_test_r2 = surface_test_r2, all_test_r2 = all_test_r2, all_quad_test_r2 = all_quad_test_r2,
      resid_test_r2=resid_test_r2, resid2_test_r2=resid2_test_r2
    )
    
    estimate1_factor = ((1/J)+(n2/n1))
    
    result_tbl <- data.frame(
      variable = names(result_dict),
      mean = sapply(result_dict, mean),
      sd = sqrt(estimate1_factor*sapply(result_dict, var))
    )
    method1_dict[[J_index]] = result_tbl
    
    # print the result
    #(result_tbl[, c(2,3)] <- round(result_tbl[, c(2,3)], 4))
    
  }
  save(method1_dict, file = "method1_dict.RData")
}



###############  Var estimation method 2 (Nadeau et al.) ############### 


method2_dict = list()
M_list = c(20, 40, 60, 80, 100)

if (run_again==TRUE) {
  print(paste0("Start running at:", Sys.time()))
  for (J_index in seq_along(J_list)) {
    J = J_list[[J_index]] 
    test_ratio = 0.2
    n = nrow(reg_data_org)
    n1 = as.integer(n*(1-test_ratio))
    n2 = n-n1
    method2_dict[[J_index]] = list()
    
    print(paste0("-------------------J=",J, "-------------------"))
    
    # create two disjoint splits
    for (M_index in seq_along(M_list)) {
      M = M_list[[M_index]] 
      print(paste0("-------------------M=",M, "-------------------"))
      
      mu_hat_diff_sqr_tbl = matrix(nrow=M, ncol=6)
      
      for (m in 1:M) {
        two_splits <- vfold_cv(reg_data_org, v = 2)
        D = analysis(two_splits$splits[[1]])
        D_c = assessment(two_splits$splits[[1]])
        disjoint_sets = c(D, D_c)
        disjoint_sets_label = c("D", "D_c")
        
        n1_prime = round((n/2) - n2)
        
        results_list = list()
        for (d in 1:2) {
          if (d==1) {
            D_set = D
          } else if (d==2) {
            D_set = D_c
          }
          
          covar_test_r2 = c()
          surface_test_r2 = c()
          all_test_r2 = c()
          all_quad_test_r2 = c()
          resid_test_r2 = c()
          resid2_test_r2 = c()
          
          
          # loop through
          for (j in 1:J) {
            #print(paste0("Dc=", disjoint_sets_label[d]," split=",j,"/",J, " start at ", Sys.time()))
            
            # define train and test
            train_idx = sample(nrow(D_set), n1_prime) 
            reg_train_data = D_set[train_idx, ]  
            reg_test_data = D_set[-train_idx, ]  
            
            ### regular analysis
            model_covar <- lm(nihtbx_cryst_uncorrected ~ sex + interview_age , data=reg_train_data)
            model_surface <- lm(nihtbx_cryst_uncorrected ~ total_surface_area, data=reg_train_data)
            model_all <- lm(nihtbx_cryst_uncorrected ~sex +  interview_age + total_surface_area , data=reg_train_data)
            model_all_quad <- lm(nihtbx_cryst_uncorrected ~sex +  interview_age + total_surface_area +I(total_surface_area^2), data=reg_train_data)
            
            ### partial analysis
            reg_train_data_partial = reg_train_data
            reg_train_data_partial$total_surface_area =  lm(total_surface_area ~ sex + interview_age, data=reg_train_data)$resid
            reg_train_data_partial$nihtbx_cryst_uncorrected = lm(nihtbx_cryst_uncorrected ~ sex + interview_age, data=reg_train_data)$resid
            
            
            # (y~ age+Sex) ~ (tsa ~ age+sex)
            model_partial1 <- lm(nihtbx_cryst_uncorrected ~ total_surface_area, data=reg_train_data_partial)
            # (y~ age+Sex) ~ (tsa ~ age+sex) + (tsa^2 ~ age+sex)
            model_partial2 <- lm(nihtbx_cryst_uncorrected ~ total_surface_area + I(total_surface_area^2), data=reg_train_data_partial)
            
            
            
            
            ## Test data
            reg_test_data_partial = reg_test_data
            reg_test_data_partial$total_surface_area =  lm(total_surface_area ~ sex + interview_age, data=reg_test_data)$resid
            reg_test_data_partial$nihtbx_cryst_uncorrected = lm(nihtbx_cryst_uncorrected ~ sex + interview_age, data=reg_test_data)$resid
            
            test_y_true = reg_test_data$nihtbx_cryst_uncorrected
            test_y_true_partial = reg_test_data_partial$nihtbx_cryst_uncorrected
            
            covar_test_pred = predict(model_covar, newdata = reg_test_data)
            surface_test_pred = predict(model_surface, newdata = reg_test_data)
            all_test_pred = predict(model_all, newdata = reg_test_data)
            all_quad_test_pred = predict(model_all_quad, newdata = reg_test_data)
            resid_test_pred = predict(model_partial1, newdata = reg_test_data_partial)
            resid2_test_pred = predict(model_partial2, newdata = reg_test_data_partial)
            
            
            # Calculate R-squared on test data
            covar_test_r2 = c(covar_test_r2, my_R2(true=test_y_true, pred=covar_test_pred))
            surface_test_r2 = c(surface_test_r2, my_R2(true=test_y_true, pred=surface_test_pred))
            all_test_r2 = c(all_test_r2, my_R2(true=test_y_true, pred=all_test_pred))
            all_quad_test_r2 = c(all_quad_test_r2, my_R2(true=test_y_true, pred=all_quad_test_pred))
            resid_test_r2 = c(resid_test_r2, my_R2(true=test_y_true_partial, pred=resid_test_pred))
            resid2_test_r2 = c(resid2_test_r2, my_R2(true=test_y_true_partial, pred=resid2_test_pred))
          }
          
          result_dict <- list(
            covar_test_r2 = covar_test_r2, surface_test_r2 = surface_test_r2, all_test_r2 = all_test_r2, all_quad_test_r2 = all_quad_test_r2,
            resid_test_r2=resid_test_r2, resid2_test_r2=resid2_test_r2)
          
          result_tbl <- data.frame(
            variable = names(result_dict),
            mean = sapply(result_dict, mean)
          )
          results_list[[disjoint_sets_label[d]]] = result_tbl
        }
        mu_hat_diff_sqr_tbl[m,] = (results_list[['D']][,2] - results_list[['D_c']][,2])^2
      }
      
      colnames(mu_hat_diff_sqr_tbl) = result_tbl$variable
      (estimate_2_final_result = t((1/2)*apply(mu_hat_diff_sqr_tbl, 2, mean)))
      var2_result = t(rbind(estimate_2_final_result, sqrt(estimate_2_final_result)))
      colnames(var2_result) = c("var", "sd")
      
      method2_dict[[J_index]][[M_index]] = var2_result
    }
  }
  print(paste0("Finished running at:", Sys.time()))
  # start at 19:16
  save(method2_dict, file = "method2_dict.RData")
}






##########################  Var estimation Bootstrap  ########################## 

# Bootstrap sampling
B = 50
set.seed(1004)
boot_samples <- bootstraps(reg_data_org, times = B)


# initialize loop
covar_test_r2 = c()
surface_test_r2 = c()
all_test_r2 = c()
all_quad_test_r2 = c()
resid_test_r2 = c()
resid2_test_r2 = c()

# Loop through each bootstrap resample
for (b in 1:B) {
  test_ratio = 0.2
  n = nrow(reg_data_org)
  n1 = as.integer(n*(1-test_ratio))
  n2 = n-n1
  print(paste0("B=",b,"/", B," start at ",Sys.time()))
  split <- boot_samples$splits[[b]]
  reg_train_data <- analysis(split)      # In-bag
  oob_samples  <- assessment(split)
  reg_test_data <- oob_samples %>% slice_sample(n = n2, replace = TRUE)
  
  ### regular analysis
  model_covar <- lm(nihtbx_cryst_uncorrected ~ sex_2 + interview_age , data=reg_train_data)
  model_surface <- lm(nihtbx_cryst_uncorrected ~ total_surface_area, data=reg_train_data)
  model_all <- lm(nihtbx_cryst_uncorrected ~sex_2 +  interview_age + total_surface_area , data=reg_train_data)
  model_all_quad <- lm(nihtbx_cryst_uncorrected ~sex_2 +  interview_age + total_surface_area +I(total_surface_area^2), data=reg_train_data)
  
  ### partial analysis
  reg_train_data_partial = reg_train_data
  reg_train_data_partial$total_surface_area =  lm(total_surface_area ~ sex_2 + interview_age, data=reg_train_data)$resid
  reg_train_data_partial$nihtbx_cryst_uncorrected = lm(nihtbx_cryst_uncorrected ~ sex_2 + interview_age, data=reg_train_data)$resid
  
  # (y~ age+Sex) ~ (tsa ~ age+sex)
  model_partial1 <- lm(nihtbx_cryst_uncorrected ~ total_surface_area, data=reg_train_data_partial)
  # (y~ age+Sex) ~ (tsa ~ age+sex) + (tsa^2 ~ age+sex)
  model_partial2 <- lm(nihtbx_cryst_uncorrected ~ total_surface_area + I(total_surface_area^2), data=reg_train_data_partial)
  
  ## Test data
  reg_test_data_partial = reg_test_data
  reg_test_data_partial$total_surface_area =  lm(total_surface_area ~ sex_2 + interview_age, data=reg_test_data)$resid
  reg_test_data_partial$nihtbx_cryst_uncorrected = lm(nihtbx_cryst_uncorrected ~ sex_2 + interview_age, data=reg_test_data)$resid
  
  test_y_true = reg_test_data$nihtbx_cryst_uncorrected
  test_y_true_partial = reg_test_data_partial$nihtbx_cryst_uncorrected
  
  covar_test_pred = predict(model_covar, newdata = reg_test_data)
  surface_test_pred = predict(model_surface, newdata = reg_test_data)
  all_test_pred = predict(model_all, newdata = reg_test_data)
  all_quad_test_pred = predict(model_all_quad, newdata = reg_test_data)
  resid_test_pred = predict(model_partial1, newdata = reg_test_data_partial)
  resid2_test_pred = predict(model_partial2, newdata = reg_test_data_partial)
  
  
  # Calculate R-squared on test data
  covar_test_r2 = c(covar_test_r2, my_R2(true=test_y_true, pred=covar_test_pred))
  surface_test_r2 = c(surface_test_r2, my_R2(true=test_y_true, pred=surface_test_pred))
  all_test_r2 = c(all_test_r2, my_R2(true=test_y_true, pred=all_test_pred))
  all_quad_test_r2 = c(all_quad_test_r2, my_R2(true=test_y_true, pred=all_quad_test_pred))
  resid_test_r2 = c(resid_test_r2, my_R2(true=test_y_true_partial, pred=resid_test_pred))
  resid2_test_r2 = c(resid2_test_r2, my_R2(true=test_y_true_partial, pred=resid2_test_pred))
}

# organize outcomes
result_dict_boot <- list(
  covar_test_r2 = covar_test_r2, surface_test_r2 = surface_test_r2, all_test_r2 = all_test_r2, all_quad_test_r2 = all_quad_test_r2,
  resid_test_r2=resid_test_r2, resid2_test_r2=resid2_test_r2
)
result_tbl_boot <- data.frame(
  variable = names(result_dict_boot),
  mean = sapply(result_dict_boot, mean),
  var_boot   = sapply(result_dict_boot, var),    # Use the first estimate ((1/J)+(n2/n1))*s2
  sd_boot =   sapply(result_dict_boot, sd)
)
result_tbl_boot[, c(2,4)] <- round(result_tbl_boot[, c(2,4)], 4)

# Print the result
print(result_tbl_boot)

save(result_tbl_boot, file="result_tbl_boot.Rdata")
0.0697+ 0.0551

##################### J and M combination simulation vis. ####################### 

# gather simulation results
load("run_output_local/method1_dict.Rdata")
load("run_output_local/method2_dict.Rdata")
load("run_output_local/result_tbl_boot.Rdata")
method1_mean <- sapply(method1_dict, function(x) x$mean[4])
method1_sd <- sapply(method1_dict, function(x) x$sd[4])
boot_mean = result_tbl_boot$mean[4]
boot_sd = result_tbl_boot$sd_boot[4]


method2_sd_tbl = matrix(nrow=length(M_list), ncol=length(J_list)) 
for (J_index in seq_along(J_list)) {
  method2_sd_tbl[,J_index]<- sapply(method2_dict[[J_index]], function(x) x[4,2])
}
colnames(method2_sd_tbl) = paste0(J_list)
rownames(method2_sd_tbl) = paste0(M_list)




#Var 1 - all
plot(J_list, method1_sd, col="violet", type="b", pch=16,
     xlab= "J", ylab="R2 Var", ylim = c(0.0, 0.02), xlim=range(J_list),
     xaxt = "n", main="Variance of sample R2")

for (M_index in 1:length(M_list)) {
  lines(J_list, method2_sd_tbl[M_index,], col=M_index)
}

lines(J_list, rep(boot_sd, length(J_list)), col="darkgreen", type="b", pch=16,)
axis(side = 1, at = J_list, labels = paste0(J_list))
legend("topright",
       legend=c("method1", paste0("method2(m=",M_list,")"), "bootstrapping"),
       fill=c( "violet", 1:length(M_list), "darkgreen")
)



#Var 2 - Ms compare
M_list1=M_list[1:3]
M_list2=M_list[4:length(M_list)]


plot(J_list, method2_sd_tbl[1,], col=1, type="l",
     xlab= "J", ylab="R2 Var", ylim = c(0.005, 0.015), xlim=range(J_list),
     xaxt = "n", main="Variance of sample R2 (different Ms)")
for (M_index in 2:3) {
  lines(J_list, method2_sd_tbl[M_index,], col=M_index)
}
axis(side = 1, at = J_list, labels = paste0(J_list))
legend("topright",
       legend=c( paste0("method2(m=",M_list1,")")),
       fill=c( 1:3)
)


plot(J_list, method2_sd_tbl[4,], col=4, type="l",
     xlab= "J", ylab="R2 Var", ylim = c(0.005, 0.015), xlim=range(J_list),
     xaxt = "n", main="Variance of sample R2 (different Ms)")

lines(J_list, method2_sd_tbl[5,], col=5)
axis(side = 1, at = J_list, labels = paste0(J_list))
legend("topright",
       legend=c( paste0("method2(m=",M_list2,")")),
       fill=c(4:5)
)




# Mean comparison
plot(J_list, method1_mean, col="purple", type="b", pch=16,
     xlab= "J", ylab="R2", ylim = c(0.12, 0.16), xlim=range(J_list),
     xaxt = "n", main="Mean of sample R2")
lines(J_list, rep(0.1392, length(J_list)), col="black", type="l", lty=2)
lines(J_list, rep(boot_mean, length(J_list)), col="darkgreen",type="b", pch=16,)


axis(side = 1, at = J_list, labels = paste0(J_list))
legend("topright", legend=c("all-data mean", "CV (method 1&2)", "bootstrapping"),
       fill=c("black", "purple", "darkgreen"))

















