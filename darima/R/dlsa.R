suppressPackageStartupMessages(require("dplyr"))
suppressPackageStartupMessages(require("magrittr"))

dlsa.comb <- function(model_mapped, sample_size){
  # model_mapped: par_id, coef, Sig_invMcoef, Sig_inv(p)
  if ("par_id" %in% colnames(model_mapped)){
    groupped_sum <- model_mapped %>% group_by(par_id) %>% summarise_all(sum)
    groupped_sum <- groupped_sum %>% arrange(groupped_sum$par_id) # ascending order
    #groupped_sum <- groupped_sum %>% arrange(desc(groupped_sum$par_id)) # descending order 
    if (dim(groupped_sum)[1] == 0){
      print("Zero-length grouped pandas DataFrame obtained, check the input.")
    } else{
      Sig_invMcoef_sum <- groupped_sum[, 3] %>% as.matrix() # p-by-1
      Sig_inv_sum <- groupped_sum[, -c(1:3)] %>% as.matrix() # p-by-p
      
      Sig_inv_sum_inv <- solve(Sig_inv_sum) # p-by-p
      
      Theta_tilde <- Sig_inv_sum_inv %*% Sig_invMcoef_sum # p-by-1
      Sig_tilde <- Sig_inv_sum_inv * sample_size # p-by-p
      
      out <- `colnames<-` (data.frame(cbind(Theta_tilde, Sig_tilde)),
                           c("Theta_tilde", colnames(model_mapped)[-c(1:3)]))
    }
  } else{
    print("ID variable 'par_id' not in input.")
  }
  return(out)
}

