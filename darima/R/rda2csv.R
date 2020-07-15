# Save data as csv
load("/Users/wangxiaoqian/Git/Git-xqnwang/dforecast/RData/gefcom2017.RData")
library(magrittr)

zones_name <- names(gefcom2017)
for(i in seq(length(gefcom2017))){
  # get series data
  demand <- gefcom2017[[i]]$x %>% as.numeric()
  time <- attr(gefcom2017[[i]]$x, "index") %>% as.character()
  series_data <- data.frame(demand = demand, time = time)
  assign(zones_name[i], series_data)
  rm(demand, time, series_data)
  
  # save series as csv
  file_name <- paste("/Users/wangxiaoqian/Git/Git-xqnwang/dforecast/darima/data/",
                     zones_name[i], "_train.csv", sep = "")
  write.csv(get(zones_name[i]), file = file_name, row.names = FALSE)
}

for(i in seq(length(gefcom2017))){
  # get series data
  demand <- gefcom2017[[i]]$xx %>% as.numeric()
  time <- attr(gefcom2017[[i]]$xx, "index") %>% as.character()
  series_data <- data.frame(demand = demand, time = time)
  assign(zones_name[i], series_data)
  rm(demand, time, series_data)
  
  # save series as csv
  file_name <- paste("/Users/wangxiaoqian/Git/Git-xqnwang/dforecast/darima/data/",
                     zones_name[i], "_test.csv", sep = "")
  write.csv(get(zones_name[i]), file = file_name, row.names = FALSE)
}
