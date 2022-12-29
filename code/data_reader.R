###################################
## GSE30931
## expr matrix
exprSet = read.table('../data/support_materials/GSE30931_series_matrix.txt', sep = '\t', comment.char = '!', fill = T, header = T)
View(exprSet)
class(exprSet)
str(exprSet)

## id mapping
library(illuminaHumanv4.db)
ids = toTable(illuminaHumanv4UNIPROT)
names(exprSet)[1] <- names(ids)[1]
exprSet$probe_id <- as.character(exprSet$probe_id)

library(dplyr)
exprSet <- exprSet %>%
  inner_join(ids, by="probe_id") %>%
  select(-probe_id) %>%
  select(uniprot_id, everything())

View(exprSet)
write.csv(exprSet, "../data/support_materials/GSE30931_exprSet.csv")
###################################

###################################
## GSE31057
#exprSet_GSE31057 = read.table('../data/support_materials/GSE31057_series_matrix.txt', sep = '\t', comment.char = '!', fill = T, header = T)
#View(exprSet_GSE31057)
#class(exprSet_GSE31057)
#str(exprSet_GSE31057)
#
#library(illuminaHumanv3.db)
#ids = toTable(illuminaHumanv3UNIPROT)
#names(exprSet_GSE31057)[1] <- names(ids)[1]
#exprSet_GSE31057$probe_id <- as.character(exprSet_GSE31057$probe_id)
#
#library(dplyr)
#exprSet_GSE31057 <- exprSet_GSE31057 %>%
#  inner_join(ids, by="probe_id") %>%
#  dplyr::select(-probe_id) %>%
#  dplyr::select(uniprot_id, everything())
#
#write.csv(exprSet_GSE31057, "../data/support_materials/GSE31057_exprSet.csv")
###################################

###################################
## GSE27182
exprSet_GSE27182 = read.table('../data/support_materials/GSE27182_series_matrix.txt', sep = '\t', comment.char = '!', fill = T, header = T)
View(exprSet_GSE27182)
class(exprSet_GSE27182)
str(exprSet_GSE27182)

library(hgu133plus2.db)
ids = toTable(hgu133plus2UNIPROT)
names(exprSet_GSE27182)[1] <- names(ids)[1]
exprSet_GSE27182$probe_id <- as.character(exprSet_GSE27182$probe_id)

library(dplyr)
exprSet_GSE27182 <- exprSet_GSE27182 %>%
  inner_join(ids, by="probe_id") %>%
  dplyr::select(-probe_id) %>%
  dplyr::select(uniprot_id, everything())

write.csv(exprSet_GSE27182, "../data/support_materials/GSE27182_exprSet.csv")
###################################

###################################
## GSE74572
exprSet_GSE74572 = read.table('../data/support_materials/GSE74572_series_matrix.txt', sep = '\t', comment.char = '!', fill = T, header = T)
View(exprSet_GSE74572)
class(exprSet_GSE74572)
str(exprSet_GSE74572)

library(hgu133plus2.db)
ids = toTable(hgu133plus2UNIPROT)
names(exprSet_GSE74572)[1] <- names(ids)[1]
exprSet_GSE74572$probe_id <- as.character(exprSet_GSE74572$probe_id)

library(dplyr)
exprSet_GSE74572 <- exprSet_GSE74572 %>%
  inner_join(ids, by="probe_id") %>%
  dplyr::select(-probe_id) %>%
  dplyr::select(uniprot_id, everything())

write.csv(exprSet_GSE74572, "../data/support_materials/GSE74572_exprSet.csv")
###################################
