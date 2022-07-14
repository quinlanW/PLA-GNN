## expr matrix
exprSet = read.table('../data/support_materials/GSE30931_series_matrix.txt', sep = '\t', comment.char = '!', fill = T, header = T)
View(exprSet)
class(exprSet)
str(exprSet)

## box plot
library(reshape2)
group_list = c(rep('control', 3), rep('Estrogen', 3), rep('Velcade', 3), rep('Estrogen (6 h) + Velcade', 3))
exprSet_L = melt(exprSet)
colnames(exprSet_L) = c('probe', 'sample', 'value')
exprSet_L$group = rep(group_list, each=nrow(exprSet))
head(exprSet_L)

library(ggplot2)
p = ggplot(exprSet_L, aes(x=sample, y=value, fill=group)) + geom_boxplot() + theme(axis.text.x = element_text(angle = 45, hjust = 0.5, vjust = 0.5))
print(p)
ggsave(
  filename = "../data/support_materials/GSE30931_boxPlot.png",
  width = 10,
  height = 10,
  units = 'in',
  dpi = 500
)


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

## PCA
library(ggfortify)
df = as.data.frame(t(exprSet))
df$group = group_list
png(filename = "../data/support_materials/GSE30931_PCA.png")
autoplot(prcomp(df[,1:(ncol(df)-1)]), data=df, colour='group')
dev.off()

