library(tidyverse)
library(gridExtra)
library(grid)
library(plyr)
library(ggsci)
source("themes.R")
gg_color_hue <- function(n) {
  hues = seq(15, 375, length = n + 1)
  hcl(h = hues, l = 65, c = 100)[1:n]
}

data_raw1 <- read_csv("./data/combined_data_clean.csv", col_types = cols()) %>% select(-1)
data_raw2 <- read_csv("./data/combined_data_clean2.csv", col_types = cols()) %>% select(-1, -2)

# id_diff <- setdiff(data_raw1$id, data_raw2$id)
# setdiff(data_raw1$id, c(data_raw2$id, id_diff))
data_raw <- data_raw1
data_raw$label <- data_raw$covid_status == 'healthy' %>% as.character(.)
data_raw$label[data_raw$label == TRUE] <- "Healthy"
data_raw$label[data_raw$label == FALSE] <- "Unhealthy"

### 1- Descriptive Statics ----
# distribution of the age for Healthy and Unhealthy group
desc_plot1 <- ggplot() + 
  geom_histogram(data=data_raw, aes(x=a, y=..density.., fill=label), binwidth=1,alpha=0.6,position='identity') + 
  geom_density(data=data_raw ,aes(x=a, color=label),size=1,alpha=0.2) + 
  ggtitle(NULL)+xlab("Age")+ylab('Density') + 
  theme_sd1()

cdat <- ddply(data_raw, "label", summarise, AF.mean=mean(a))
cdat

desc_plot1 <- desc_plot1 +
  geom_vline(data=cdat, aes(xintercept=AF.mean,color=label),linetype="dashed", size=1) + 
  #scale_color_manual(name=NULL, values=c('Healthy' = '#f4a582','Unhealthy' = '#92c5de')) + 
  #scale_fill_manual(name=NULL, values=c('Healthy' = '#f4a582','Unhealthy' = '#92c5de')) +
  scale_color_npg() + 
  scale_fill_npg() +
  scale_y_continuous(expand = c(0, 0), limits=c(0, 0.076), breaks = seq(0, 0.075, 0.015)) + 
  scale_x_continuous(expand = c(0, 0), limits = c(0, 92), breaks = seq(0, 90, 15))
desc_plot1

figure_1<-rbind(ggplotGrob(desc_plot1),size="first")
ggsave(file="./figures/distri_label_age.pdf", plot=figure_1, bg = 'white', width = 15, height = 11, units = 'cm', dpi = 600)

# pie chart for several features 
count.data1 <- as.data.frame(table(data_raw$covid_status))
count.data1 <- count.data1[order(-count.data1$Freq),]
count.data1 <- count.data1 %>%
  mutate(lab.ypos = cumsum(Freq) - 0.5*Freq) %>%
  mutate(percent = Freq/sum(count.data1$Freq)) 
count.data1

order <- c(1:nrow(count.data1))
desc_plot2 <- ggplot(count.data1, aes(x = "", y = Freq, fill = Var1)) +
  geom_bar(width = 1, size=1,stat = "identity", color = "white",alpha=0.85) +
  coord_polar("y", start = 0)+
  #geom_text(aes(x=1.5,y = lab.ypos, label = paste0(Freq," (",round(percent,digits = 2),")") ), color = "black")+
  #scale_fill_manual(name=NULL,values=gg_color_hue(7)) +
  scale_fill_npg() + 
  theme_classic() +
  theme(axis.line = element_blank(),
        axis.text = element_blank(),
        axis.ticks = element_blank(),
        plot.title = element_blank(),
        axis.title = element_blank(), 
        legend.title = element_blank(),
        panel.background=element_rect(fill='transparent',color='black',size=1),plot.margin=unit(c(0.5,1,0.5,1),'lines'),
        legend.key.width=unit(0.6,'cm'),legend.key.height=unit(0.6,'cm'),legend.key.size = unit(5, 'lines'),legend.key = element_rect(size = 0.1, color = NA),
        legend.position='right',legend.text=element_text(size=16,face='bold.italic'),legend.margin=margin(t=0.1,r=0.1,b=0,l=0.1,unit='cm')) 
desc_plot2
figure_2<-rbind(ggplotGrob(desc_plot2),size="first")
ggsave(file="./figures/pie_covid_status.pdf", plot=figure_2, bg = 'white', width = 20, height = 10, units = 'cm', dpi = 600)

# pie chart for two labels
count.data2 <- as.data.frame(table(data_raw$label))
count.data2 <- count.data2[order(-count.data2$Freq),]
count.data2 <- count.data2 %>%
  mutate(lab.ypos = cumsum(Freq) - 0.5*Freq) %>%
  mutate(percent = Freq/sum(count.data2$Freq)) 
count.data2

order <- c(1:nrow(count.data2))
desc_plot3 <- ggplot(count.data2, aes(x = "", y = Freq, fill = Var1)) +
  geom_bar(width = 1, size=1,stat = "identity", color = "white",alpha=0.85) +
  coord_polar("y", start = 0)+
  #geom_text(aes(x=0.8,y = lab.ypos, label = paste0(Freq," (",round(percent,digits = 2),")") ), color = "black")+
  #scale_fill_manual(name=NULL,values=gg_color_hue(7)) +
  scale_fill_npg() + 
  theme_classic() +
  theme(axis.line = element_blank(),
        axis.text = element_blank(),
        axis.ticks = element_blank(),
        plot.title = element_blank(),
        axis.title = element_blank(), 
        legend.title = element_blank(),
        panel.background=element_rect(fill='transparent',color='black',size=1),plot.margin=unit(c(0.5,1,0.5,1),'lines'),
        legend.key.width=unit(0.6,'cm'),legend.key.height=unit(0.6,'cm'),legend.key.size = unit(5, 'lines'),legend.key = element_rect(size = 0.1, color = NA),
        legend.position='right',legend.text=element_text(size=16,face='bold.italic'),legend.margin=margin(t=0.1,r=0.1,b=0,l=0.1,unit='cm')) 
desc_plot3
figure_3<-rbind(ggplotGrob(desc_plot3),size="first")
ggsave(file="./figures/pie_label.pdf", plot=figure_3, bg = 'white', width = 20, height = 10, units = 'cm', dpi = 600)

# pie chart for gender
count.data3 <- as.data.frame(table(data_raw$g))
count.data3 <- count.data3[order(-count.data3$Freq),]
count.data3 <- count.data3 %>%
  mutate(lab.ypos = cumsum(Freq) - 0.6*Freq) %>%
  mutate(percent = Freq/sum(count.data2$Freq)) 
count.data3

order <- c(1:nrow(count.data3))
desc_plot4 <- ggplot(count.data3, aes(x = "", y = Freq, fill = Var1)) +
  geom_bar(width = 1, size=1,stat = "identity", color = "white",alpha=0.85) +
  coord_polar("y", start = 0)+
  geom_text(aes(x=1,y = lab.ypos, label = paste0(Freq," (",round(percent,digits = 2),")") ), color = "black")+
  #scale_fill_manual(name=NULL,values=gg_color_hue(7)) +
  scale_fill_npg() + 
  theme_classic() +
  theme(axis.line = element_blank(),
        axis.text = element_blank(),
        axis.ticks = element_blank(),
        plot.title = element_blank(),
        axis.title = element_blank(), 
        legend.title = element_blank(),
        panel.background=element_rect(fill='transparent',color='black',size=1),plot.margin=unit(c(0.5,1,0.5,1),'lines'),
        legend.key.width=unit(0.6,'cm'),legend.key.height=unit(0.6,'cm'),legend.key.size = unit(5, 'lines'),legend.key = element_rect(size = 0.1, color = NA),
        legend.position='right',legend.text=element_text(size=16,face='bold.italic'),legend.margin=margin(t=0.1,r=0.1,b=0,l=0.1,unit='cm')) 
desc_plot4
figure_4<-rbind(ggplotGrob(desc_plot4),size="first")
ggsave(file="./figures/pie_gender.pdf", plot=figure_4, bg = 'white', width = 18, height = 10, units = 'cm', dpi = 600)

# distribution for 5 labels
desc_plot5 <- ggplot() + 
  geom_histogram(data=data_raw, aes(x=a, y=..density.., fill=covid_status), binwidth=1,alpha=0.6,position='identity') + 
  geom_density(data=data_raw ,aes(x=a, color=covid_status),size=1,alpha=0.2) + 
  ggtitle(NULL)+xlab("Age")+ylab('Density') + 
  theme_sd2()

cdat <- ddply(data_raw, "covid_status", summarise, AF.mean=mean(a))
cdat

desc_plot5 <- desc_plot5 +
  geom_vline(data=cdat, aes(xintercept=AF.mean,color=covid_status),linetype="dashed", size=1) + 
  #scale_color_manual(name=NULL, values=c('Healthy' = '#f4a582','Unhealthy' = '#92c5de')) + 
  #scale_fill_manual(name=NULL, values=c('Healthy' = '#f4a582','Unhealthy' = '#92c5de')) +
  scale_color_npg() + 
  scale_fill_npg() +
  scale_y_continuous(expand = c(0, 0), limits=c(0, 0.076), breaks = seq(0, 0.075, 0.015)) + 
  scale_x_continuous(expand = c(0, 0), limits = c(0, 92), breaks = seq(0, 90, 15))
desc_plot5

figure_5<-rbind(ggplotGrob(desc_plot5),size="first")
ggsave(file="./figures/distri_status_age.pdf", plot=figure_5, bg = 'white', width = 25, height = 11, units = 'cm', dpi = 600)


### correlation analysis part ----
library('DescTools')
library('rcompanion')
df_ana <- data_raw[, c(2, 3, 5, 6, 7, 11:15, 19:22, 25:37)]

df_cor <- df_ana[, c('covid_status', 'label', 'ep', 'asthma', 'smoker',
                     'diabetes', 'cough', 'ht', 'cold', 'diarrhoea', 'ihd')]
df_cor$asthma[is.na(df_cor$asthma)] <- 'False'
df_cor$smoker[is.na(df_cor$smoker)] <- 'False'
df_cor$diabetes[is.na(df_cor$diabetes)] <- 'False'
df_cor$cough[is.na(df_cor$cough)] <- 'False'
df_cor$ht[is.na(df_cor$ht)] <- 'False'
df_cor$cold[is.na(df_cor$cold)] <- 'False'
df_cor$diarrhoea[is.na(df_cor$diarrhoea)] <- 'False'
df_cor$ihd[is.na(df_cor$ihd)] <- 'False'

res_cor <- data.frame()

X_squared1 <- chisq.test(df_cor$covid_status, df_cor$ep)$statistic
X_squared2 <- chisq.test(df_cor$label, df_cor$ep)$statistic
C_contin1 <- ContCoef(df_cor$covid_status, df_cor$ep, correct = TRUE)
C_contin2 <- ContCoef(df_cor$label, df_cor$ep, correct = TRUE)
V_cramer1 <- cramerV(df_cor$covid_status, df_cor$ep, bias.correct = TRUE)
V_cramer2 <- cramerV(df_cor$label, df_cor$ep, bias.correct = TRUE)
U_uncertainty1 <- UncertCoef(df_cor$covid_status, df_cor$ep, direction = "column")
U_uncertainty2 <- UncertCoef(df_cor$label, df_cor$ep, direction = "column")
tmp <- data.frame(feature = rep('ep', 8), 
                  num = c(X_squared1, X_squared2, C_contin1, C_contin2,
                          V_cramer1, V_cramer2, U_uncertainty1, U_uncertainty2), 
                  method = c('X_squared', 'X_squared', 'C_Contingency_coef', 'C_Contingency_coef',
                             'V_Cramer', 'V_Cramer', 'U_Thiel', 'U_Thiel'), 
                  group = rep(c('covid_status', 'label'), 4))
res_cor <- rbind(res_cor, tmp)

X_squared1 <- chisq.test(df_cor$covid_status, df_cor$asthma)$statistic
X_squared2 <- chisq.test(df_cor$label, df_cor$asthma)$statistic
C_contin1 <- ContCoef(df_cor$covid_status, df_cor$asthma, correct = TRUE)
C_contin2 <- ContCoef(df_cor$label, df_cor$asthma, correct = TRUE)
V_cramer1 <- cramerV(df_cor$covid_status, df_cor$asthma, bias.correct = TRUE)
V_cramer2 <- cramerV(df_cor$label, df_cor$asthma, bias.correct = TRUE)
U_uncertainty1 <- UncertCoef(df_cor$covid_status, df_cor$asthma, direction = "column")
U_uncertainty2 <- UncertCoef(df_cor$label, df_cor$asthma, direction = "column")
tmp <- data.frame(feature = rep('asthma', 8), 
                  num = c(X_squared1, X_squared2, C_contin1, C_contin2,
                          V_cramer1, V_cramer2, U_uncertainty1, U_uncertainty2), 
                  method = c('X_squared', 'X_squared', 'C_Contingency_coef', 'C_Contingency_coef',
                             'V_Cramer', 'V_Cramer', 'U_Thiel', 'U_Thiel'), 
                  group = rep(c('covid_status', 'label'), 4))
res_cor <- rbind(res_cor, tmp)

X_squared1 <- chisq.test(df_cor$covid_status, df_cor$smoker)$statistic
X_squared2 <- chisq.test(df_cor$label, df_cor$smoker)$statistic
C_contin1 <- ContCoef(df_cor$covid_status, df_cor$smoker, correct = TRUE)
C_contin2 <- ContCoef(df_cor$label, df_cor$smoker, correct = TRUE)
V_cramer1 <- cramerV(df_cor$covid_status, df_cor$smoker, bias.correct = TRUE)
V_cramer2 <- cramerV(df_cor$label, df_cor$smoker, bias.correct = TRUE)
U_uncertainty1 <- UncertCoef(df_cor$covid_status, df_cor$smoker, direction = "column")
U_uncertainty2 <- UncertCoef(df_cor$label, df_cor$smoker, direction = "column")
tmp <- data.frame(feature = rep('smoker', 8), 
                  num = c(X_squared1, X_squared2, C_contin1, C_contin2,
                          V_cramer1, V_cramer2, U_uncertainty1, U_uncertainty2), 
                  method = c('X_squared', 'X_squared', 'C_Contingency_coef', 'C_Contingency_coef',
                             'V_Cramer', 'V_Cramer', 'U_Thiel', 'U_Thiel'), 
                  group = rep(c('covid_status', 'label'), 4))
res_cor <- rbind(res_cor, tmp)

X_squared1 <- chisq.test(df_cor$covid_status, df_cor$cough)$statistic
X_squared2 <- chisq.test(df_cor$label, df_cor$cough)$statistic
C_contin1 <- ContCoef(df_cor$covid_status, df_cor$cough, correct = TRUE)
C_contin2 <- ContCoef(df_cor$label, df_cor$cough, correct = TRUE)
V_cramer1 <- cramerV(df_cor$covid_status, df_cor$cough, bias.correct = TRUE)
V_cramer2 <- cramerV(df_cor$label, df_cor$cough, bias.correct = TRUE)
U_uncertainty1 <- UncertCoef(df_cor$covid_status, df_cor$cough, direction = "column")
U_uncertainty2 <- UncertCoef(df_cor$label, df_cor$cough, direction = "column")
tmp <- data.frame(feature = rep('cough', 8), 
                  num = c(X_squared1, X_squared2, C_contin1, C_contin2,
                          V_cramer1, V_cramer2, U_uncertainty1, U_uncertainty2), 
                  method = c('X_squared', 'X_squared', 'C_Contingency_coef', 'C_Contingency_coef',
                             'V_Cramer', 'V_Cramer', 'U_Thiel', 'U_Thiel'), 
                  group = rep(c('covid_status', 'label'), 4))
res_cor <- rbind(res_cor, tmp)

X_squared1 <- chisq.test(df_cor$covid_status, df_cor$ht)$statistic
X_squared2 <- chisq.test(df_cor$label, df_cor$ht)$statistic
C_contin1 <- ContCoef(df_cor$covid_status, df_cor$ht, correct = TRUE)
C_contin2 <- ContCoef(df_cor$label, df_cor$ht, correct = TRUE)
V_cramer1 <- cramerV(df_cor$covid_status, df_cor$ht, bias.correct = TRUE)
V_cramer2 <- cramerV(df_cor$label, df_cor$ht, bias.correct = TRUE)
U_uncertainty1 <- UncertCoef(df_cor$covid_status, df_cor$ht, direction = "column")
U_uncertainty2 <- UncertCoef(df_cor$label, df_cor$ht, direction = "column")
tmp <- data.frame(feature = rep('ht', 8), 
                  num = c(X_squared1, X_squared2, C_contin1, C_contin2,
                          V_cramer1, V_cramer2, U_uncertainty1, U_uncertainty2), 
                  method = c('X_squared', 'X_squared', 'C_Contingency_coef', 'C_Contingency_coef',
                             'V_Cramer', 'V_Cramer', 'U_Thiel', 'U_Thiel'), 
                  group = rep(c('covid_status', 'label'), 4))
res_cor <- rbind(res_cor, tmp)


X_squared1 <- chisq.test(df_cor$covid_status, df_cor$cold)$statistic
X_squared2 <- chisq.test(df_cor$label, df_cor$cold)$statistic
C_contin1 <- ContCoef(df_cor$covid_status, df_cor$cold, correct = TRUE)
C_contin2 <- ContCoef(df_cor$label, df_cor$cold, correct = TRUE)
V_cramer1 <- cramerV(df_cor$covid_status, df_cor$cold, bias.correct = TRUE)
V_cramer2 <- cramerV(df_cor$label, df_cor$cold, bias.correct = TRUE)
U_uncertainty1 <- UncertCoef(df_cor$covid_status, df_cor$cold, direction = "column")
U_uncertainty2 <- UncertCoef(df_cor$label, df_cor$cold, direction = "column")
tmp <- data.frame(feature = rep('cold', 8), 
                  num = c(X_squared1, X_squared2, C_contin1, C_contin2,
                          V_cramer1, V_cramer2, U_uncertainty1, U_uncertainty2), 
                  method = c('X_squared', 'X_squared', 'C_Contingency_coef', 'C_Contingency_coef',
                             'V_Cramer', 'V_Cramer', 'U_Thiel', 'U_Thiel'), 
                  group = rep(c('covid_status', 'label'), 4))
res_cor <- rbind(res_cor, tmp)

X_squared1 <- chisq.test(df_cor$covid_status, df_cor$ihd)$statistic
X_squared2 <- chisq.test(df_cor$label, df_cor$ihd)$statistic
C_contin1 <- ContCoef(df_cor$covid_status, df_cor$ihd, correct = TRUE)
C_contin2 <- ContCoef(df_cor$label, df_cor$ihd, correct = TRUE)
V_cramer1 <- cramerV(df_cor$covid_status, df_cor$ihd, bias.correct = TRUE)
V_cramer2 <- cramerV(df_cor$label, df_cor$ihd, bias.correct = TRUE)
U_uncertainty1 <- UncertCoef(df_cor$covid_status, df_cor$ihd, direction = "column")
U_uncertainty2 <- UncertCoef(df_cor$label, df_cor$ihd, direction = "column")
tmp <- data.frame(feature = rep('ihd', 8), 
                  num = c(X_squared1, X_squared2, C_contin1, C_contin2,
                          V_cramer1, V_cramer2, U_uncertainty1, U_uncertainty2), 
                  method = c('X_squared', 'X_squared', 'C_Contingency_coef', 'C_Contingency_coef',
                             'V_Cramer', 'V_Cramer', 'U_Thiel', 'U_Thiel'), 
                  group = rep(c('covid_status', 'label'), 4))
res_cor <- rbind(res_cor, tmp)
colnames(res_cor) <- c("feature", "num", "method", "group")
corplot1_df <- res_cor %>%  filter(method == 'X_squared')
corplot1 <- ggplot(data = corplot1_df) +
  geom_bar(aes(x=reorder(feature, -num), y=num, fill = group), stat='identity', position = 'dodge') +
  scale_fill_npg() + 
  theme_sd1() +
  ggtitle(NULL) + xlab("Features") + ylab("Chi-squared test parameter") +
  theme(legend.position =		c(0.85, 0.8),
        axis.text.x = element_text(angle = 15, vjust = .5, hjust=.35))
corplot1
figure_6<-rbind(ggplotGrob(corplot1),size="first")
ggsave(file="./figures/cor_X_squared.pdf", plot=figure_6, bg = 'white', width = 16, height = 11, units = 'cm', dpi = 600)

corplot2_df <- res_cor %>%  filter(method == 'C_Contingency_coef')
corplot2 <- ggplot(data = corplot2_df) +
  geom_bar(aes(x=reorder(feature, -num), y=num, fill = group), stat='identity', position = 'dodge') +
  scale_fill_npg() + 
  theme_sd1() +
  ggtitle(NULL) + xlab("Features") + ylab("Contingency coefficient") +
  theme(legend.position =		c(0.85, 0.8),
        axis.text.x = element_text(angle = 15, vjust = .5, hjust=.35))
corplot2
figure_7<-rbind(ggplotGrob(corplot2),size="first")
ggsave(file="./figures/cor_C_contin.pdf", plot=figure_7, bg = 'white', width = 16, height = 11, units = 'cm', dpi = 600)


corplot3_df <- res_cor %>%  filter(method == 'V_Cramer')
corplot3 <- ggplot(data = corplot3_df) +
  geom_bar(aes(x=reorder(feature, -num), y=num, fill = group), stat='identity', position = 'dodge') +
  scale_fill_npg() + 
  theme_sd1() +
  ggtitle(NULL) + xlab("Features") + ylab("Contingency coefficient") +
  theme(legend.position =		c(0.85, 0.8),
        axis.text.x = element_text(angle = 15, vjust = .5, hjust=.35))
corplot3
figure_8<-rbind(ggplotGrob(corplot3),size="first")
ggsave(file="./figures/cor_V_Cramer.pdf", plot=figure_8, bg = 'white', width = 16, height = 11, units = 'cm', dpi = 600)

corplot4_df <- res_cor %>%  filter(method == 'U_Thiel')
corplot4 <- ggplot(data = corplot4_df) +
  geom_bar(aes(x=reorder(feature, -num), y=num, fill = group), stat='identity', position = 'dodge') +
  scale_fill_npg() + 
  theme_sd1() +
  ggtitle(NULL) + xlab("Features") + ylab("Thiel’s U correlation coefficient") +
  theme(legend.position =		c(0.85, 0.8),
        axis.text.x = element_text(angle = 15, vjust = .5, hjust=.35))
corplot4
figure_9<-rbind(ggplotGrob(corplot4),size="first")
ggsave(file="./figures/cor_U_Thiel.pdf", plot=figure_9, bg = 'white', width = 16, height = 11, units = 'cm', dpi = 600)

figure_10a <- cbind(ggplotGrob(corplot1), ggplotGrob(corplot2), size = "first")
figure_10b <- cbind(ggplotGrob(corplot3), ggplotGrob(corplot4), size = "first")
figure_10 <- rbind(figure_10a, figure_10b)
ggsave(file="./figures/cor_total.pdf", plot=figure_10, bg = 'white', width = 40, height = 30, units = 'cm', dpi = 600)

