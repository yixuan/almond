library(showtext)
library(ggplot2)
library(scales)

# ALMOND
gdat = read.table("../results/mnist-2d.txt", header = FALSE, sep = " ")
gdat$V3 = factor(gdat$V3, levels = 0:9)

g = ggplot(gdat, aes(x = V1, y = V2)) +
    geom_point(aes(color = V3), alpha = 0.5) +
    xlab("Latent Dimension 1") + ylab("Latent Dimension 2") +
    scale_color_hue("True Labels") +
    coord_fixed(xlim = c(-3.5, 3.5), ylim = c(-3.5, 3.5)) +
    guides(color = guide_legend(keyheight = 2,
                                override.aes = list(size = 8, alpha = 1))) +
    theme_bw(base_size = 20, base_family = "source")

font_add_google("Source Sans Pro", "source")
showtext_auto()
ggsave("../results/mnist-2d.pdf", g, width = 9, height = 7)

# PCA
gdat = read.table("../results/mnist-2d-pca.txt", header = FALSE, sep = " ")
gdat$V3 = factor(gdat$V3, levels = 0:9)

g = ggplot(gdat, aes(x = V1, y = V2)) +
    geom_point(aes(color = V3), alpha = 0.5) +
    xlab("Latent Dimension 1") + ylab("Latent Dimension 2") +
    scale_color_hue("True Labels") +
    coord_fixed(xlim = c(-4, 8), ylim = c(-6, 6)) +
    guides(color = FALSE) +
    theme_bw(base_size = 20, base_family = "source")

font_add_google("Source Sans Pro", "source")
showtext_auto()
ggsave("../results/mnist-2d-pca.pdf", g, width = 7, height = 7)

