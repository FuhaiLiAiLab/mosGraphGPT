library(shiny)
library(dplyr)
library(igraph)
library(networkD3)
library(kdensity)
library(ggplot2)
library(Jmisc)

# Function to round a dataframe
round_df <- function(df, digits) {
  nums <- vapply(df, is.numeric, FUN.VALUE = logical(1))
  df[, nums] <- round(df[, nums], digits = digits)
  df
}

# Function to process and plot graph
process_and_plot_graph <- function(type) {
  # Set working directory
  setwd('E:/LabWork/mosFoundation2')
  # setwd('C:/Users/hemingzhang/Documents/vs-files/mosFoundation')
  
  # Read network edge weight data
  net_edge_weight <- read.csv(paste0('./ROSMAP-analysis/fold_0/filtered_average_attention_', type, '.csv'))
  colnames(net_edge_weight)[1] <- 'From'
  colnames(net_edge_weight)[2] <- 'To'
  
  # Read node data
  net_node <- read.csv('./ROSMAP-graph-data/map-all-gene.csv') # NODE LABEL
  
  ### 2.1 FILTER EDGE BY [edge_weight]
  edge_threshold <- 0.15
  filter_net_edge <- dplyr::filter(net_edge_weight, Attention > edge_threshold)
  filter_net_edge_node <- unique(c(filter_net_edge$From, filter_net_edge$To))
  filter_net_node <- net_node[net_node$Gene_num %in% filter_net_edge_node, ]
  
  ### 2.2 FILTER WITH GIANT COMPONENT
  tmp_net <- igraph::graph_from_data_frame(d = filter_net_edge, vertices = filter_net_node, directed = FALSE)
  all_components <- igraph::groups(igraph::components(tmp_net))
  
  # COLLECT ALL LARGE COMPONENTS
  giant_comp_node <- c()
  giant_comp_threshold <- 20
  for (x in 1:length(all_components)) {
    each_comp <- all_components[[x]]
    if (length(each_comp) >= giant_comp_threshold) {
      giant_comp_node <- c(giant_comp_node, each_comp)
    }
  }
  
  refilter_net_edge <- subset(filter_net_edge, (From %in% giant_comp_node | To %in% giant_comp_node))
  refilter_net_edge_node <- unique(c(refilter_net_edge$From, refilter_net_edge$To))
  refilter_net_node <- filter_net_node[filter_net_node$Gene_num %in% refilter_net_edge_node, ]
  
  ### 3. BUILD UP GRAPH
  net <- igraph::graph_from_data_frame(d = refilter_net_edge, vertices = refilter_net_node, directed = FALSE)
  
  # Network Parameters Settings
  vertex_fcol <- rep('black', igraph::vcount(net))
  vertex_col <- rep('mediumpurple1', igraph::vcount(net))
  vertex_col[igraph::V(net)$NodeType == 'Gene-METH'] <- 'plum1'
  vertex_col[igraph::V(net)$NodeType == 'Gene-PROT'] <- 'gray'
  vertex_size <- rep(5.0, igraph::vcount(net))
  vertex_cex <- rep(0.5, igraph::vcount(net))
  edge_width <- rep(2, igraph::ecount(net))
  edge_color <- rep('gray', igraph::ecount(net))
  edge_color[igraph::E(net)$EdgeType == 'Gene-TRAN-Gene-PROT'] <- 'mediumpurple1'
  edge_color[igraph::E(net)$EdgeType == 'Gene-TRAN-Gene-METH'] <- 'plum1'
  
  # Set seed for consistent layout
  set.seed(10)
  
  # Plot the graph and save to PNG
  png(file = paste0('./ROSMAP-analysis/fold_0/', type, '.png'), width = 3000, height = 3000, res = 300)
  
plot(net,
     vertex.frame.width = 0.1,
     vertex.frame.color = vertex_fcol,
     vertex.color = vertex_col,
     vertex.size = vertex_size,
    #  vertex.shape = c('square', 'circle')[1+(V(net)$NodeType=='gene')],
     vertex.label = V(net)$Gene_name,
     vertex.label.color = 'black',
     vertex.label.cex = vertex_cex,
     edge.width = edge_width,
     edge.color = edge_color,
     edge.curved = 0.2,
     layout=layout_nicely)
  
  # Add legends within the plot area using normalized coordinates
  legend('topleft', inset=c(0.02, 0.05), 
         legend = c('Genes', 'Promoters', 'Proteins'), 
         pch = c(21, 21, 21),
         pt.bg = c('mediumpurple1', 'plum1', 'gray'), 
         pt.cex = 2, 
         cex = 1.2, 
         bty = 'n')
  
  legend('topleft', inset=c(0.02, 0.2), 
         legend = c('Protein-Protein', 'Gene-Protein', 'Promoter-Gene'),
         col = c('gray', 'mediumpurple1', 'plum1'), 
         lwd = 10, 
         cex = 1.2,  
         bty = 'n')
  
  dev.off()

  # Export gene names to CSV
  gene_names <- data.frame(Gene_name = V(net)$Gene_name)
  write.csv(gene_names, paste0('./ROSMAP-analysis/fold_0/', type, '_gene_names.csv'), row.names = FALSE)
}

# Process files for 'female' and 'male'
process_and_plot_graph('AD')
process_and_plot_graph('NOAD')


