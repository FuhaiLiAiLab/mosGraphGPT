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
  net_edge_weight <- read.csv(paste0('./ROSMAP-analysis/fold_0/average_attention_', type, '.csv'))
  colnames(net_edge_weight)[1] <- 'From'
  colnames(net_edge_weight)[2] <- 'To'
  
  # Read node data
  net_node <- read.csv('./ROSMAP-graph-data/map-all-gene.csv') # NODE LABEL
  
  file_name_pvalue <- sprintf('./ROSMAP-analysis/modified_p_values_20.csv')
  pvalue_df <- read.csv(file_name_pvalue)

  ### 2.1 FILTER EDGE BY [edge_weight]
  edge_threshold <- 0.23
  filter_net_edge <- dplyr::filter(net_edge_weight, Attention > edge_threshold)
  filter_net_edge_node <- unique(c(filter_net_edge$From, filter_net_edge$To))
  filter_net_node <- net_node[net_node$Gene_num %in% filter_net_edge_node, ]
  
  # Merge P_value data into filter_net_node based on Gene_name
  merged_pvalues <- pvalue_df %>%
    left_join(net_node, by = c("Gene_name" = "Gene_name")) %>%
    filter(!is.na(Gene_num)) %>%
    group_by(Gene_num) %>%
    summarize(P_values = paste(P_value, collapse = "|")) %>%
    ungroup()
  
  filter_net_node <- filter_net_node %>%
    left_join(merged_pvalues, by = c("Gene_num" = "Gene_num")) %>%
    mutate(Gene_name = ifelse(is.na(P_values), Gene_name, paste(Gene_name, P_values, sep = "|")))

  ### 2.2 FILTER WITH GIANT COMPONENT
  tmp_net <- igraph::graph_from_data_frame(d = filter_net_edge, vertices = filter_net_node, directed = FALSE)
  all_components <- igraph::groups(igraph::components(tmp_net))
  
  # COLLECT ALL LARGE COMPONENTS
  giant_comp_node <- c()
  giant_comp_threshold <- 15
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

  V(net)$size <- sapply(V(net)$Gene_name, function(name) {
    # split the name string by the pipe character
    parts <- unlist(strsplit(name, split="\\s*\\|\\s*"))
  
  # check if the string was successfully split
  if (length(parts) >= 2) {
    # print("sucessfully split")
    p_value_str <- parts[2]  # extract the p-value string
    p_value <- as.numeric(p_value_str)  # convert the p-value string to a number
    
    # check if the p-value is valid and greater than 0
    if (!is.na(p_value) && p_value > 0) {
      # print("valid p-value and format")
      return(max(3.5, min(5, -log10(p_value) * 3.5)))
    }
  }

  # if the p-value is invalid or the format is incorrect, return a default size
  # print("invalid p-value or format")
  return(2.5)
})

  vertex_fcol <- rep('black', igraph::vcount(net))
  vertex_col <- rep('mediumpurple1', igraph::vcount(net))
  vertex_col[igraph::V(net)$NodeType == 'Gene-Epi'] <- 'plum1'
  vertex_col[igraph::V(net)$NodeType == 'Gene-Geno'] <- 'skyblue1'
  vertex_col[igraph::V(net)$NodeType == 'Gene-Tran'] <- 'mediumpurple1'
  vertex_col[igraph::V(net)$NodeType == 'Gene-Prot'] <- 'gray'

  # vertex_size <- rep(5.0, igraph::vcount(net))
  vertex_cex <- rep(0.5, igraph::vcount(net))
  edge_width <- rep(0.7, igraph::ecount(net))

  edge_color <- rep('gray', igraph::ecount(net))
  edge_color[igraph::E(net)$EdgeType == 'Gene-Tran-Gene-Prot'] <- 'mediumpurple1'
  edge_color[igraph::E(net)$EdgeType == 'Gene-Tran-Gene-Geno'] <- 'skyblue1'
  edge_color[igraph::E(net)$EdgeType == 'Gene-Geno-Gene-Epi'] <- 'plum1'
  
  # Set seed for consistent layout
  set.seed(12)

  # Custom label
  vertex_label <- rep("", igraph::vcount(net))
  prot_nodes <- igraph::V(net)$NodeType == 'Gene-Prot'
  sub_gene_nodes <- igraph::V(net)$NodeType %in% c('Gene-Epi', 'Gene-Tran', 'Gene-Geno')
  vertex_label[prot_nodes] <- gsub("-Prot", "", igraph::V(net)$Gene_name[prot_nodes])
  vertex_label[sub_gene_nodes] <- ""
  
  # Plot the graph and save to PNG
  png(file = paste0('./ROSMAP-analysis/fold_0/', type, '.png'), width = 4000, height = 4000, res = 300)
  
plot(net,
     vertex.frame.width = 0.1,
     vertex.frame.color = vertex_fcol,
     vertex.color = vertex_col,
    #  vertex.size = vertex_size,
     vertex.size = V(net)$size,
    #  vertex.shape = c('square', 'circle')[1+(V(net)$NodeType=='gene')],
    #  vertex.label = V(net)$Gene_name,
     vertex.label = vertex_label,
     vertex.label.color = 'black',
     vertex.label.cex = vertex_cex,
     edge.width = edge_width,
     edge.color = edge_color,
     edge.curved = 0.2,
     layout=layout_nicely)
  
  # Add legends within the plot area
  legend('topleft', inset=c(0.05, 0.05),  # Adjusted y coordinate for upper-left corner
        legend = c( 'Protein', 'Transcription', 'Genotype', 'Epigenetic'), 
        pch = c(21, 21, 21),
        pt.bg = c( 'grey', 'mediumpurple1', 'skyblue1', 'plum1'), 
        pt.cex = 1.5, 
        cex = 1, 
        bty = 'n')
  
  legend('topleft', inset=c(0.05, 0.15),  # Adjusted y coordinate for upper-left corner
         legend = c('Protein-Protein', 'Protein-Tran', 'Tran-Geno', 'Geno-Epi'),
         col = c('gray', 'mediumpurple1', 'skyblue1', 'plum1'), 
         lwd = 10,
         pt.cex = 1.5,
         cex = 0.8,  
         bty = 'n')
  
  dev.off()

  print("Graph saved.")

  # Export gene names to CSV
  gene_names <- data.frame(Gene_name = V(net)$Gene_name)
  write.csv(gene_names, paste0('./ROSMAP-analysis/fold_0/', type, '_gene_names.csv'), row.names = FALSE)
}

# Process files for 'female' and 'male'
process_and_plot_graph('AD')
process_and_plot_graph('NOAD')


