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

# top_values <- c(20, 30, 50, 70)
top_values <- c(100)

for (top in top_values) {
print(top)

  # Set seed for consistent layout
  set.seed(10)

# Function to process and plot graph
process_and_plot_graph <- function(type) {
  # Set working directory
  # setwd('E:/LabWork/mosFoundation2')
  setwd('C:/Users/hemingzhang/Documents/vs-files/mosFoundation')

  # Read network edge weight data
  net_edge_weight <- read.csv(paste0('./ROSMAP-analysis/fold_0/average_attention_', type, '_final_', top,'.csv'))
  # net_edge_weight <- read.csv(paste0('./ROSMAP-analysis/fold_0/average_attention_', type, '_ht.csv'))
  colnames(net_edge_weight)[1] <- 'From'
  colnames(net_edge_weight)[2] <- 'To'
  
  # Read node data
  net_node <- read.csv('./ROSMAP-graph-data/map-all-gene.csv') # NODE LABEL
  
  file_name_pvalue <- sprintf('./ROSMAP-analysis/modified_p_values_%d.csv', top)
  pvalue_df <- read.csv(file_name_pvalue)

  # ## 2.1 FILTER EDGE BY [edge_weight] (Pre filtered in notebook)
  # edge_threshold <- 0.3
  # filter_net_edge <- dplyr::filter(net_edge_weight, Attention > edge_threshold)
  # filter_net_edge_node <- unique(c(filter_net_edge$From, filter_net_edge$To))
  # filter_net_node <- net_node[net_node$Gene_num %in% filter_net_edge_node, ]
  
  filter_net_edge <- net_edge_weight
  # print(filter_net_edge)
  # write.csv(filter_net_edge, "filter_net_edge.csv", row.names = FALSE)
  filter_net_edge_node <- unique(c(filter_net_edge$From, filter_net_edge$To))
  filter_net_node <- net_node[net_node$Gene_num %in% filter_net_edge_node, ]

  ### 2.2 FILTER WITH GIANT COMPONENT
  tmp_net <- igraph::graph_from_data_frame(d = filter_net_edge, vertices = filter_net_node, directed = FALSE)
  all_components <- igraph::groups(igraph::components(tmp_net))
  
  # COLLECT ALL LARGE COMPONENTS
  # Warning: Too large threshold (~20) could lead to missing nodes

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

  # refilter_net_edge <- filter_net_edge 
  # refilter_net_edge_node <- unique(c(refilter_net_edge$From, refilter_net_edge$To))
  # refilter_net_node <- filter_net_node[filter_net_node$Gene_num %in% refilter_net_edge_node, ]

  ### 3. ADD Epi AND Tran NODES
  additional_edges <- data.frame(From = character(), To = character(), EdgeType = character(), stringsAsFactors = FALSE)
  for (gene in pvalue_df$Gene_name) {
    
    base_gene_name <- sub("-.*", "", gene)
    epi_node_name <- paste0(base_gene_name, "-Epi")
    geno_node_name <- paste0(base_gene_name, "-Geno")
    tran_node_name <- paste0(base_gene_name, "-Tran")
    prot_node_name <- paste0(base_gene_name, "-Prot")
    prot_node <- net_node$Gene_num[net_node$Gene_name == prot_node_name]
    geno_node_num <- net_node$Gene_num[net_node$Gene_name == geno_node_name]
    epi_node_num <- net_node$Gene_num[net_node$Gene_name == epi_node_name]
    tran_node_num <- net_node$Gene_num[net_node$Gene_name == tran_node_name]
    if (length(epi_node_num) > 0 && !(epi_node_num %in% refilter_net_edge_node)) {
      new_node <- data.frame(Gene_num = epi_node_num, Gene_name = epi_node_name, NodeType = "Gene-Epi", stringsAsFactors = FALSE)
      refilter_net_node <- rbind(refilter_net_node, new_node)
      refilter_net_edge_node <- c(refilter_net_edge_node, epi_node_num)
    }
    if (length(geno_node_num) > 0 && !(geno_node_num %in% refilter_net_edge_node)) {
      new_node <- data.frame(Gene_num = geno_node_num, Gene_name = geno_node_name, NodeType = "Gene-Geno", stringsAsFactors = FALSE)
      refilter_net_node <- rbind(refilter_net_node, new_node)
      refilter_net_edge_node <- c(refilter_net_edge_node, geno_node_num)
    }
    
    if (length(tran_node_num) > 0 && !(tran_node_num %in% refilter_net_edge_node)) {
      new_node <- data.frame(Gene_num = tran_node_num, Gene_name = tran_node_name, NodeType = "Gene-Tran", stringsAsFactors = FALSE)
      refilter_net_node <- rbind(refilter_net_node, new_node)
      refilter_net_edge_node <- c(refilter_net_edge_node, tran_node_num)
    }
    
    # Add edges from Tran to Prot and Epi nodes
    if (length(epi_node_num) > 0 && length(geno_node_name) > 0) {
      additional_edges <- rbind(additional_edges, data.frame(From = epi_node_name, To = geno_node_name, EdgeType = "Gene-Epi-Gene-Geno", stringsAsFactors = FALSE))
    }

    if (length(geno_node_num) > 0 && length(tran_node_num) > 0) {
      additional_edges <- rbind(additional_edges, data.frame(From = geno_node_name, To = tran_node_name, EdgeType = "Gene-Geno-Gene-Tran", stringsAsFactors = FALSE))
    }

    if (length(tran_node_num) > 0 && length(prot_node_name) > 0) {
      additional_edges <- rbind(additional_edges, data.frame(From = tran_node_name, To = prot_node_name, EdgeType = "Gene-Tran-Gene-Prot", stringsAsFactors = FALSE))
    }

    # Combine edges
    # Add edges from Tran to Prot and Epi nodes
    if (length(epi_node_num) > 0 && length(geno_node_name) > 0) {
      additional_edge <- data.frame(
        From = epi_node_num,
        To = geno_node_num,
        # Hop = NA,
        # SignalingPath = NA,
        # SpNotation = NA,
        EdgeType = "Gene-Epi-Gene-Geno",
        Attention = NA,
        stringsAsFactors = FALSE
      )
      refilter_net_edge <- rbind(refilter_net_edge, additional_edge)
    }

    if (length(geno_node_num) > 0 && length(tran_node_num) > 0) {
      additional_edge <- data.frame(
        From = geno_node_num,
        To = tran_node_num,
        # Hop = NA,
        # SignalingPath = NA,
        # SpNotation = NA,
        EdgeType = "Gene-Geno-Gene-Tran",
        Attention = NA,
        stringsAsFactors = FALSE
      )
      refilter_net_edge <- rbind(refilter_net_edge, additional_edge)
    }
    
    if (length(tran_node_num) > 0 && length(prot_node_name) > 0) {
      additional_edge <- data.frame(
        From = tran_node_num,
        To = prot_node,
        # Hop = NA,
        # SignalingPath = NA,
        # SpNotation = NA,
        EdgeType = "Gene-Tran-Gene-Prot",
        Attention = NA,
        stringsAsFactors = FALSE
      )
      refilter_net_edge <- rbind(refilter_net_edge, additional_edge)
    }
    
  }

# Add edges and nodes from pvalue with node naming logic based on category and appending names and types

for (i in 1:nrow(pvalue_df)) {
  pvalue_node <- pvalue_df[i, ]
  gene_name <- pvalue_node$Gene_name
  node_type <- pvalue_node$NodeType
  category <- pvalue_node$Category
  p_value <- pvalue_node$P_value

  # Determine corresponding node numbers for Epi, Tran, and Prot
  base_gene_name <- sub("-.*", "", gene_name)
  epi_node_name <- paste0(base_gene_name, "-Epi")
  geno_node_name <- paste0(base_gene_name, "-Geno")
  tran_node_name <- paste0(base_gene_name, "-Tran")
  prot_node_name <- paste0(base_gene_name, "-Prot")
  epi_node_num <- net_node$Gene_num[net_node$Gene_name == epi_node_name]
  geno_node_num <- net_node$Gene_num[net_node$Gene_name == geno_node_name]
  tran_node_num <- net_node$Gene_num[net_node$Gene_name == tran_node_name]
  prot_node <- net_node$Gene_num[net_node$Gene_name == prot_node_name]

  # Create edge from Geno to Tran and append category to node names and types
  if (!is.na(geno_node_num) && any(geno_node_num == net_node$Gene_num[net_node$Gene_name == gene_name & net_node$NodeType == node_type])) {
    additional_edge <- data.frame(
      From = geno_node_num,
      To = tran_node_num,
      # Hop = NA,
      # SignalingPath = NA,
      # SpNotation = NA,
      EdgeType = "Gene-Geno-Gene-Tran",
      Attention = NA,
      stringsAsFactors = FALSE
    )
    refilter_net_edge <- rbind(refilter_net_edge, additional_edge)

    # Append category and pvalue to node name and type
    index <- which(refilter_net_node$Gene_num == geno_node_num)
    current_name <- refilter_net_node$Gene_name[index]
    current_type <- refilter_net_node$NodeType[index]
    new_name <- paste(current_name, category, format(p_value, scientific = FALSE), sep=" | ")
    new_type <- paste(current_type, category, sep="-")
    refilter_net_node$Gene_name[index] <- new_name
    refilter_net_node$NodeType[index] <- new_type
  }

  # Create edge from Epi to Geno and append category to node names and types
  if (!is.na(epi_node_num) && any(epi_node_num == net_node$Gene_num[net_node$Gene_name == gene_name & net_node$NodeType == node_type])) {
    additional_edge <- data.frame(
      From = epi_node_num,
      To = geno_node_num,
      # Hop = NA,
      # SignalingPath = NA,
      # SpNotation = NA,
      EdgeType = "Gene-Geno-Gene-Epi",
      Attention = NA,
      stringsAsFactors = FALSE
    )
    refilter_net_edge <- rbind(refilter_net_edge, additional_edge)

    # Append category and pvalue to node name and type
    index <- which(refilter_net_node$Gene_num == epi_node_num)
    current_name <- refilter_net_node$Gene_name[index]
    current_type <- refilter_net_node$NodeType[index]
    new_name <- paste(current_name, category, format(p_value, scientific = FALSE), sep=" | ")
    new_type <- paste(current_type, category, sep="-")
    refilter_net_node$Gene_name[index] <- new_name
    refilter_net_node$NodeType[index] <- new_type
  }

  # Create edge from Tran to Prot and append category to node names and types
  if (!is.na(tran_node_num) && any(tran_node_num == net_node$Gene_num[net_node$Gene_name == gene_name & net_node$NodeType == node_type])) {
    additional_edge <- data.frame(
      From = tran_node_num,
      To = prot_node,
      # Hop = NA,
      # SignalingPath = NA,
      # SpNotation = NA,
      EdgeType = "Gene-Tran-Gene-Prot",
      Attention = NA,
      stringsAsFactors = FALSE
    )
    refilter_net_edge <- rbind(refilter_net_edge, additional_edge)

    # Append category and pvalue to node name and type
    index <- which(refilter_net_node$Gene_num == tran_node_num)
    current_name <- refilter_net_node$Gene_name[index]
    current_type <- refilter_net_node$NodeType[index]
    new_name <- paste(current_name, category, format(p_value, scientific = FALSE), sep=" | ")
    new_type <- paste(current_type, category, sep="-")
    refilter_net_node$Gene_name[index] <- new_name
    refilter_net_node$NodeType[index] <- new_type
  }

}

  # print("Filtered edges:")
  # print(head(refilter_net_edge))
  # print("Filtered nodes:")
  # print(head(refilter_net_node))

missing_vertices_from <- setdiff(refilter_net_edge$From, refilter_net_node$Gene_num)
missing_vertices_to <- setdiff(refilter_net_edge$To, refilter_net_node$Gene_num)
all_missing_vertices <- unique(c(missing_vertices_from, missing_vertices_to))


if (length(all_missing_vertices) > 0) {
  print("Missing vertices from edge list not listed in vertex data frame:")
  print(all_missing_vertices)
  
  # # Automatically add missing nodes
  # missing_nodes <- data.frame(
  #   Gene_num = all_missing_vertices,
  #   Gene_name = paste("Unknown_Gene", all_missing_vertices, sep="_"), # Placeholder names
  #   NodeType = "Unknown_Type" # Placeholder type
  # )
  
  # # Append the missing nodes to the node data frame
  # refilter_net_node <- rbind(refilter_net_node, missing_nodes)

  
  refilter_net_edge <- refilter_net_edge[!refilter_net_edge$From %in% all_missing_vertices & !refilter_net_edge$To %in% all_missing_vertices,]
  
  missing_gene_names <- net_node$Gene_name[net_node$Gene_num %in% all_missing_vertices]
  missing_gene_names_cleaned <- gsub("-Prot", "", missing_gene_names)
  print(missing_gene_names_cleaned)
} else {
  missing_gene_names_cleaned <- NA
  print("No missing vertices. All vertices in edge list are accounted for in vertex data frame.")
}


# Build the graph if no vertices are missing
if (length(all_missing_vertices) == 0) {
  net <- igraph::graph_from_data_frame(d = refilter_net_edge, vertices = refilter_net_node, directed = FALSE)
  # print(table(V(net)$NodeType))  

  # Network Parameters Settings

  V(net)$size <- sapply(V(net)$Gene_name, function(name) {
    # split the name string by the pipe character
    parts <- unlist(strsplit(name, split="\\s*\\|\\s*"))
  
  # check if the string was successfully split
  if (length(parts) >= 3) {
    # print("sucessfully split")
    p_value_str <- parts[3]  # extract the p-value string
    p_value <- as.numeric(p_value_str)  # convert the p-value string to a number
    
    # check if the p-value is valid and greater than 0
    if (!is.na(p_value) && p_value > 0) {
      # print("valid p-value and format")
      return(max(3.5, min(4, -log10(p_value) * 3)))
    }
  }

  # if the p-value is invalid or the format is incorrect, return a default size
  # print("invalid p-value or format")
  return(2.5)
})

  # Set size to 0 for single 'Gene-Epi' nodes
  V(net)$size[V(net)$NodeType == 'Gene-Epi'] <- 0


  # find the indices of the missing gene nodes
  if (length(all_missing_vertices) > 0) {
    missing_gene_nodes <- which(sapply(V(net)$Gene_name, function(name) any(grepl(paste(missing_gene_names_cleaned, collapse="|"), name))))
    V(net)$size[missing_gene_nodes] <- 0
    print(missing_gene_nodes)
  } 

  # vertex_size <- rep(4.0, igraph::vcount(net))
  vertex_cex <- rep(0.5, igraph::vcount(net))
  edge_width <- rep(0.7, igraph::ecount(net))

  # Assuming colors and other settings are defined earlier in the script
  # Define vertex colors and pie settings
  vertex_col <- rep('mediumpurple1', igraph::vcount(net))  # default color
  vertex_fcol <- rep('black', igraph::vcount(net))         # default font color
  vertex_fcol <- NA
  vertex_col[V(net)$size == 0] <- NA
  vertex_fcol[V(net)$size == 0] <- NA
  vertex_pie <- vector("list", length = igraph::vcount(net))
  vertex_pie_colors <- vector("list", length = igraph::vcount(net))

  # Define colors by node type
  vertex_col[igraph::V(net)$NodeType == 'Gene-Epi'] <- 'skyblue4'
  vertex_col[igraph::V(net)$NodeType == 'Gene-Geno'] <- 'orange'
  vertex_col[igraph::V(net)$NodeType == 'Gene-Tran'] <- 'mediumpurple1'
  vertex_col[igraph::V(net)$NodeType == 'Gene-Prot'] <- 'skyblue2'

  # Set pie colors for 'sub_tran' nodes
  exclude_methy_types <- c('methy-Core-Promoter', 'methy-Distal-Promoter', 'methy-Downstream', 
                          'methy-Proximal-Promoter', 'methy-Upstream')

  sub_tran_nodes <- which(
    grepl("geno", igraph::V(net)$NodeType) & 
    !sapply(igraph::V(net)$NodeType, function(type) {
      any(sapply(exclude_methy_types, function(methy_type) grepl(methy_type, type)))
    })
  )

  # vertex_pie[sub_tran_nodes] <- lapply(sub_tran_nodes, function(x) c(1, 1))  # Equal parts
  # vertex_pie_colors[sub_tran_nodes] <- lapply(sub_tran_nodes, function(x) c("mediumpurple1", "orange"))

  # igraph::V(net)$NodeType[sub_tran_nodes] <- "sub_tran"

  # # Set pie colors for 'sub_meth' nodes
  # sub_meth_nodes_1 <- which(grepl("methy-Core-Promoter", igraph::V(net)$NodeType) & grepl("mutation", igraph::V(net)$NodeType))

  # vertex_pie[sub_meth_nodes_1] <- lapply(sub_meth_nodes_1, function(x) c(1, 1))  # Equal parts
  # vertex_pie_colors[sub_meth_nodes_1] <- lapply(sub_meth_nodes_1, function(x) c("hotpink", "orange"))

  # igraph::V(net)$NodeType[sub_meth_nodes_1] <- "sub_meth_1"

  # sub_meth_nodes_2 <- which(grepl("methy-Proximal-Promoter", igraph::V(net)$NodeType) & grepl("mutation", igraph::V(net)$NodeType))

  # vertex_pie[sub_meth_nodes_2] <- lapply(sub_meth_nodes_2, function(x) c(1, 1))  # Equal parts
  # vertex_pie_colors[sub_meth_nodes_2] <- lapply(sub_meth_nodes_2, function(x) c("rosybrown1", "orange"))

  # igraph::V(net)$NodeType[sub_meth_nodes_2] <- "sub_meth_2"

  # sub_meth_nodes_3 <- which(grepl("methy-Downstream", igraph::V(net)$NodeType) & grepl("mutation", igraph::V(net)$NodeType))

  # vertex_pie[sub_meth_nodes_3] <- lapply(sub_meth_nodes_3, function(x) c(1, 1))  # Equal parts
  # vertex_pie_colors[sub_meth_nodes_3] <- lapply(sub_meth_nodes_3, function(x) c("seagreen3", "orange"))

  # igraph::V(net)$NodeType[sub_meth_nodes_3] <- "sub_meth_3"

  # Set colors for 'geno' nodes
  geno_nodes <- which(grepl("geno", igraph::V(net)$NodeType))
  vertex_col[geno_nodes] <- 'orange'

  # Set colors for 'sub_meth_1' nodes
  sub_meth_nodes_1 <- which(grepl("methy-Core-Promoter", igraph::V(net)$NodeType))
  vertex_col[sub_meth_nodes_1] <- "hotpink"

  # Set colors for 'sub_meth_2' nodes
  sub_meth_nodes_2 <- which(grepl("methy-Proximal-Promoter", igraph::V(net)$NodeType))
  vertex_col[sub_meth_nodes_2] <- "rosybrown1"

  # Set colors for 'sub_meth_3' nodes
  sub_meth_nodes_3 <- which(grepl("methy-Downstream", igraph::V(net)$NodeType))
  vertex_col[sub_meth_nodes_3] <- "seagreen3"



  # Find edges from 'sub_tran' to 'Gene-Epi'
  gene_meth_nodes <- which(V(net)$NodeType == 'Gene-Epi')
  # print(gene_meth_nodes)
  edges_to_hide <- E(net)[.from(V(net)[sub_tran_nodes]) & .to(V(net)[gene_meth_nodes])]


  # find edges between missing gene nodes
  if (length(all_missing_vertices) > 0) {
  missing_gene_edges <- E(net)[.from(missing_gene_nodes) & .to(missing_gene_nodes)]
  }

  # Find edges from 'Gene-Tran' to 'sub_meth'
  tran_nodes <- which(V(net)$NodeType == 'Gene-Tran' | V(net)$NodeType %in% V(net)$NodeType[sub_tran_nodes])
  edge_sub_meth_nodes <- which(igraph::V(net)$NodeType == 'sub_meth_1' | igraph::V(net)$NodeType == 'sub_meth_2' | igraph::V(net)$NodeType == 'sub_meth_3')
  edges_from_tran_to_meth <- E(net)[.from(V(net)[tran_nodes]) & .to(V(net)[edge_sub_meth_nodes])]


  # # For accurate remove
  # vertex_size <- rep(1.5, igraph::vcount(net))
  # vertex_cex <- rep(0.2, igraph::vcount(net))
  # edge_width <- rep(0.5, igraph::ecount(net))

  # Edge color for Prot-Prot and port-tran
  edge_color <- rep('gray', igraph::ecount(net))
  edge_color[igraph::E(net)$EdgeType == 'Gene-Tran-Gene-Prot'] <- 'mediumpurple1'
  edge_color[igraph::E(net)$EdgeType == 'Gene-Geno-Gene-Tran'] <- 'orange'

  # Hide edges from 'sub_tran' to 'Gene-Epi'
  edge_color[edges_to_hide] <- NA
  edge_width[edges_to_hide] <- 0

  # # Edge color for tran-meth
  # edge_color[edges_from_tran_to_meth] <- 'hotpink'

   # Hide edges from between missing gene nodes
   if (length(all_missing_vertices) > 0) {
  edge_color[missing_gene_edges] <- NA
  edge_width[missing_gene_edges] <- 0
   }

  # file_name_edge <- paste0("refilter_net_edge_", type, ".csv")
  # file_name_node <- paste0("refilter_net_node_", type, ".csv")
  # write.csv(refilter_net_edge, file_name_edge, row.names = FALSE)
  # write.csv(refilter_net_node, file_name_node, row.names = FALSE)

  file_name_node <- paste0('./ROSMAP-analysis/gene_name_', type,'_', top, '.csv')
  write.csv(refilter_net_node, file_name_node, row.names = FALSE)

  # Custom label
  vertex_label <- rep("", igraph::vcount(net))
  prot_nodes <- igraph::V(net)$NodeType == 'Gene-Prot'
  sub_gene_nodes <- igraph::V(net)$NodeType == 'sub_gene'
  vertex_label[prot_nodes] <- gsub("-Prot", "", igraph::V(net)$Gene_name[prot_nodes])
  vertex_label[sub_gene_nodes] <- igraph::V(net)$Gene_name[sub_gene_nodes]
  
  # Plot the graph and save to PNG
  png(file = paste0('./ROSMAP-analysis/fold_0/', type, '_', top, '.png'), width = 4000, height = 4000, res = 300)

  # vertex.shape <- ifelse(igraph::V(net)$NodeType %in% c('sub_tran', 'sub_meth_1', 'sub_meth_2', 'sub_meth_3'), "pie", "circle")

  # Set margins for the plot 
  # par(mar = c(bottom, left, top, right))
  par(mar = c(0, 0, 0, 0))

  plot(net,
     vertex.frame.color = vertex_fcol,
     vertex.color = vertex_col,
    #  vertex.size = vertex_size,
     vertex.size = V(net)$size,
    #  vertex.shape = vertex.shape,
     vertex.pie = vertex_pie,
     vertex.pie.color = vertex_pie_colors,
     vertex.label = vertex_label,
    # vertex.label = sub("-Prot|-Epi|-Tran", "", V(net)$Gene_name),
    # vertex.label = V(net)$Gene_name,
     vertex.label.color = 'black',
     vertex.label.cex = vertex_cex,
     edge.width = edge_width,
     edge.color = edge_color,
     edge.curved = 0.2,
     layout = layout_nicely(net))


# layout=layout_nicely
# layout=layout_as_tree
# layout_with_kk
# layout=layout_with_dh
# layout=layout_with_gem
  
  # Add legends within the plot area
  legend('topleft', inset=c(0.05, 0.05),  # Adjusted y coordinate for upper-left corner
        legend = c('Genes',  'Proteins', 'Mutation', 'methy-Core-Promoter', 'methy-Proximal-Promoter', 'methy-Downstream'), 
        pch = c(21, 21, 21),
        pt.bg = c('mediumpurple1', 'skyblue2', 'orange', 'hotpink', 'rosybrown1', 'seagreen3'), 
        pt.cex = 1.5, 
        cex = 1, 
        bty = 'n')
  
  legend('topleft', inset=c(0.05, 0.15),  # Adjusted y coordinate for upper-left corner
         legend = c('Protein-Protein', 'Protein-Gene', 'Gene-Mutation' ,'Gene-Promoter'),
         col = c('gray', 'mediumpurple1', 'orange', 'hotpink'), 
         lwd = 10,
         pt.cex = 1.5,
         cex = 0.8,  
         bty = 'n')
  
  dev.off()

  print("Graph saved.")
  
}
}

# Process files for 'AD' and 'NOAD'
process_and_plot_graph('AD')
process_and_plot_graph('NOAD')

}