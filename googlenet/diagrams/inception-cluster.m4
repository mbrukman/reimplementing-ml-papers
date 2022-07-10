  // Begin: Inception module `module'

  subgraph format(`cluster_Inception_%s_module', module) {
    label = format(`"Inception %s"', module);
    // Since we inverted the direction of the graph to go bottom -> top, to put
    // the label at the "top" of the box, we have to invert it and say "bottom"
    // instead.
    labelloc = "bottom";
    fontsize = 18;

    graph [
      style="dashed",
    ];

    include(`inception-paper.m4')
  }

  // End: Inception module `module'
