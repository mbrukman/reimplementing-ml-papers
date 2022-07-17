# GoogLeNet diagrams

> Note: GitHub rendering of SVG does not seem to handle stylesheets; you should
> view these SVG files locally with the provided stylesheet to get the colors
> for the nodes.

This directory contains the following diagrams:

* [`googlenet-paper.svg`](googlenet-paper.svg) - the network architecture as
  described in the paper
* [`googlenet-clusters.svg`](googlenet-clusters.svg) - groups Inception modules
  together with a dashed outline to make it easier to recognize repeated
  high-level components
* [`googlenet-simplified.svg`](googlenet-simplified.svg) - each Inception module
  collapsed to a single ndoe for easier understanding and implementation

> **Important:** GitHub's web file viewer renders SVGs without the associated
> [stylesheet] that they refer to, which means you're missing the text and
> background colors to match the rendering in the paper. To properly see the
> diagrams as indented, please see the [published version] of this README on
> GitHub Pages, and click on the above links on that page.

## Implementation

Due to the highly repeating nature of the Inception modules in this diagram,
instead of copy-pasting a lot of the DOT code, we used the M4 macro language to
enable reuse of the Inception modules.

Thus, instead of writing raw DOT code, we have `*.m4` files which are mostly DOT
files, except for the M4 directives.

After updating any of the `*.m4` files, you ran simply run `make` in this
directory to regenerate all of the SVG output files.

`*.dot` files are only generated on the fly and passed to the `dot` tool via
stdin and are not versioned here. If you'd like to see what they look like
before the rendering to SVG, simply run `m4` on any of the `googlenet-*.m4`
files:

```sh
$ m4 googlenet-paper.m4 | less
$ m4 googlenet-clusters.m4 | less
$ m4 googlenet-simplified.m4 | less
```

[stylesheet]: googlenet.css
[published version]: https://mbrukman.github.io/reimplementing-ml-papers/googlenet/diagrams
