#!/usr/bin/env python3

"""Generate a simplified workflow diagram for the README.

Produces a compact, publication-ready DAG image that represents
the workflow structure without expanding every parallel job.
"""

import argparse
import subprocess
import sys


def make_dot(n_images=2):
    """Build a Graphviz DOT string for the simplified workflow."""

    colors = {
        'split': '#4472C4',
        'segment': '#ED7D31',
        'merge': '#70AD47',
        'autolabel': '#BF8F00',
        'stage2': '#7030A0',
        'plots': '#C00000',
        'output': '#F2F2F2',
    }
    fc = 'white'

    lines = []
    def L(s=''):
        lines.append(s)

    L('digraph S2_Segmentation {')
    L('  rankdir=TB;')
    L('  dpi=200;')
    L('  bgcolor=white;')
    L('  pad=0.4;')
    L('  nodesep=0.6;')
    L('  ranksep=0.8;')
    L('  compound=true;')
    L('  newrank=true;')
    L()
    L('  node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=13];')
    L('  edge [color="#555555", arrowsize=0.7];')
    L()

    # ── Title ──
    L('  labelloc=t;')
    L('  label="S2 Segmentation Workflow (auto-label mode)";')
    L('  fontname="Helvetica-Bold"; fontsize=18; fontcolor="#333333";')
    L()

    # ── Stage 1: per-image columns ──
    for i in range(n_images):
        tag = f"Image {i}"
        L(f'  subgraph cluster_img{i} {{')
        L(f'    label="{tag}"; labeljust=c; fontname="Helvetica-Bold"; fontsize=14; fontcolor="#444444";')
        L(f'    style="dashed,rounded"; color="#AAAAAA";')
        L()
        L(f'    split_{i}      [label="image_split",                fillcolor="{colors["split"]}",     fontcolor={fc}];')
        L(f'    seg_{i}        [label="color_segment\\n(×64 parallel)", fillcolor="{colors["segment"]}",  fontcolor={fc}];')
        L(f'    merge_{i}      [label="image_merge",                fillcolor="{colors["merge"]}",     fontcolor={fc}];')
        L(f'    seg_out_{i}    [label="*_seg.png",                  shape=note, style=filled, fillcolor="{colors["output"]}", fontcolor="#333333", fontsize=11];')
        L(f'    split_img_{i}  [label="split_images\\n(256×256 tiles)", fillcolor="{colors["autolabel"]}", fontcolor={fc}];')
        L(f'    split_mask_{i} [label="split_masks\\n(256×256 tiles)",  fillcolor="{colors["autolabel"]}", fontcolor={fc}];')
        L()
        L(f'    split_{i} -> seg_{i};')
        L(f'    seg_{i} -> merge_{i};')
        L(f'    merge_{i} -> seg_out_{i} [style=dotted, color="#AAAAAA"];')
        L(f'    merge_{i} -> split_mask_{i};')
        L(f'    split_{i} -> split_img_{i} [style=dashed, color="#999999"];')
        L(f'  }}')
        L()

    # ── Ellipsis ──
    if n_images == 2:
        L('  ellipsis [label="  ...  \\n(× N images)", shape=plaintext, fontsize=13, fontname="Helvetica", fontcolor="#888888"];')
        L(f'  {{ rank=same; merge_0; ellipsis; merge_1; }}')
        L()

    # ── Stage 2 ──
    L('  subgraph cluster_stage2 {')
    L('    label="Stage 2 — U-Net Training & Evaluation"; labeljust=c;')
    L('    fontname="Helvetica-Bold"; fontsize=14; fontcolor="#444444";')
    L('    style="dashed,rounded"; color="#AAAAAA";')
    L()
    L(f'    preprocess [label="preprocess_data",         fillcolor="{colors["stage2"]}", fontcolor={fc}];')
    L(f'    train      [label="train_unet\\n(GPU)",       fillcolor="{colors["stage2"]}", fontcolor={fc}];')
    L(f'    evaluate   [label="evaluate_model\\n(GPU)",   fillcolor="{colors["stage2"]}", fontcolor={fc}];')
    L(f'    plots      [label="generate_plots\\n(GPU)",   fillcolor="{colors["plots"]}",  fontcolor={fc}];')
    L()
    L('    preprocess -> train -> evaluate -> plots;')
    L('  }')
    L()

    # ── Edges: auto-label → preprocess ──
    for i in range(n_images):
        L(f'  split_img_{i}  -> preprocess;')
        L(f'  split_mask_{i} -> preprocess;')
    L()

    # ── Stage 2 outputs ──
    L('  node [shape=note, style=filled, fillcolor="#F2F2F2", fontcolor="#333333", fontsize=11];')
    L('  out_model   [label="model.hdf5"];')
    L('  out_history [label="training_history.json"];')
    L('  out_eval    [label="evaluation_results.json"];')
    L('  out_curves  [label="training_curves.png"];')
    L('  out_cm      [label="confusion_matrix.png"];')
    L('  out_pred    [label="prediction_samples.png"];')
    L('  out_table   [label="metrics_table.png"];')
    L('  out_json    [label="per_class_metrics.json"];')
    L()
    L('  train    -> out_model   [style=dotted, color="#AAAAAA"];')
    L('  train    -> out_history [style=dotted, color="#AAAAAA"];')
    L('  evaluate -> out_eval    [style=dotted, color="#AAAAAA"];')
    L('  plots    -> out_curves  [style=dotted, color="#AAAAAA"];')
    L('  plots    -> out_cm      [style=dotted, color="#AAAAAA"];')
    L('  plots    -> out_pred    [style=dotted, color="#AAAAAA"];')
    L('  plots    -> out_table   [style=dotted, color="#AAAAAA"];')
    L('  plots    -> out_json    [style=dotted, color="#AAAAAA"];')
    L()

    # ── Rank hints ──
    splits = " ".join(f"split_{i};" for i in range(n_images))
    segs = " ".join(f"seg_{i};" for i in range(n_images))
    merges = " ".join(f"merge_{i};" for i in range(n_images))
    L(f'  {{ rank=same; {splits} }}')
    L(f'  {{ rank=same; {segs} }}')

    auto_nodes = " ".join(f"split_img_{i}; split_mask_{i};" for i in range(n_images))
    L(f'  {{ rank=same; {auto_nodes} }}')
    L('  { rank=same; out_model; out_history; }')
    L('  { rank=same; out_eval; }')
    L('  { rank=same; out_curves; out_cm; out_pred; out_table; out_json; }')

    L('}')
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Generate a simplified workflow diagram",
    )
    parser.add_argument("-n", "--n-images", type=int, default=2,
                        help="Number of representative images to show (default: 2)")
    parser.add_argument("-o", "--output", type=str, default="images/workflow.png",
                        help="Output file (png, pdf, svg, or dot)")
    parser.add_argument("--dot-only", action="store_true",
                        help="Print DOT to stdout instead of rendering")
    args = parser.parse_args()

    dot_str = make_dot(n_images=args.n_images)

    if args.dot_only:
        print(dot_str)
        return

    ext = args.output.rsplit(".", 1)[-1].lower()

    if ext == "dot":
        with open(args.output, "w") as f:
            f.write(dot_str)
        print(f"DOT file written to {args.output}")
        return

    try:
        subprocess.run(
            ["dot", f"-T{ext}", "-o", args.output],
            input=dot_str,
            text=True,
            capture_output=True,
            check=True,
        )
        print(f"Diagram written to {args.output}")
    except FileNotFoundError:
        print("Error: 'dot' command not found. Install graphviz:", file=sys.stderr)
        print("  brew install graphviz    # macOS", file=sys.stderr)
        print("  apt install graphviz     # Debian/Ubuntu", file=sys.stderr)
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"Error running dot: {e.stderr}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
