"""Generate architecture diagram for the LLM Data Structures Optimizer.

This script creates a visual architecture diagram showing the relationships
between major components in the system.
"""

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np


def generate_architecture_diagram(output_path: Path = Path("audit/ARCH_DIAGRAM.png")):
    """
    Generate architecture diagram showing system components and relationships.
    
    Args:
        output_path: Path to save the diagram (default: audit/ARCH_DIAGRAM.png)
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(16, 12))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    
    # Define colors
    colors = {
        "kv_cache": "#E8F4F8",
        "scheduler": "#FFF4E6",
        "retrieval": "#F0F8E8",
        "data_structure": "#F5E6F8",
    }
    
    # Title
    ax.text(5, 9.5, "LLM Data Structures Optimizer Architecture", 
            ha="center", va="top", fontsize=20, weight="bold")
    
    # ===== KV Cache System =====
    kv_y = 7.5
    ax.add_patch(mpatches.Rectangle((0.2, kv_y), 3.0, 1.5, 
                                     facecolor=colors["kv_cache"], 
                                     edgecolor="black", linewidth=2))
    ax.text(1.7, kv_y + 1.2, "KV Cache System", 
            ha="center", va="center", fontsize=14, weight="bold")
    
    # KVCache
    ax.add_patch(mpatches.Rectangle((0.4, kv_y + 0.7), 1.2, 0.4,
                                     facecolor="white", edgecolor="black", linewidth=1))
    ax.text(1.0, kv_y + 0.9, "KVCache", ha="center", va="center", fontsize=10)
    
    # PagedAllocator
    ax.add_patch(mpatches.Rectangle((1.8, kv_y + 0.7), 1.2, 0.4,
                                     facecolor="white", edgecolor="black", linewidth=1))
    ax.text(2.4, kv_y + 0.9, "PagedAllocator", ha="center", va="center", fontsize=10)
    
    # TokenLRU
    ax.add_patch(mpatches.Rectangle((0.4, kv_y - 0.2), 1.2, 0.4,
                                     facecolor="white", edgecolor="black", linewidth=1))
    ax.text(1.0, kv_y, "TokenLRU", ha="center", va="center", fontsize=10)
    
    # Connections within KV Cache
    ax.arrow(1.6, kv_y + 0.9, 0.2, 0, head_width=0.05, head_length=0.05, 
             fc="black", ec="black")
    ax.arrow(1.0, kv_y + 0.5, 0, 0.2, head_width=0.05, head_length=0.05, 
             fc="black", ec="black")
    
    # ===== Scheduler & Batching =====
    scheduler_y = 5.5
    ax.add_patch(mpatches.Rectangle((0.2, scheduler_y), 3.0, 1.5,
                                     facecolor=colors["scheduler"],
                                     edgecolor="black", linewidth=2))
    ax.text(1.7, scheduler_y + 1.2, "Scheduler & Batching", 
            ha="center", va="center", fontsize=14, weight="bold")
    
    # Scheduler
    ax.add_patch(mpatches.Rectangle((0.4, scheduler_y + 0.7), 1.2, 0.4,
                                     facecolor="white", edgecolor="black", linewidth=1))
    ax.text(1.0, scheduler_y + 0.9, "Scheduler", ha="center", va="center", fontsize=10)
    
    # IndexedHeap
    ax.add_patch(mpatches.Rectangle((1.8, scheduler_y + 0.7), 1.2, 0.4,
                                     facecolor="white", edgecolor="black", linewidth=1))
    ax.text(2.4, scheduler_y + 0.9, "IndexedHeap", ha="center", va="center", fontsize=10)
    
    # AdmissionController
    ax.add_patch(mpatches.Rectangle((1.1, scheduler_y - 0.2), 1.2, 0.4,
                                     facecolor="white", edgecolor="black", linewidth=1))
    ax.text(1.7, scheduler_y, "AdmissionController", ha="center", va="center", fontsize=10)
    
    # Connections within Scheduler
    ax.arrow(1.6, scheduler_y + 0.9, 0.2, 0, head_width=0.05, head_length=0.05,
             fc="black", ec="black")
    ax.arrow(1.7, scheduler_y + 0.5, 0, 0.2, head_width=0.05, head_length=0.05,
             fc="black", ec="black")
    
    # ===== Retrieval Pipeline =====
    retrieval_y = 3.5
    ax.add_patch(mpatches.Rectangle((0.2, retrieval_y), 3.0, 1.5,
                                     facecolor=colors["retrieval"],
                                     edgecolor="black", linewidth=2))
    ax.text(1.7, retrieval_y + 1.2, "Retrieval Pipeline", 
            ha="center", va="center", fontsize=14, weight="bold")
    
    # RetrievalPipeline
    ax.add_patch(mpatches.Rectangle((1.1, retrieval_y + 0.7), 1.2, 0.4,
                                     facecolor="white", edgecolor="black", linewidth=2))
    ax.text(1.7, retrieval_y + 0.9, "RetrievalPipeline", 
            ha="center", va="center", fontsize=11, weight="bold")
    
    # HNSW
    ax.add_patch(mpatches.Rectangle((0.4, retrieval_y - 0.2), 1.2, 0.4,
                                     facecolor="white", edgecolor="black", linewidth=1))
    ax.text(1.0, retrieval_y, "HNSW", ha="center", va="center", fontsize=10)
    
    # InvertedIndex
    ax.add_patch(mpatches.Rectangle((1.8, retrieval_y - 0.2), 1.2, 0.4,
                                     facecolor="white", edgecolor="black", linewidth=1))
    ax.text(2.4, retrieval_y, "InvertedIndex", ha="center", va="center", fontsize=10)
    
    # CountMinSketch
    ax.add_patch(mpatches.Rectangle((0.4, retrieval_y - 0.9), 1.2, 0.4,
                                     facecolor="white", edgecolor="black", linewidth=1))
    ax.text(1.0, retrieval_y - 0.7, "CountMinSketch", ha="center", va="center", fontsize=10)
    
    # Tokenizer
    ax.add_patch(mpatches.Rectangle((1.8, retrieval_y - 0.9), 1.2, 0.4,
                                     facecolor="white", edgecolor="black", linewidth=1))
    ax.text(2.4, retrieval_y - 0.7, "Tokenizer", ha="center", va="center", fontsize=10)
    
    # Connections within Retrieval Pipeline
    ax.arrow(1.7, retrieval_y + 0.5, -0.3, 0.2, head_width=0.05, head_length=0.05,
             fc="black", ec="black")
    ax.arrow(1.7, retrieval_y + 0.5, 0.3, 0.2, head_width=0.05, head_length=0.05,
             fc="black", ec="black")
    ax.arrow(1.7, retrieval_y + 0.5, -0.3, -0.5, head_width=0.05, head_length=0.05,
             fc="black", ec="black")
    ax.arrow(1.7, retrieval_y + 0.5, 0.3, -0.5, head_width=0.05, head_length=0.05,
             fc="black", ec="black")
    
    # ===== Data Flow Arrows =====
    # KV Cache to Scheduler
    ax.arrow(1.7, scheduler_y + 1.5, 0, 0.3, head_width=0.1, head_length=0.08,
             fc="blue", ec="blue", linewidth=2, linestyle="--")
    ax.text(2.2, scheduler_y + 1.8, "uses", ha="left", va="center", 
            fontsize=9, color="blue", style="italic")
    
    # Scheduler to Retrieval
    ax.arrow(1.7, scheduler_y - 0.5, 0, -0.3, head_width=0.1, head_length=0.08,
             fc="green", ec="green", linewidth=2, linestyle="--")
    ax.text(2.2, retrieval_y + 1.5, "schedules", ha="left", va="center", 
            fontsize=9, color="green", style="italic")
    
    # ===== Right Side: Data Structures =====
    ds_x = 6.0
    ax.add_patch(mpatches.Rectangle((ds_x, 6.5), 3.5, 3.0,
                                     facecolor=colors["data_structure"],
                                     edgecolor="black", linewidth=2))
    ax.text(ds_x + 1.75, 9.0, "Core Data Structures", 
            ha="center", va="center", fontsize=14, weight="bold")
    
    # List data structures
    structures = [
        "IndexedHeap: O(log n) priority queue",
        "PagedAllocator: Page-based memory",
        "TokenLRU: Token-aware cache",
        "HNSW: Hierarchical graph ANN",
        "InvertedIndex: BM25 search",
        "CountMinSketch: Frequency estimation",
    ]
    
    for i, struct in enumerate(structures):
        y_pos = 8.3 - i * 0.45
        ax.text(ds_x + 0.2, y_pos, "•", ha="left", va="center", fontsize=12)
        ax.text(ds_x + 0.4, y_pos, struct, ha="left", va="center", fontsize=9)
    
    # ===== Legend =====
    legend_y = 1.5
    ax.text(0.2, legend_y + 1.2, "Legend:", ha="left", va="top", 
            fontsize=12, weight="bold")
    
    # Legend items
    legend_items = [
        ("───", "blue", "KV Cache usage"),
        ("───", "green", "Scheduler flow"),
        ("────", "black", "Component relationships"),
    ]
    
    for i, (style, color, label) in enumerate(legend_items):
        y_pos = legend_y + 0.8 - i * 0.3
        ax.plot([0.4, 0.7], [y_pos, y_pos], color=color, linewidth=2, 
                linestyle="--" if "usage" in label or "flow" in label else "-")
        ax.text(0.8, y_pos, label, ha="left", va="center", fontsize=9)
    
    # ===== Notes =====
    notes_x = 5.0
    notes_y = 2.0
    ax.add_patch(mpatches.Rectangle((notes_x, notes_y), 4.5, 1.8,
                                     facecolor="#F5F5F5",
                                     edgecolor="gray", linewidth=1))
    ax.text(notes_x + 2.25, notes_y + 1.5, "Key Features", 
            ha="center", va="center", fontsize=11, weight="bold")
    
    key_features = [
        "• Copy-on-write prefix sharing",
        "• Reference counting for memory",
        "• Hybrid dense + sparse retrieval",
        "• Score fusion with configurable weights",
    ]
    
    for i, feature in enumerate(key_features):
        y_pos = notes_y + 1.1 - i * 0.35
        ax.text(notes_x + 0.2, y_pos, feature, ha="left", va="center", fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Architecture diagram saved to: {output_path}")
    return output_path


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate architecture diagram")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("audit/ARCH_DIAGRAM.png"),
        help="Output file path (default: audit/ARCH_DIAGRAM.png)",
    )
    args = parser.parse_args()
    
    generate_architecture_diagram(args.output)

