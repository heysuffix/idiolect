"""
idiolect CLI — Cross-lingual author attribution tool.

Usage:
    idiolect compare text_a.txt text_b.txt
    idiolect compare text_a.txt text_b.txt --no-embeddings
    idiolect features text.txt
    idiolect multi author_sample.txt unknown1.txt unknown2.txt unknown3.txt
"""

import sys
import os
import json as _json
import click

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.text import Text
    from rich import box
    HAS_RICH = True
except ImportError:
    HAS_RICH = False


console = Console() if HAS_RICH else None


def _read_file(path: str) -> str:
    if path == "-":
        return sys.stdin.read().strip()
    if not os.path.exists(path):
        click.echo(f"Error: file not found: {path}", err=True)
        sys.exit(1)
    with open(path, encoding="utf-8") as f:
        return f.read().strip()


def _print_score_bar(score: float, width: int = 40) -> str:
    filled = int(score / 100 * width)
    bar = "█" * filled + "░" * (width - filled)
    return f"[{bar}] {score:.1f}/100"


def _color_score(score: float) -> str:
    if score >= 75:
        return "bold green"
    elif score >= 58:
        return "yellow"
    elif score >= 42:
        return "white"
    elif score >= 28:
        return "dark_orange"
    else:
        return "bold red"


def _print_result(result, file_a: str, file_b: str):
    from idiolect.scorer import AttributionResult

    if HAS_RICH:
        # Header
        console.print()
        console.print(Panel.fit(
            "[bold cyan]idiolect[/bold cyan] — Cross-lingual Author Attribution",
            border_style="cyan"
        ))
        console.print()

        # Files
        console.print(f"  [dim]File A:[/dim] {file_a}  [dim](detected: {result.text_a_lang})[/dim]")
        console.print(f"  [dim]File B:[/dim] {file_b}  [dim](detected: {result.text_b_lang})[/dim]")
        console.print()

        # Score bar
        color = _color_score(result.score)
        bar_filled = int(result.score / 100 * 40)
        bar = "█" * bar_filled + "░" * (40 - bar_filled)
        console.print(f"  [bold]Attribution Score[/bold]")
        if result.ci_low is not None and result.ci_high is not None:
            console.print(f"  [{color}][{bar}][/{color}] [{color}]{result.score:.1f}/100[/{color}]  [dim](95% CI: {result.ci_low}–{result.ci_high})[/dim]")
        else:
            console.print(f"  [{color}][{bar}][/{color}] [{color}]{result.score:.1f}/100[/{color}]")
        console.print()

        # Verdict
        console.print(f"  [bold]Verdict:[/bold]     {result.verdict}")
        console.print(f"  [bold]Confidence:[/bold]  {result.confidence}")
        console.print()

        # Sub-scores
        sub_table = Table(show_header=False, box=None, padding=(0, 2))
        sub_table.add_row(
            "[dim]Structural features[/dim]",
            f"{result.feature_score * 100:.1f}%"
        )
        sub_table.add_row(
            "[dim]Semantic style (multilingual)[/dim]",
            f"{result.embedding_score * 100:.1f}%"
        )
        console.print(sub_table)
        console.print()

        # Feature breakdown
        feat_table = Table(
            "Feature", "Text A", "Text B", "Match",
            box=box.SIMPLE,
            title="[bold]Top Matching Features[/bold]",
            title_style="bold green",
        )
        for name, val_a, val_b, sim in result.top_matches:
            sim_pct = f"{sim * 100:.0f}%"
            feat_table.add_row(name, f"{val_a:.3f}", f"{val_b:.3f}", f"[green]{sim_pct}[/green]")
        console.print(feat_table)

        mismatch_table = Table(
            "Feature", "Text A", "Text B", "Match",
            box=box.SIMPLE,
            title="[bold]Diverging Features[/bold]",
            title_style="bold red",
        )
        for name, val_a, val_b, sim in result.top_mismatches:
            sim_pct = f"{sim * 100:.0f}%"
            mismatch_table.add_row(name, f"{val_a:.3f}", f"{val_b:.3f}", f"[red]{sim_pct}[/red]")
        console.print(mismatch_table)

        if result.warnings:
            console.print()
            for w in result.warnings:
                console.print(f"  [yellow]Warning:[/yellow] {w}")

    else:
        # Plain fallback
        print(f"\nidiolect — Cross-lingual Author Attribution")
        print(f"=" * 50)
        print(f"File A: {file_a} (lang: {result.text_a_lang})")
        print(f"File B: {file_b} (lang: {result.text_b_lang})")
        print(f"\nScore:      {_print_score_bar(result.score)}")
        print(f"Verdict:    {result.verdict}")
        print(f"Confidence: {result.confidence}")
        print(f"\nStructural:   {result.feature_score * 100:.1f}%")
        print(f"Semantic:     {result.embedding_score * 100:.1f}%")
        for w in result.warnings:
            print(f"Warning: {w}")


@click.group()
@click.version_option(version="0.1.0", prog_name="idiolect")
def cli():
    """idiolect — Find the same voice across different languages."""
    pass


@cli.command()
@click.argument("file_a", type=click.Path())
@click.argument("file_b", type=click.Path())
@click.option("--no-embeddings", is_flag=True, help="Skip ML embeddings (faster, structural only)")
@click.option("--bootstrap", "-b", is_flag=True, help="Compute 95%% confidence interval via bootstrap sampling")
@click.option("--bootstrap-n", default=100, help="Bootstrap iterations (default: 100)")
@click.option("--output", "-o", type=click.Choice(["text", "json"]), default="text", help="Output format")
@click.option("--verbose", "-v", is_flag=True)
def compare(file_a, file_b, no_embeddings, bootstrap, bootstrap_n, output, verbose):
    """Compare two text files for common authorship. Use - for stdin."""
    from idiolect import compare as _compare, compare_with_ci

    text_a = _read_file(file_a)
    text_b = _read_file(file_b)

    if HAS_RICH and verbose and output == "text":
        console.print("[dim]Loading model and extracting features...[/dim]")

    if bootstrap:
        if HAS_RICH and output == "text":
            console.print(f"[dim]Running {bootstrap_n} bootstrap iterations...[/dim]")
        result = compare_with_ci(text_a, text_b, n_bootstrap=bootstrap_n, verbose=verbose)
    else:
        result = _compare(text_a, text_b, use_embeddings=not no_embeddings, verbose=verbose, quiet=output == "json")

    if output == "json":
        click.echo(_json.dumps({
            "file_a": file_a,
            "file_b": file_b,
            "lang_a": result.text_a_lang,
            "lang_b": result.text_b_lang,
            "score": result.score,
            "verdict": result.verdict,
            "confidence": result.confidence,
            "feature_score": round(result.feature_score * 100, 1),
            "embedding_score": round(result.embedding_score * 100, 1),
            "top_matches": [
                {"feature": n, "val_a": round(a, 4), "val_b": round(b, 4), "similarity": round(s, 4)}
                for n, a, b, s in result.top_matches
            ],
            "top_mismatches": [
                {"feature": n, "val_a": round(a, 4), "val_b": round(b, 4), "similarity": round(s, 4)}
                for n, a, b, s in result.top_mismatches
            ],
            "warnings": result.warnings,
            "ci_low": result.ci_low,
            "ci_high": result.ci_high,
        }, indent=2))
    else:
        _print_result(result, file_a, file_b)


@cli.command()
@click.argument("file", type=click.Path(exists=True))
def features(file):
    """Show extracted style features for a text file."""
    from idiolect import extract_features
    text = _read_file(file)
    feat = extract_features(text)

    if HAS_RICH:
        table = Table("Feature", "Value", box=box.SIMPLE, title=f"Style Features: {file}")
        table.add_row("Sentence length (mean/std)", f"{feat.sent_length_mean:.2f} ± {feat.sent_length_std:.2f}")
        table.add_row("Comma per sentence", f"{feat.comma_per_sent:.3f}")
        table.add_row("Question ratio", f"{feat.question_ratio:.3f}")
        table.add_row("Exclamation ratio", f"{feat.exclamation_ratio:.3f}")
        table.add_row("Word length (mean)", f"{feat.word_length_mean:.2f}")
        table.add_row("Lexical richness", f"{feat.lexical_richness:.3f}")
        table.add_row("Short sentence ratio", f"{feat.short_sent_ratio:.3f}")
        table.add_row("Long sentence ratio", f"{feat.long_sent_ratio:.3f}")
        table.add_row("Mid-sent capitalization", f"{feat.capitalized_mid_sent:.3f}")
        table.add_row("Ellipsis density", f"{feat.ellipsis_density:.3f}")
        table.add_row("Dash density", f"{feat.dash_density:.3f}")
        table.add_row("Para avg sentences", f"{feat.para_sent_mean:.2f}")
        console.print(table)
    else:
        print(f"\nStyle features: {file}")
        print(f"Sentence length: {feat.sent_length_mean:.2f} ± {feat.sent_length_std:.2f}")
        print(f"Comma/sent: {feat.comma_per_sent:.3f}")
        print(f"Lexical richness: {feat.lexical_richness:.3f}")


@cli.command()
@click.argument("author_sample", type=click.Path(exists=True))
@click.argument("unknowns", nargs=-1, type=click.Path(exists=True))
@click.option("--no-embeddings", is_flag=True)
@click.option("--output", "-o", type=click.Choice(["text", "json"]), default="text")
@click.option("--verbose", "-v", is_flag=True)
def multi(author_sample, unknowns, no_embeddings, output, verbose):
    """
    Compare one known author sample against multiple unknown texts.

    Example: idiolect multi known_author.txt suspect1.txt suspect2.txt suspect3.txt
    """
    from idiolect import compare as _compare

    if not unknowns:
        click.echo("Error: provide at least one unknown text file.", err=True)
        sys.exit(1)

    text_a = _read_file(author_sample)
    results = []

    for path in unknowns:
        text_b = _read_file(path)
        r = _compare(text_a, text_b, use_embeddings=not no_embeddings, verbose=verbose, quiet=output == "json")
        results.append((path, r))

    results.sort(key=lambda x: x[1].score, reverse=True)

    if output == "json":
        click.echo(_json.dumps([{
            "rank": i + 1,
            "file": path,
            "lang": r.text_b_lang,
            "score": r.score,
            "verdict": r.verdict,
            "confidence": r.confidence,
            "warnings": r.warnings,
        } for i, (path, r) in enumerate(results)], indent=2))
        return

    if HAS_RICH:
        console.print()
        console.print(Panel.fit(
            "[bold cyan]idiolect[/bold cyan] — Multi-target Attribution",
            border_style="cyan"
        ))
        console.print(f"\n  [dim]Author sample:[/dim] {author_sample}\n")

        table = Table("Rank", "File", "Score", "Verdict", "Confidence", box=box.SIMPLE)
        for i, (path, r) in enumerate(results, 1):
            color = _color_score(r.score)
            table.add_row(
                str(i),
                os.path.basename(path),
                f"[{color}]{r.score:.1f}[/{color}]",
                r.verdict,
                r.confidence,
            )
        console.print(table)
    else:
        print(f"\nMulti-target attribution — Author: {author_sample}\n")
        print(f"{'Rank':<5} {'File':<30} {'Score':<8} {'Verdict'}")
        print("-" * 80)
        for i, (path, r) in enumerate(results, 1):
            print(f"{i:<5} {os.path.basename(path):<30} {r.score:<8.1f} {r.verdict}")


@cli.command()
@click.argument("directory", type=click.Path(exists=True, file_okay=False))
@click.option("--no-embeddings", is_flag=True)
@click.option("--output", "-o", type=click.Choice(["text", "json"]), default="text")
@click.option("--threshold", "-t", default=60.0, help="Only show pairs above this score (default: 60)")
def batch(directory, no_embeddings, output, threshold):
    """
    Compare all .txt files in a directory against each other.

    Useful for finding related texts in a corpus.

    Example: idiolect batch ./corpus/ --threshold 65
    """
    from idiolect import compare as _compare

    txt_files = [
        os.path.join(directory, f)
        for f in sorted(os.listdir(directory))
        if f.endswith(".txt")
    ]

    if len(txt_files) < 2:
        click.echo("Error: need at least 2 .txt files in the directory.", err=True)
        sys.exit(1)

    pairs = []
    total = len(txt_files) * (len(txt_files) - 1) // 2

    if HAS_RICH and output == "text":
        console.print(f"\n[dim]Comparing {len(txt_files)} files ({total} pairs)...[/dim]\n")

    for i, path_a in enumerate(txt_files):
        for path_b in txt_files[i + 1:]:
            text_a = _read_file(path_a)
            text_b = _read_file(path_b)
            r = _compare(text_a, text_b, use_embeddings=not no_embeddings, quiet=output == "json")
            if r.score >= threshold:
                pairs.append((path_a, path_b, r))

    pairs.sort(key=lambda x: x[2].score, reverse=True)

    if output == "json":
        click.echo(_json.dumps([{
            "file_a": a, "file_b": b,
            "score": r.score, "verdict": r.verdict, "confidence": r.confidence,
        } for a, b, r in pairs], indent=2))
        return

    if not pairs:
        msg = f"No pairs found above threshold {threshold}"
        if HAS_RICH:
            console.print(f"[dim]{msg}[/dim]")
        else:
            print(msg)
        return

    if HAS_RICH:
        table = Table("File A", "File B", "Score", "Verdict", box=box.SIMPLE,
                      title=f"Pairs above {threshold:.0f}")
        for a, b, r in pairs:
            color = _color_score(r.score)
            table.add_row(
                os.path.basename(a), os.path.basename(b),
                f"[{color}]{r.score:.1f}[/{color}]", r.verdict,
            )
        console.print(table)
    else:
        print(f"\nPairs above {threshold:.0f}:\n")
        for a, b, r in pairs:
            print(f"{r.score:.1f}  {os.path.basename(a)} <-> {os.path.basename(b)}  {r.verdict}")


@cli.command()
@click.argument("files", nargs=-1, type=click.Path(exists=True), required=True)
@click.option("--no-embeddings", is_flag=True)
@click.option("--output", "-o", type=click.Choice(["text", "json", "csv"]), default="text")
@click.option("--heatmap", type=click.Path(), default=None, help="Save heatmap image to this path (requires matplotlib)")
def matrix(files, no_embeddings, output, heatmap):
    """
    Compute full N×N similarity matrix for a set of texts.

    Example: idiolect matrix a.txt b.txt c.txt d.txt --heatmap out.png
    """
    from idiolect import compare as _compare

    if len(files) < 2:
        click.echo("Error: need at least 2 files.", err=True)
        sys.exit(1)

    labels = [os.path.basename(f) for f in files]
    texts = [_read_file(f) for f in files]
    n = len(texts)

    # Build matrix (symmetric, diagonal = 100)
    scores = [[0.0] * n for _ in range(n)]
    for i in range(n):
        scores[i][i] = 100.0
    for i in range(n):
        for j in range(i + 1, n):
            r = _compare(texts[i], texts[j], use_embeddings=not no_embeddings, quiet=output != "text")
            scores[i][j] = r.score
            scores[j][i] = r.score

    if output == "json":
        click.echo(_json.dumps({
            "labels": labels,
            "matrix": scores,
        }, indent=2))

    elif output == "csv":
        import csv, io
        out = io.StringIO()
        w = csv.writer(out)
        w.writerow([""] + labels)
        for i, row in enumerate(scores):
            w.writerow([labels[i]] + [f"{v:.1f}" for v in row])
        click.echo(out.getvalue())

    else:
        if HAS_RICH:
            col_width = max(len(l) for l in labels) + 2
            t = Table(box=box.SIMPLE, title="Similarity Matrix")
            t.add_column("", style="dim", min_width=col_width)
            for l in labels:
                t.add_column(l, justify="right", min_width=7)
            for i, row in enumerate(scores):
                cells = []
                for j, v in enumerate(row):
                    if i == j:
                        cells.append("[dim]—[/dim]")
                    else:
                        color = _color_score(v)
                        cells.append(f"[{color}]{v:.1f}[/{color}]")
                t.add_row(labels[i], *cells)
            console.print()
            console.print(t)
        else:
            header = f"{'':20}" + "".join(f"{l:>10}" for l in labels)
            print(header)
            for i, row in enumerate(scores):
                line = f"{labels[i]:20}" + "".join(f"{v:>10.1f}" for v in row)
                print(line)

    if heatmap:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import numpy as np

            arr = np.array(scores)
            fig, ax = plt.subplots(figsize=(max(6, n * 1.2), max(5, n * 1.0)))
            im = ax.imshow(arr, cmap="RdYlGn", vmin=0, vmax=100)
            ax.set_xticks(range(n))
            ax.set_yticks(range(n))
            ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
            ax.set_yticklabels(labels, fontsize=9)
            for i in range(n):
                for j in range(n):
                    ax.text(j, i, f"{arr[i,j]:.0f}", ha="center", va="center",
                            fontsize=8, color="black" if 30 < arr[i,j] < 75 else "white")
            plt.colorbar(im, ax=ax, label="Attribution Score")
            ax.set_title("idiolect — Similarity Matrix", fontsize=12, pad=15)
            plt.tight_layout()
            plt.savefig(heatmap, dpi=150)
            plt.close()
            if HAS_RICH:
                console.print(f"[dim]Heatmap saved to {heatmap}[/dim]")
            else:
                print(f"Heatmap saved to {heatmap}")
        except ImportError:
            click.echo("matplotlib not installed. pip install matplotlib", err=True)


@cli.command()
@click.option("--host", default="127.0.0.1", help="Host to bind (default: 127.0.0.1)")
@click.option("--port", "-p", default=8080, help="Port to listen on (default: 8080)")
@click.option("--reload", is_flag=True, help="Auto-reload on code changes (dev mode)")
def serve(host, port, reload):
    """
    Start the idiolect REST API server.

    Endpoints:
      POST /compare   — compare two texts
      POST /multi     — rank suspects against known author
      POST /matrix    — N×N similarity matrix
      POST /features  — extract style features
      GET  /health    — health check

    Requires: pip install fastapi uvicorn
    """
    try:
        import uvicorn
    except ImportError:
        click.echo("uvicorn is required: pip install fastapi uvicorn", err=True)
        sys.exit(1)

    if HAS_RICH:
        console.print(f"\n[bold cyan]idiolect[/bold cyan] API server")
        console.print(f"  [dim]Listening on[/dim] http://{host}:{port}")
        console.print(f"  [dim]Docs at[/dim]      http://{host}:{port}/docs\n")

    uvicorn.run(
        "idiolect.server:create_app",
        factory=True,
        host=host,
        port=port,
        reload=reload,
        log_level="warning",
    )


if __name__ == "__main__":
    cli()
