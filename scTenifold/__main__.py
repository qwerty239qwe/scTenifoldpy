import typer
from pathlib import Path
from types import SimpleNamespace
import yaml
from scTenifold import scTenifoldNet, scTenifoldKnk

app = typer.Typer()


@app.command(name="config")
def get_config_file(
        config_type: int = typer.Option(1, "--type", "-t",
                                        help="Type, 1: scTenifoldNet, 2: scTenifoldKnk",
                                        min=1, max=2),
        file_path: str = typer.Option(
                        ".config.yml",
                        "--path",
                        "-p",
                        help="Path to generate empty config file")) -> None:
    """Write an empty scTenifoldNet or scTenifoldKnk config YAML."""
    config = scTenifoldNet.get_empty_config() if config_type == 1 else scTenifoldKnk.get_empty_config()
    with open(Path(file_path), 'w') as outfile:
        yaml.dump(config, outfile, default_flow_style=False)


@app.command(name="net")
def build_net(config_file_path: str = typer.Option(...,
                                                   "--config",
                                                   "-c",
                                                   help="Loaded config file's path"),
              output_dir_path: str = typer.Option("./saved_net",
                                                  "--output",
                                                  "-o",
                                                  help="Output folder containing all analysis results"),
              ) -> None:
    """Build a scTenifoldNet from a config YAML and save the result."""
    with open(Path(config_file_path), "r") as f:
        data = yaml.safe_load(f)
    sc = scTenifoldNet.load_config(config=data)
    sc.build()
    sc.save(output_dir_path)


@app.command(name="knk")
def build_knk(config_file_path: str = typer.Option(...,
                                                   "--config",
                                                   "-c",
                                                   help="Loaded config file's path"),
              output_dir_path: str = typer.Option("./saved_knk",
                                                  "--output",
                                                  "-o",
                                                  help="Output folder containing all analysis results"),
              ) -> None:
    """Build a scTenifoldKnk from a config YAML and save the result."""
    with open(Path(config_file_path), "r") as f:
        data = yaml.safe_load(f)
    sc = scTenifoldKnk.load_config(config=data)
    sc.build()
    sc.save(output_dir_path)


@app.command(name="xct")
def build_xct(
        file: str = typer.Argument(..., help="Path to log-normalised AnnData (.h5ad)"),
        sender: str = typer.Option("cell_A", "--sender", "-s", help="Sender cell type label"),
        receiver: str = typer.Option("cell_B", "--receiver", "-r", help="Receiver cell type label"),
        label: str = typer.Option("ident", "--label", "-l", help="obs column with cell-type labels"),
        workdir: str = typer.Option("xct_results", "--workdir", "-w", help="Output directory"),
        output: str = typer.Option("xct_enriched", "--output", "-o", help="Output file stem"),
        n_cpus: int = typer.Option(-1, "--n_cpus", help="CPUs for GRN construction (-1 = all)"),
        rebuild: bool = typer.Option(True, "--rebuild/--no-rebuild", help="Rebuild the gene regulatory networks"),
        verbose: bool = typer.Option(False, "--verbose", "-v"),
        ) -> None:
    """Run scTenifoldXct single-sample cell-cell interaction analysis (requires the [xct] extra)."""
    try:
        from scTenifoldXct.core import main
    except ImportError as exc:
        raise typer.BadParameter(
            "scTenifoldXct is not installed. Install it with: pip install scTenifoldpy[xct]"
        ) from exc
    args = SimpleNamespace(file=file, sender=sender, receiver=receiver, label=label,
                           workdir=workdir, output=output, n_cpus=n_cpus, rebuild=rebuild,
                           verbose=verbose, eva=False, n_sample=100, n_feature=3000)
    main(args)


@app.command(name="xct-merge")
def build_xct_merge(
        file: str = typer.Argument(..., help="Path to log-normalised AnnData (.h5ad)"),
        cond_label: str = typer.Argument(..., help="obs column distinguishing the two conditions"),
        cond_wt: str = typer.Argument(..., help="Reference condition label"),
        cond_ko: str = typer.Argument(..., help="Comparison condition label"),
        sender: str = typer.Option("cell_A", "--sender", "-s", help="Sender cell type label"),
        receiver: str = typer.Option("cell_B", "--receiver", "-r", help="Receiver cell type label"),
        label: str = typer.Option("ident", "--label", "-l", help="obs column with cell-type labels"),
        workdir: str = typer.Option("xct_results", "--workdir", "-w", help="Output directory"),
        output: str = typer.Option("xct_enriched_diff", "--output", "-o", help="Output file stem"),
        n_cpus: int = typer.Option(-1, "--n_cpus", help="CPUs for GRN construction (-1 = all)"),
        rebuild: bool = typer.Option(True, "--rebuild/--no-rebuild", help="Rebuild the gene regulatory networks"),
        verbose: bool = typer.Option(False, "--verbose", "-v"),
        ) -> None:
    """Run scTenifoldXct two-sample differential interaction analysis (requires the [xct] extra)."""
    try:
        from scTenifoldXct.merge import main
    except ImportError as exc:
        raise typer.BadParameter(
            "scTenifoldXct is not installed. Install it with: pip install scTenifoldpy[xct]"
        ) from exc
    args = SimpleNamespace(file=file, cond_label=cond_label, cond_WT=cond_wt, cond_KO=cond_ko,
                           sender=sender, receiver=receiver, label=label,
                           workdir=workdir, output=output, n_cpus=n_cpus, rebuild=rebuild,
                           verbose=verbose, eva=False, n_sample=100, n_feature=3000)
    main(args)


if __name__ == '__main__':
    app()