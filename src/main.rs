use std::fs;
use std::path::PathBuf;

use _emet::{DecisionReport, ProblemInput, decide_problem};
use clap::{Args, Parser, Subcommand};

#[derive(Parser, Debug)]
#[command(name = "emet")]
#[command(about = "Given a split, compute chi and interpret it.")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Args, Debug, Clone)]
struct ReportArgs {
    input: PathBuf,
    #[arg(long)]
    json: bool,
    #[arg(long)]
    pretty_json: bool,
    #[arg(long, hide = true)]
    pretty: bool,
}

#[derive(Subcommand, Debug)]
enum Command {
    Decide(ReportArgs),
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();
    match cli.command {
        Command::Decide(args) => run_report(args)?,
    }
    Ok(())
}

fn run_report(args: ReportArgs) -> Result<(), Box<dyn std::error::Error>> {
    let raw = fs::read_to_string(args.input)?;
    let problem: ProblemInput = serde_json::from_str(&raw)?;
    let report = decide_problem(&problem);

    if args.pretty || args.pretty_json {
        println!("{}", serde_json::to_string_pretty(&report)?);
    } else if args.json {
        println!("{}", serde_json::to_string(&report)?);
    } else {
        print!("{}", render_human_report(&report));
    }

    Ok(())
}

fn render_human_report(report: &DecisionReport) -> String {
    let mut lines = vec![
        format!("Regime: {}", report.regime.as_str().to_uppercase()),
        format!(
            "Licensed: {}",
            if report.valid { "YES" } else { "NO" }
        ),
        format!("Reason: {}", report.reason_code.as_str()),
        format!(
            "Retained / omitted: {} / {}",
            report.partition_profile.retained_dimension, report.partition_profile.omitted_dimension
        ),
    ];

    if let Some(metrics) = &report.advanced_metrics {
        lines.push(format!("lambda (coupling):    {:.12}", metrics.lambda));
        lines.push(format!("gamma (control):      {:.12}", metrics.gamma));
        if let Some(chi) = metrics.chi {
            lines.push(format!("chi = (lambda/gamma)^2: {:.12}", chi));
        }
        if let Some(delta) = metrics.delta {
            lines.push(format!("delta = lambda^2/gamma: {:.12}", delta));
        }
        if let Some(window) = metrics.residual_window {
            lines.push(format!("residual window:     {:.12}", window));
        }
    }

    if let Some(semantics) = report.partition_profile.control_semantics {
        lines.push(format!("control semantics: {}", semantics.as_str()));
    }

    if report.reduced_matrix.is_some() {
        lines.push(if report.valid {
            "effective matrix: computed, licensed".to_string()
        } else {
            "effective matrix: computed, NOT licensed".to_string()
        });
    }

    format!("{}\n", lines.join("\n"))
}
