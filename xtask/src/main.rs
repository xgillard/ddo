use clap::Parser;
use xtaskops::tasks;

#[derive(Debug, clap::Parser)]
enum Args {
    Coverage{
        /// Generate html report
        #[clap(short, long)]
        dev_mode: bool
    },
    Ci,
    Powerset,
    BloatDeps{package: String},
    BloatTime{package: String},
    Docs,
}

fn main() -> Result<(), anyhow::Error> {
    let args = Args::parse();
    match args {
        Args::Coverage{dev_mode} => tasks::coverage(dev_mode),
        Args::Ci                 => tasks::ci(),
        Args::Powerset           => tasks::powerset(),
        Args::BloatDeps{package} => tasks::bloat_deps(&package),
        Args::BloatTime{package} => tasks::bloat_time(&package),
        Args::Docs               => tasks::docs(),
    }
}
