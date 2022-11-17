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
    BloatDeps,
    BloatTime,
    Docs,
}

fn main() -> Result<(), anyhow::Error> {
    let args = Args::parse();
    match args {
        Args::Coverage{dev_mode} => tasks::coverage(dev_mode),
        Args::Ci                       => tasks::ci(),
        Args::Powerset                 => tasks::powerset(),
        Args::BloatDeps                => tasks::bloat_deps(),
        Args::BloatTime                => tasks::bloat_time(),
        Args::Docs                     => tasks::docs(),
    }
}