pub mod instance;
pub mod model;
pub mod relax;
pub mod heuristics;


pub mod main;

/*
#[cfg(test)]
mod testutils {
    use crate::examples::max2sat::model::Max2Sat;
    use std::path::PathBuf;
    use std::fs::File;

    pub fn locate(id: &str) -> PathBuf {
        PathBuf::new()
            .join(env!("CARGO_MANIFEST_DIR"))
            .join("tests/resources/max2sat/")
            .join(id)
    }

    pub fn instance(id: &str) -> Max2Sat {
        let location = locate(id);
        File::open(location).expect("File not found").into()
    }
}*/