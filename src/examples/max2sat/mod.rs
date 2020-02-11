pub mod instance;
pub mod model;
pub mod relax;
pub mod heuristics;

#[cfg(test)]
mod testutils {
    use crate::examples::max2sat::model::Max2Sat;
    use std::path::PathBuf;
    use std::fs::File;

    pub fn instance(id: &str) -> Max2Sat {
        let location = PathBuf::new()
            .join(env!("CARGO_MANIFEST_DIR"))
            .join("tests/resources/max2sat/")
            .join(id);

        File::open(location).expect("File not found").into()
    }
}