// Copyright 2020 Xavier Gillard
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of
// this software and associated documentation files (the "Software"), to deal in
// the Software without restriction, including without limitation the rights to
// use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
// the Software, and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
// FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
// IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
// CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

use std::{fs::File, path::PathBuf};

use ddo::{MaxUB, NoDupFringe, Problem, Solver, DefaultCachingSolver, SimpleDominanceChecker, NoCutoff};

use crate::{dominance::TsptwDominance, heuristics::{TsptwRanking, TsptwWidth}, instance::TsptwInstance, model::Tsptw, relax::TsptwRelax};

fn locate(id: &str) -> PathBuf {
    PathBuf::new()
        .join(env!("CARGO_MANIFEST_DIR"))
        .join("../resources/tsptw/")
        .join(id)
}

pub fn solve(instance: &str, width: Option<usize>, threads: Option<usize>) -> f32 {
    let file = File::open(locate(instance)).expect("file not found");
    let inst = TsptwInstance::from(file);
    let pb = Tsptw::new(inst);
    let mut fringe = NoDupFringe::new(MaxUB::new(&TsptwRanking));
    let relax = TsptwRelax::new(&pb);
    let width = TsptwWidth::new(pb.nb_variables(), width.unwrap_or(1));
    let dominance = SimpleDominanceChecker::new(TsptwDominance);
    let cutoff = NoCutoff;
    let mut solver = DefaultCachingSolver::custom(
        &pb,
        &relax,
        &TsptwRanking,
        &width,
        &dominance,
        &cutoff,
        &mut fringe,
        threads.unwrap_or(num_cpus::get()),
    );
    let outcome = solver.maximize();

    outcome
        .best_value
        .map(|v| -(v as f32) / 10000.0)
        .unwrap_or(-1.0)

    /*if outcome.is_exact {
        outcome
            .best_value
            .map(|v| -(v as f32) / 10000.0)
            .unwrap_or(-1.0)
    } else {
        -1.0
    }*/
}

fn solve_langevin(id: &str) -> f32 {
    let id = format!("Langevin/{}", id);
    solve(&id, Some(1), Some(1))
}

fn solve_solomon_potvin_bengio(id: &str) -> f32 {
    let id = format!("SolomonPotvinBengio/{}", id);
    solve(&id, Some(1), Some(1))
}


#[test]
fn n20ft301_dat() {
    assert_eq!(661.60, solve_langevin("N20ft301.dat"));
}


#[test]
fn n20ft302_dat() {
    assert_eq!(703.00, solve_langevin("N20ft302.dat"));
}


#[test]
fn n20ft303_dat() {
    assert_eq!(746.40, solve_langevin("N20ft303.dat"));
}


#[test]
fn n20ft304_dat() {
    assert_eq!(817.00, solve_langevin("N20ft304.dat"));
}


#[test]
fn n20ft305_dat() {
    assert_eq!(724.70, solve_langevin("N20ft305.dat"));
}


#[test]
fn n20ft306_dat() {
    assert_eq!(729.50, solve_langevin("N20ft306.dat"));
}


#[test]
fn n20ft307_dat() {
    assert_eq!(691.80, solve_langevin("N20ft307.dat"));
}


#[test]
fn n20ft308_dat() {
    assert_eq!(788.20, solve_langevin("N20ft308.dat"));
}


#[test]
fn n20ft309_dat() {
    assert_eq!(751.80, solve_langevin("N20ft309.dat"));
}


#[test]
fn n20ft310_dat() {
    assert_eq!(693.80, solve_langevin("N20ft310.dat"));
}


#[test]
fn n20ft401_dat() {
    assert_eq!(660.90, solve_langevin("N20ft401.dat"));
}


#[test]
fn n20ft402_dat() {
    assert_eq!(701.00, solve_langevin("N20ft402.dat"));
}


#[test]
fn n20ft403_dat() {
    assert_eq!(746.40, solve_langevin("N20ft403.dat"));
}


#[test]
fn n20ft404_dat() {
    assert_eq!(817.00, solve_langevin("N20ft404.dat"));
}


#[test]
fn n20ft405_dat() {
    assert_eq!(724.70, solve_langevin("N20ft405.dat"));
}


#[test]
fn n20ft406_dat() {
    assert_eq!(728.50, solve_langevin("N20ft406.dat"));
}


#[test]
fn n20ft407_dat() {
    assert_eq!(691.80, solve_langevin("N20ft407.dat"));
}


#[test]
fn n20ft408_dat() {
    assert_eq!(786.10, solve_langevin("N20ft408.dat"));
}


#[test]
fn n20ft409_dat() {
    assert_eq!(749.80, solve_langevin("N20ft409.dat"));
}


#[test]
fn n20ft410_dat() {
    assert_eq!(693.80, solve_langevin("N20ft410.dat"));
}


#[test]
fn n40ft201_dat() {
    assert_eq!(1109.30, solve_langevin("N40ft201.dat"));
}


#[test]
fn n40ft202_dat() {
    assert_eq!(1017.40, solve_langevin("N40ft202.dat"));
}


#[test]
fn n40ft203_dat() {
    assert_eq!(903.10, solve_langevin("N40ft203.dat"));
}


#[test]
fn n40ft204_dat() {
    assert_eq!(897.40, solve_langevin("N40ft204.dat"));
}


#[test]
fn n40ft205_dat() {
    assert_eq!(983.60, solve_langevin("N40ft205.dat"));
}


#[test]
fn n40ft206_dat() {
    assert_eq!(1081.90, solve_langevin("N40ft206.dat"));
}


#[test]
fn n40ft207_dat() {
    assert_eq!(884.90, solve_langevin("N40ft207.dat"));
}


#[test]
fn n40ft208_dat() {
    assert_eq!(1051.60, solve_langevin("N40ft208.dat"));
}


#[test]
fn n40ft209_dat() {
    assert_eq!(1027.50, solve_langevin("N40ft209.dat"));
}


#[test]
fn n40ft210_dat() {
    assert_eq!(1035.30, solve_langevin("N40ft210.dat"));
}


#[test]
fn n40ft401_dat() {
    assert_eq!(1105.20, solve_langevin("N40ft401.dat"));
}


#[test]
fn n40ft402_dat() {
    assert_eq!(1016.40, solve_langevin("N40ft402.dat"));
}


#[test]
fn n40ft403_dat() {
    assert_eq!(903.10, solve_langevin("N40ft403.dat"));
}


#[test]
fn n40ft404_dat() {
    assert_eq!(897.40, solve_langevin("N40ft404.dat"));
}


#[test]
fn n40ft405_dat() {
    assert_eq!(982.60, solve_langevin("N40ft405.dat"));
}


#[test]
fn n40ft406_dat() {
    assert_eq!(1081.90, solve_langevin("N40ft406.dat"));
}


#[test]
fn n40ft407_dat() {
    assert_eq!(872.20, solve_langevin("N40ft407.dat"));
}


#[test]
fn n40ft408_dat() {
    assert_eq!(1043.50, solve_langevin("N40ft408.dat"));
}


#[test]
fn n40ft409_dat() {
    assert_eq!(1025.50, solve_langevin("N40ft409.dat"));
}


#[test]
fn n40ft410_dat() {
    assert_eq!(1034.30, solve_langevin("N40ft410.dat"));
}


#[test]
fn n60ft201_dat() {
    assert_eq!(1375.40, solve_langevin("N60ft201.dat"));
}


#[test]
fn n60ft202_dat() {
    assert_eq!(1186.40, solve_langevin("N60ft202.dat"));
}


#[test]
fn n60ft203_dat() {
    assert_eq!(1194.20, solve_langevin("N60ft203.dat"));
}


#[test]
fn n60ft204_dat() {
    assert_eq!(1283.60, solve_langevin("N60ft204.dat"));
}


#[test]
fn n60ft205_dat() {
    assert_eq!(1215.50, solve_langevin("N60ft205.dat"));
}


#[test]
fn n60ft206_dat() {
    assert_eq!(1238.80, solve_langevin("N60ft206.dat"));
}


#[test]
fn n60ft207_dat() {
    assert_eq!(1305.30, solve_langevin("N60ft207.dat"));
}


#[test]
fn n60ft208_dat() {
    assert_eq!(1172.60, solve_langevin("N60ft208.dat"));
}


#[test]
fn n60ft209_dat() {
    assert_eq!(1243.80, solve_langevin("N60ft209.dat"));
}


#[test]
fn n60ft210_dat() {
    assert_eq!(1273.20, solve_langevin("N60ft210.dat"));
}


#[test]
fn n60ft301_dat() {
    assert_eq!(1375.40, solve_langevin("N60ft301.dat"));
}


#[test]
fn n60ft302_dat() {
    assert_eq!(1184.40, solve_langevin("N60ft302.dat"));
}


#[test]
fn n60ft303_dat() {
    assert_eq!(1194.20, solve_langevin("N60ft303.dat"));
}


#[test]
fn n60ft304_dat() {
    assert_eq!(1283.60, solve_langevin("N60ft304.dat"));
}


#[test]
fn n60ft305_dat() {
    assert_eq!(1214.50, solve_langevin("N60ft305.dat"));
}


#[test]
fn n60ft306_dat() {
    assert_eq!(1237.80, solve_langevin("N60ft306.dat"));
}


#[test]
fn n60ft307_dat() {
    assert_eq!(1298.40, solve_langevin("N60ft307.dat"));
}


#[test]
fn n60ft308_dat() {
    assert_eq!(1168.80, solve_langevin("N60ft308.dat"));
}


#[test]
fn n60ft309_dat() {
    assert_eq!(1242.80, solve_langevin("N60ft309.dat"));
}


#[test]
fn n60ft310_dat() {
    assert_eq!(1273.20, solve_langevin("N60ft310.dat"));
}


#[test]
fn n60ft401_dat() {
    assert_eq!(1375.40, solve_langevin("N60ft401.dat"));
}


#[test]
fn n60ft402_dat() {
    assert_eq!(1183.40, solve_langevin("N60ft402.dat"));
}


#[test]
fn n60ft403_dat() {
    assert_eq!(1194.20, solve_langevin("N60ft403.dat"));
}


#[test]
fn n60ft404_dat() {
    assert_eq!(1283.60, solve_langevin("N60ft404.dat"));
}


#[test]
fn n60ft405_dat() {
    assert_eq!(1212.50, solve_langevin("N60ft405.dat"));
}


#[test]
fn n60ft406_dat() {
    assert_eq!(1236.80, solve_langevin("N60ft406.dat"));
}


#[test]
fn n60ft407_dat() {
    assert_eq!(1296.40, solve_langevin("N60ft407.dat"));
}


#[test]
fn n60ft408_dat() {
    assert_eq!(1150.00, solve_langevin("N60ft408.dat"));
}


#[test]
fn n60ft409_dat() {
    assert_eq!(1241.80, solve_langevin("N60ft409.dat"));
}


#[test]
fn n60ft410_dat() {
    assert_eq!(1273.20, solve_langevin("N60ft410.dat"));
}


// The solutions to the instances are to be found in the below link.
// However, they have been rounded a little bit too much so they must be 
// recomputed (but at least the values give an idea)
// http://lopez-ibanez.eu/files/TSPTW/SolomonPotvinBengio-best-known-makespan.txt

#[test]
fn rc_201_1_txt() {
    assert_eq!(592.0611, solve_solomon_potvin_bengio("rc_201.1.txt"));
}


#[test]
fn rc_201_2_txt() {
    assert_eq!(860.1748, solve_solomon_potvin_bengio("rc_201.2.txt"));
}


#[test]
fn rc_201_3_txt() {
    assert_eq!(853.7075, solve_solomon_potvin_bengio("rc_201.3.txt"));
}


#[test]
fn rc_201_4_txt() {
    assert_eq!(889.1761, solve_solomon_potvin_bengio("rc_201.4.txt"));
}


#[test] #[ignore]
fn rc_202_1_txt() {
    assert_eq!(850.48, solve_solomon_potvin_bengio("rc_202.1.txt"));
}


#[test]
fn rc_202_2_txt() {
    assert_eq!(338.5183, solve_solomon_potvin_bengio("rc_202.2.txt"));
}


#[test]
fn rc_202_3_txt() {
    assert_eq!(894.1028, solve_solomon_potvin_bengio("rc_202.3.txt"));
}


#[test] #[ignore]
fn rc_202_4_txt() {
    assert_eq!(853.7075, solve_solomon_potvin_bengio("rc_202.4.txt"));
}


#[test]
fn rc_203_1_txt() {
    assert_eq!(488.4224, solve_solomon_potvin_bengio("rc_203.1.txt"));
}


#[ignore] #[test]
fn rc_203_2_txt() {
    assert_eq!(853.7075, solve_solomon_potvin_bengio("rc_203.2.txt"));
}


#[ignore] #[test]
fn rc_203_3_txt() {
    assert_eq!(921.4397, solve_solomon_potvin_bengio("rc_203.3.txt"));
}


#[test]
fn rc_203_4_txt() {
    assert_eq!(338.5183, solve_solomon_potvin_bengio("rc_203.4.txt"));
}


#[ignore] #[test]
fn rc_204_1_txt() {
    assert_eq!(917.83, solve_solomon_potvin_bengio("rc_204.1.txt"));
}


#[test] #[ignore]
fn rc_204_2_txt() {
    assert_eq!(690.06, solve_solomon_potvin_bengio("rc_204.2.txt"));
}


#[ignore] #[test]
fn rc_204_3_txt() {
    assert_eq!(455.0315, solve_solomon_potvin_bengio("rc_204.3.txt"));
}


#[test]
fn rc_205_1_txt() {
    assert_eq!(417.8058, solve_solomon_potvin_bengio("rc_205.1.txt"));
}


#[test]
fn rc_205_2_txt() {
    assert_eq!(820.1853, solve_solomon_potvin_bengio("rc_205.2.txt"));
}


#[test]
fn rc_205_3_txt() {
    assert_eq!(950.0539, solve_solomon_potvin_bengio("rc_205.3.txt"));
}


#[test]
fn rc_205_4_txt() {
    assert_eq!(837.7083, solve_solomon_potvin_bengio("rc_205.4.txt"));
}


#[test]
fn rc_206_1_txt() {
    assert_eq!(117.8479, solve_solomon_potvin_bengio("rc_206.1.txt"));
}


#[test]
fn rc_206_2_txt() {
    assert_eq!(870.4875, solve_solomon_potvin_bengio("rc_206.2.txt"));
}


#[test]
fn rc_206_3_txt() {
    assert_eq!(650.5942, solve_solomon_potvin_bengio("rc_206.3.txt"));
}


#[test]
fn rc_206_4_txt() {
    assert_eq!(911.9814, solve_solomon_potvin_bengio("rc_206.4.txt"));
}


#[ignore] #[test]
fn rc_207_1_txt() {
    assert_eq!(804.6735, solve_solomon_potvin_bengio("rc_207.1.txt"));
}


#[test] #[ignore]
fn rc_207_2_txt() {
    assert_eq!(713.90, solve_solomon_potvin_bengio("rc_207.2.txt"));
}


#[ignore] #[test]
fn rc_207_3_txt() {
    assert_eq!(745.7717, solve_solomon_potvin_bengio("rc_207.3.txt"));
}


#[test]
fn rc_207_4_txt() {
    assert_eq!(133.1421, solve_solomon_potvin_bengio("rc_207.4.txt"));
}


#[ignore] #[test]
fn rc_208_1_txt() {
    assert_eq!(810.70, solve_solomon_potvin_bengio("rc_208.1.txt"));
}


#[ignore] #[test]
fn rc_208_2_txt() {
    assert_eq!(579.51, solve_solomon_potvin_bengio("rc_208.2.txt"));
}


#[ignore] #[test]
fn rc_208_3_txt() {
    assert_eq!(686.7954, solve_solomon_potvin_bengio("rc_208.3.txt"));
}

