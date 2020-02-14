use std::hash::Hash;

use binary_heap_plus::BinaryHeap;
use compare::Compare;

use crate::core::abstraction::mdd::MDD;
use crate::core::abstraction::solver::Solver;
use crate::core::common::{Decision, Node, NodeInfo};
use parking_lot::{Condvar, Mutex};
use std::sync::Arc;

/// The shared data that may only be manipulated within critical sections
struct Critical<T, BO> where T: Eq + Clone, BO: Compare<Node<T>> {
    pub fringe   : BinaryHeap<Node<T>, BO>,
    pub ongoing  : usize,
    pub explored : usize,
    pub best_lb  : i32,
    pub best_node: Option<NodeInfo>
}
/// The state which is shared among the many running threads
struct Shared<T, BO> where T: Eq + Clone, BO: Compare<Node<T>> {
    pub critical : Mutex<Critical<T, BO>>,
    pub monitor  : Condvar
}

pub struct BBSolver<T, DD, BO>
    where T: Hash + Eq + Clone + Send, DD: MDD<T> + Clone + Send, BO: Compare<Node<T>> + Send {

    mdd          : DD,
    shared       : Arc<Shared<T, BO>>,
    pub best_sol : Option<Vec<Decision>>,
    pub verbosity: u8
}

impl <T, DD, BO> BBSolver<T, DD, BO>
    where T: Hash + Eq + Clone + Send, DD: MDD<T> + Clone + Send, BO: Compare<Node<T>> + Send {
    pub fn new(mdd: DD, bo : BO) -> Self {
        BBSolver {
            mdd,
            shared: Arc::new(Shared {
                monitor : Condvar::new(),
                critical: Mutex::new(Critical {
                    best_node: None,
                    best_lb  : i32::min_value(),
                    ongoing  : 0,
                    explored : 0,
                    fringe   : BinaryHeap::from_vec_cmp(vec![], bo)
                })
            }),
            best_sol: None,
            verbosity: 0
        }
    }

    fn maybe_update_best(mdd: &DD, shared: &Arc<Shared<T, BO>>) {
        let mut shared = shared.critical.lock();
        if mdd.best_value() > shared.best_lb {
            shared.best_lb   = mdd.best_value();
            shared.best_node = mdd.best_node().clone();
        }
    }

    fn process_one_node(verbosity: u8, mdd: &mut DD, shared: &Arc<Shared<T, BO>>, node: Node<T>) {
        let mut best_lb = {shared.critical.lock().best_lb};

        // 1. RESTRICTION
        mdd.restricted(&node, best_lb);
        Self::maybe_update_best(mdd, shared);
        if mdd.is_exact() {
            shared.critical.lock().ongoing -= 1;
            shared.monitor.notify_all();
            return;
        }

        // 2. RELAXATION
        best_lb = {shared.critical.lock().best_lb};
        mdd.relaxed(&node, best_lb);
        if mdd.is_exact() {
            Self::maybe_update_best(mdd, shared);
        } else {
            let ub           = node.info.ub;
            { // CRITICAL SECTION:: push cutset nodes
                let mut critical = shared.critical.lock();
                let best_lb = critical.best_lb;
                let fringe = &mut critical.fringe;
                mdd.consume_cutset(|state, mut info| {
                    info.ub = ub.min(info.ub);
                    if info.ub > best_lb {
                        fringe.push(Node { state, info });
                    }
                });
            }
        }
        shared.critical.lock().ongoing -= 1;
        shared.monitor.notify_all();
    }
}

impl <T, DD, BO> Solver for BBSolver<T, DD, BO>
    where T: Hash + Eq + Clone + Send, DD: MDD<T> + Clone + Send, BO: Compare<Node<T>> + Send {

    fn maximize(&mut self) -> (i32, &Option<Vec<Decision>>) {
        let cpus   = num_cpus::get();

        { // Kickstart
            let root = self.mdd.root();
            self.shared.critical.lock().fringe.push(root);
        }

        crossbeam::thread::scope(|s|{
            for i in 0..cpus {
                let shared = Arc::clone(&self.shared);
                let mut mdd = self.mdd.clone();
                let verbosity = self.verbosity;

                s.spawn(move |_| {
                    loop {
                        let mut node      = None;
                        let mut explored  = 0;
                        let mut lb        = 0;
                        let mut fringe_sz = 0;
                        { // Critical section
                            let mut critical = shared.critical.lock();
                            // Are we done ?
                            if critical.ongoing == 0 && critical.fringe.is_empty() {
                                break;
                            }
                            // Nothing to do yet ? => Wait for someone to post jobs
                            if critical.fringe.is_empty() {
                                shared.monitor.wait(&mut critical);
                                continue;
                            } else {
                                let nn = critical.fringe.pop().unwrap();
                                if nn.info.ub < critical.best_lb {
                                    // all remaining nodes have lessed bound, drop them
                                    critical.fringe.clear();
                                    continue;
                                } else {
                                    critical.ongoing += 1;
                                    critical.explored+= 1;
                                    explored  = critical.explored;
                                    lb        = critical.best_lb;
                                    fringe_sz = critical.fringe.len();
                                    node = Some(nn);
                                }
                            }
                        }

                        if verbosity >= 2 && explored % 100 == 0 {
                            println!("Explored {}, LB {}, Fringe sz {}", explored, lb, fringe_sz);
                        }
                        Self::process_one_node(verbosity, &mut mdd, &shared, node.unwrap());
                    }
                });
            }
        });

        let shared = self.shared.critical.lock();
        if let Some(bn) = &shared.best_node {
            self.best_sol = Some(bn.longest_path());
        }

        // return
        if self.verbosity >= 1 {
            println!("Final {}, Explored {}", shared.best_lb, shared.explored);
        }
        (shared.best_lb, &self.best_sol)
    }
}