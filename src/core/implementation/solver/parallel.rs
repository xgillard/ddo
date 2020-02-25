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

use binary_heap_plus::BinaryHeap;

use crate::core::abstraction::mdd::MDD;
use crate::core::abstraction::solver::Solver;
use crate::core::common::{Decision, Node, NodeInfo};
use parking_lot::{Condvar, Mutex};
use std::sync::Arc;
use crate::core::implementation::heuristics::MaxUB;

/// The shared data that may only be manipulated within critical sections
struct Critical<T> {
    fringe   : BinaryHeap<Node<T>, MaxUB>,
    ongoing  : usize,
    explored : usize,
    best_lb  : i32,
    best_node: Option<NodeInfo>
}
/// The state which is shared among the many running threads
struct Shared<T> {
    critical : Mutex<Critical<T>>,
    monitor  : Condvar
}
/// The workload a thread can get from the shared state
enum WorkLoad<T> {
    /// There is no work left to be done: you can safely terminate
    Complete,
    /// There is nothing you can do right now. Check again when you wake up
    Starvation,
    /// The item to process
    WorkItem {
        explored : usize,
        fringe_sz: usize,
        best_lb  : i32,
        node     : Node<T>
    }
}

pub struct ParallelSolver<T, DD> where T: Send, DD: MDD<T> + Clone + Send {
    mdd       : DD,
    shared    : Arc<Shared<T>>,
    best_sol  : Option<Vec<Decision>>,
    verbosity : u8,
    nb_threads: usize
}

impl <T, DD> ParallelSolver<T, DD> where T: Send, DD: MDD<T> + Clone + Send {
    pub fn new(mdd: DD) -> Self {
        Self::customized(mdd, 0, num_cpus::get())
    }
    pub fn with_verbosity(mdd: DD, verbosity: u8) -> Self {
        Self::customized(mdd, verbosity, num_cpus::get())
    }
    pub fn with_nb_threads(mdd: DD, nb_threads: usize) -> Self {
        Self::customized(mdd, 0, nb_threads)
    }
    pub fn customized(mdd: DD, verbosity: u8, nb_threads: usize) -> Self {
        ParallelSolver {
            mdd,
            shared: Arc::new(Shared {
                monitor : Condvar::new(),
                critical: Mutex::new(Critical {
                    best_node: None,
                    best_lb  : i32::min_value(),
                    ongoing  : 0,
                    explored : 0,
                    fringe   : BinaryHeap::from_vec_cmp(vec![], MaxUB)
                })
            }),
            best_sol: None,
            verbosity,
            nb_threads
        }
    }

    fn initialize(&self) {
        let root = self.mdd.root();
        self.shared.critical.lock().fringe.push(root);
    }

    fn process_one_node(mdd: &mut DD, shared: &Arc<Shared<T>>, node: Node<T>) {
        let mut best_lb = {shared.critical.lock().best_lb};

        // 1. RESTRICTION
        mdd.restricted(&node, best_lb);
        Self::maybe_update_best(mdd, shared);
        if mdd.is_exact() {
            Self::notify_node_finished(shared);
            return;
        }

        // 2. RELAXATION
        best_lb = {shared.critical.lock().best_lb};
        mdd.relaxed(&node, best_lb);
        if mdd.is_exact() {
            Self::maybe_update_best(mdd, shared);
        } else {
            Self::enqueue_cutset(mdd, shared, node.info.ub);
        }

        Self::notify_node_finished(shared);
    }

    fn maybe_update_best(mdd: &DD, shared: &Arc<Shared<T>>) {
        let mut shared = shared.critical.lock();
        if mdd.best_value() > shared.best_lb {
            shared.best_lb   = mdd.best_value();
            shared.best_node = mdd.best_node().clone();
        }
    }

    fn enqueue_cutset(mdd: &mut DD, shared: &Arc<Shared<T>>, ub: i32) {
        let mut critical = shared.critical.lock();
        let best_lb      = critical.best_lb;
        let fringe       = &mut critical.fringe;
        mdd.consume_cutset(|state, mut info| {
            info.ub = ub.min(info.ub);
            if info.ub > best_lb {
                fringe.push(Node { state, info });
            }
        });
    }

    fn notify_node_finished(shared: &Arc<Shared<T>>) {
        shared.critical.lock().ongoing -= 1;
        shared.monitor.notify_all();
    }

    fn get_workload(shared: &Arc<Shared<T>>) -> WorkLoad<T> {
        let mut critical = shared.critical.lock();
        // Are we done ?
        if critical.ongoing == 0 && critical.fringe.is_empty() {
            return WorkLoad::Complete;
        }
        // Nothing to do yet ? => Wait for someone to post jobs
        if critical.fringe.is_empty() {
            shared.monitor.wait(&mut critical);
            return WorkLoad::Starvation;
        }
        // Nothing relevant ? =>  Wait for someone to post jobs
        let nn = critical.fringe.pop().unwrap();
        if nn.info.ub < critical.best_lb {
            critical.fringe.clear();
            return WorkLoad::Starvation;
        }

        // Consume the current node and process it
        critical.ongoing += 1;
        critical.explored+= 1;

        WorkLoad::WorkItem {
            explored : critical.explored,
            fringe_sz: critical.fringe.len(),
            best_lb  : critical.best_lb,
            node     : nn
        }
    }

    fn maybe_log(verbosity: u8, explored: usize, fringe_sz: usize, lb: i32) {
        if verbosity >= 2 && explored % 100 == 0 {
            println!("Explored {}, LB {}, Fringe sz {}", explored, lb, fringe_sz);
        }
    }
}

impl <T, DD> Solver for ParallelSolver<T, DD> where T: Send, DD: MDD<T> + Clone + Send {

    fn maximize(&mut self) -> (i32, &Option<Vec<Decision>>) {
        self.initialize();

        crossbeam::thread::scope(|s|{
            for _i in 0..self.nb_threads {
                let shared    = Arc::clone(&self.shared);
                let mut mdd   = self.mdd.clone();
                let verbosity = self.verbosity;

                s.spawn(move |_| {
                    loop {
                        match Self::get_workload(&shared) {
                            WorkLoad::Complete   => break,
                            WorkLoad::Starvation => continue,
                            WorkLoad::WorkItem {explored, fringe_sz, best_lb, node} => {
                                Self::maybe_log(verbosity, explored, fringe_sz, best_lb);
                                Self::process_one_node(&mut mdd, &shared, node);
                            }
                        }
                    }
                });
            }
        }).expect("Something went wrong with the worker threads");

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