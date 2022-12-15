//! A naive neural network implementation
//! with all fully connected layers

//! Starting with a single layer net,
//! with one input and one output
//! ```
//!          0    1
//!          .    .
//!          .    .          neuron position
//!                               -v--v-
//!      0.. o <- n[0,0] = relu(w[0, 0, 0]x + b[0, 0, 0])
//!    /       \                        -^- input position
//! x -- 1.. o -- y <- relu(sum(w[1, x, i]*n[0,i] + b[1, x, i]))
//!    \       /
//!      2.. o
//!
//!
//! ```

use rand::thread_rng;
use rand::{distributions::Standard, Rng};
use std::cmp::min;
use std::time::Instant;

/// Data structure to hold the net
struct Net {
    ws: Vec<f64>,
    bs: Vec<f64>,
    ns: usize,
}

impl Net {
    /// Create a fully-connected net with hidden layer size
    pub fn new(ns: usize) -> Net {
        let size = ns * 2;
        let ws: Vec<f64> = thread_rng().sample_iter(Standard).take(size).collect();
        let bs: Vec<f64> = thread_rng().sample_iter(Standard).take(size).collect();

        Net { ws, bs, ns }
    }

    /// Calculates an index into the weights/biases vector
    /// for a given net
    pub fn pt(self: &Self, x: usize, y: usize, z: usize) -> usize {
        match x {
            0 if z == 0 && y < self.ns => y,
            1 if y == 0 && z < self.ns => self.ns + z,
            _ => panic!("Invalid location: {}, {}, {}", x, y, z),
        }
    }

    pub fn train(
        self: &mut Self,
        training_data: &[(f64, f64)],
        epochs: usize,
        batch_size: usize,
        learning_rate: f64,
    ) {
        let log_interval = epochs / 10;

        for epoch in 0..epochs {
            let mut point = 0;
            while point <= training_data.len() {
                let limit = min(point + batch_size, training_data.len());
                self.backprop(&training_data[point..limit], learning_rate);
                point += batch_size;
            }

            if log_interval > 0 && epoch % log_interval == 0 {
                eprintln!("Epoch {}: {}", epoch, self.cost(training_data));
            }
        }
    }

    pub fn cost(self: &Self, data: &[(f64, f64)]) -> f64 {
        let mut loss = 0.0;
        for (x, y) in data {
            let val = self.eval(*x);
            loss += (y - val).powi(2);
        }
        loss / self.ns as f64
    }

    fn backprop(self: &mut Self, data: &[(f64, f64)], learning_rate: f64) {
        let mut dws: Vec<f64> = vec![0.0; self.ns * 2];
        let mut dbs: Vec<f64> = vec![0.0; self.ns * 2];

        for i in 0..self.ns {
            let pt1 = self.pt(0, i, 0);
            let pt2 = self.pt(1, 0, i);

            for (x, y) in data {
                let yy = self.eval(*x);

                dws[pt2] += -2.0 * (y - yy) * relu_ish(self.rwxb(*x, 0, i, 0), yy);
                dbs[pt2] += -2.0 * (y - yy) * relu_ish(1.0, yy);

                dws[pt1] += -2.0
                    * (y - yy)
                    * relu_ish(self.ws[pt2] * relu_ish(*x, self.wxb(*x, 0, i, 0)), yy);
                dbs[pt1] += -2.0
                    * (y - yy)
                    * relu_ish(self.ws[pt2] * relu_ish(1.0, self.wxb(*x, 0, i, 0)), yy);
            }
        }

        for i in 0..self.ns {
            for pt in &[self.pt(1, 0, i), self.pt(0, i, 0)] {
                self.ws[*pt] -= dws[*pt] * learning_rate;
                self.bs[*pt] -= dbs[*pt] * learning_rate;
            }
        }
    }

    pub fn eval(self: &Self, val: f64) -> f64 {
        relu(
            (0..self.ns)
                .map(|i| self.rwxb(self.rwxb(val, 0, i, 0), 1, 0, i))
                .sum(),
        )
    }

    /// Relu(wx + b) for coordinates x, y, z with input val
    fn rwxb(self: &Self, val: f64, x: usize, y: usize, z: usize) -> f64 {
        relu(self.wxb(val, x, y, z))
    }

    /// wx + b for coordinates x, y, z with input val
    fn wxb(self: &Self, val: f64, x: usize, y: usize, z: usize) -> f64 {
        self.w(x, y, z) * val + self.b(x, y, z)
    }

    fn w(self: &Self, x: usize, y: usize, z: usize) -> f64 {
        self.ws[self.pt(x, y, z)]
    }

    fn b(self: &Self, x: usize, y: usize, z: usize) -> f64 {
        self.bs[self.pt(x, y, z)]
    }
}

/// Leaky relu
fn relu(v: f64) -> f64 {
    relu_ish(v, v)
}

/// Leaky relu based on another variable, useful for derivatives
fn relu_ish(v: f64, point: f64) -> f64 {
    if point >= 0.0 {
        v
    } else {
        0.01 * v
    }
}

pub fn main() {
    fn original_fn(x: f64) -> f64 {
        x * x * x + x * x + x
    };

    let training_data: Vec<(f64, f64)> = (1..=100)
        .step_by(7)
        .map(|x| (x as f64) / 100.0)
        .map(|x| (x, original_fn(x)))
        .collect();

    let validation_data: Vec<(f64, f64)> = (20..=60)
        .map(|x| (x as f64) / 100.0)
        .map(|x| (x, original_fn(x)))
        .collect();

    let start = Instant::now();
    let mut net = Net::new(20);
    net.train(&training_data, 100000, 100, 0.000001);
    eprintln!("Training duration: {}s", start.elapsed().as_secs());
    eprintln!("Validation error: {}", net.cost(&validation_data));

    for x in 0..1000 {
        let x = x as f64 / 1000.0;
        println!("{}\t{}\t{}", x, original_fn(x), net.eval(x));
    }
}
