mod activation;

use crate::activation::*;
use ndarray as nd;

fn one_hot(y: &nd::Array1<u64>) -> nd::Array2<f64> {
    let y_size = y.len_of(nd::Axis(0));
    let y_max = *y.iter().max().unwrap() as usize;
    let mut one_hot_y = nd::Array2::zeros((y_max + 1, y_size));
    for (i, j) in y.iter().enumerate() {
        one_hot_y[[*j as usize, i]] = 1.0;
    }
    assert_eq!(one_hot_y.shape()[0], 10);
    one_hot_y
}

fn argmax(a: &nd::Array2<f64>) -> nd::Array1<u64> {
    let mut result = nd::Array1::zeros(a.shape()[1]);
    for i in 0..a.shape()[1] {
        let mut max = 0.0;
        let mut max_index = 0;
        for j in 0..a.shape()[0] {
            if a.get((j, i)).unwrap() > &max {
                max = *a.get((j, i)).unwrap();
                max_index = j;
            }
        }
        result[i] = max_index as u64;
    }
    result
}

fn get_predictions(a_2: &nd::Array2<f64>) -> nd::Array1<u64> {
    argmax(a_2)
}
fn get_accuracy(predictions: &nd::Array1<u64>, y: &nd::Array1<u64>) -> f64 {
    // give an array of true and false
    let bools = predictions
        .iter()
        .zip(y.iter())
        .map(|(x, y)| x == &(*y as u64))
        .collect::<Vec<_>>();

    // count the number of true
    let true_count = bools.iter().filter(|x| **x).count();
    true_count as f64 / y.len() as f64
}

#[derive(Debug, Clone, PartialEq)]
struct WeightAndBias {
    weight: nd::Array2<f64>,
    bias: nd::Array2<f64>,
}

#[derive(Clone)]
struct ZAndA {
    unactivated: nd::Array2<f64>,
    activated: nd::Array2<f64>,
}
fn gradient_descent(x: &nd::Array2<f64>, y: &nd::Array1<u64>, alpha: f64, iterations: usize) {
    let example_num = x.len_of(nd::Axis(1));
    let data_inputs = x.len_of(nd::Axis(0));
    let multiplier = data_inputs as f32 / example_num as f32;
    println!("Example num: {}", example_num);
    println!("Data inputs: {}", data_inputs);
    println!("Multiplier: {}", multiplier);
    let mut weights_and_biases = init_params();
    let mut max_accuracy = 0.0;
    let curr_milestone = 0.0;
    let mut max_accuracy_iteration = 0;
    for i in 0..iterations {
        let perceptrons = forward_prop(&weights_and_biases, x);

        let zipped: Vec<(ZAndA, nd::Array2<f64>)> = weights_and_biases
            .iter()
            .zip(perceptrons.iter())
            .map(|(wb, perceptron)| (perceptron.clone(), wb.weight.clone()))
            .collect();

        let delta_weights_and_biases = backward_prop(&zipped, x, y);

        let zipped_lbs: Vec<(WeightAndBias, WeightAndBias)> = weights_and_biases
            .iter()
            .zip(delta_weights_and_biases.iter())
            .map(|(wb, dwb)| (wb.clone(), dwb.clone()))
            .collect();
        weights_and_biases = update_params(&zipped_lbs, alpha);

        let prediction = get_predictions(&perceptrons.last().unwrap().activated);
        let accuracy = get_accuracy(&prediction, y);
        if accuracy > max_accuracy {
            max_accuracy = accuracy;
            max_accuracy_iteration = i;
        }

        if accuracy > curr_milestone {
            println!("Iteration: {}", i);
            println!("Accuracy: {}", accuracy);
            println!(
                "Max Accuracy: {} since {} (d {})",
                max_accuracy,
                max_accuracy_iteration,
                i - max_accuracy_iteration
            );
            println!("---------------------");
            // curr_milestone = milestones[curr_milestone as usize];
        }

        // if i % 100 == 0 {
        //     println!("Iteration: {}", i);
        //     let prediction = get_predictions(&a_2);
        //     println!("Accuracy: {}", get_accuracy(&prediction, y));
        // }
    }
}

fn forward_prop(wbs: &Vec<WeightAndBias>, x: &nd::Array2<f64>) -> Vec<ZAndA> {
    let perceptrons_a_0 = x; // That is the input

    let mut result = Vec::<ZAndA>::new();
    let mut last_activated_perceptrons = perceptrons_a_0.clone();
    for weights_and_biases in wbs {
        let activation = if weights_and_biases == wbs.last().unwrap() {
            Activation::Softmax
        } else {
            Activation::LeakyRelu
        };

        let perceptrons_u = weights_and_biases.weight.dot(&last_activated_perceptrons)
            + weights_and_biases.bias.clone();
        let perceptrons_a = activation.forward(&perceptrons_u);

        last_activated_perceptrons = perceptrons_a.clone();

        result.push(ZAndA {
            unactivated: perceptrons_u,
            activated: perceptrons_a,
        });
    }
    result
}

fn backward_prop(
    zaws: &Vec<(ZAndA, nd::Array2<f64>)>,
    x: &nd::Array2<f64>,
    y: &nd::Array1<u64>,
) -> Vec<WeightAndBias> {
    use std::collections::VecDeque;
    let m = 784.0;
    let perceptrons = zaws
        .iter()
        .map(|(zaw, _)| zaw.clone())
        .collect::<Vec<ZAndA>>();
    let weights = zaws
        .iter()
        .map(|(_, weight)| weight.clone())
        .collect::<Vec<nd::Array2<f64>>>();

    let mut last_d_perceptrons_u = zaws.last().unwrap().0.activated.clone() - one_hot(y);

    let mut delta_weights_and_biases = VecDeque::new();
    for (i, _) in zaws.iter().enumerate().rev() {
        let is_last = i == zaws.len() - 1;
        let is_first = i == 0;

        let activation = Activation::LeakyRelu;

        let d_perceptrons_u = if is_last {
            last_d_perceptrons_u
        } else {
            weights[i + 1].t().dot(&last_d_perceptrons_u)
                * activation.backward(&perceptrons[i].unactivated)
        };

        let d_weights = if is_first {
            d_perceptrons_u.dot(&x.t()) / m as f64
        } else {
            d_perceptrons_u.dot(&perceptrons[i - 1].activated.t()) / m as f64
        };

        let d_biases = d_perceptrons_u
            .sum_axis(nd::Axis(1))
            .into_shape((d_perceptrons_u.shape()[0], 1))
            .unwrap()
            / m;

        last_d_perceptrons_u = d_perceptrons_u;
        delta_weights_and_biases.push_front(WeightAndBias {
            weight: d_weights,
            bias: d_biases,
        });
    }

    // convert a `DequeVec` into a `Vec`
    delta_weights_and_biases.into_iter().collect()
}

fn update_params(lds: &[(WeightAndBias, WeightAndBias)], alpha: f64) -> Vec<WeightAndBias> {
    let mut result = Vec::new();
    for (_, ld) in lds.iter().enumerate() {
        // assert_eq!(ld.0.weight.shape(), ld.1.weight.shape());
        // assert_eq!(ld.0.bias.shape(), ld.1.bias.shape());

        let weights = ld.0.weight.clone() - ld.1.weight.clone() * alpha;
        let biases = ld.0.bias.clone() - ld.1.bias.clone() * alpha;
        result.push(WeightAndBias {
            weight: weights,
            bias: biases,
        });
    }
    result
}

fn init_params() -> Vec<WeightAndBias> {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let range = -0.5..=0.5;

    // For right now, this array has to begin with 784 and end with 10. They represent the inpuit and output layer.
    // Later this will be refactored to be more flexible.
    let layers_num = [784, 5, 10];

    let mut weights_and_biases = Vec::new();

    let mut i = 0;
    while i < layers_num.len() - 1 {
        let weights = nd::Array2::<f64>::zeros((layers_num[i + 1], layers_num[i]))
            .map(|_| rng.gen_range(range.clone()));
        let biases =
            nd::Array2::<f64>::zeros((layers_num[i + 1], 1)).map(|_| rng.gen_range(range.clone()));
        weights_and_biases.push(WeightAndBias {
            weight: weights,
            bias: biases,
        });

        i += 1;
    }
    weights_and_biases
}

fn get_mnist_data_2() -> ndarray::Array2<f64> {
    use std::fs::File;
    use std::io::BufReader;

    let file = File::open("mnist_test.csv").unwrap();
    let mut rdr = csv::Reader::from_reader(BufReader::new(file));

    let records = rdr
        .records()
        .map(|result| result.unwrap())
        .collect::<Vec<_>>();

    let mut res = ndarray::Array2::<f64>::zeros((records.len(), 785));
    for (i, record) in records.iter().enumerate() {
        for (j, pixel) in record.iter().enumerate() {
            res[[i, j]] = pixel.parse::<f64>().unwrap();
        }
    }
    res
}

fn main() {
    // Enable RUST_BACKTRACE=1 to see the backtrace
    std::env::set_var("RUST_BACKTRACE", "1");

    let mnist_pics = get_mnist_data_2();

    println!("Mnist pics were loaded");
    println!("mnist_pics: {:?}", mnist_pics.shape());
    let m = mnist_pics.shape()[0];
    let n = mnist_pics.shape()[1];

    let dev_size: i32 = 1000;

    // slice `mnist_pics` from 0 to 1000
    let mnist_pics_test = mnist_pics.slice(nd::s![0..dev_size, ..]);

    let data_test = mnist_pics_test.t();
    let test_labels = data_test.slice(nd::s![0, ..]);
    let test_pixels = data_test.slice(nd::s![1..n, ..]);
    let test_pixels = test_pixels.map(|x| *x / 255.0);

    let mnist_pics_train = mnist_pics.slice(nd::s![dev_size..m as i32, ..]);
    let data_train = mnist_pics_train.t();
    let train_labels = data_train.slice(nd::s![0, ..]);
    let train_pixels = data_train.slice(nd::s![1..n, ..]);
    let train_pixels = train_pixels.map(|x| *x / 255.0);

    println!("x_train: {:?}", train_pixels.shape());
    println!("y_train: {:?}", train_labels);
    gradient_descent(
        &train_pixels,
        &train_labels.map(|x| (*x as u64)),
        0.01,
        5000,
    );
}

/* Here are some learning resources.
    - https://www.kaggle.com/code/wwsalmon/simple-mnist-nn-from-scratch-numpy-no-tf-keras/notebook
    - https://www.youtube.com/watch?v=w8yWXqWQYmU
    - https://www.youtube.com/watch?v=9RN2Wr8xvro&t=463s

    - http://neuralnetworksanddeeplearning.com/chap4.html
    - https://explog.in/notes/funnn.html
    - https://python-course.eu/machine-learning/neural-networks-structure-weights-and-matrices.php


    Rust github projects:
    - https://github.dev/danhper/rust-simple-nn
    - https://github.com/daniel-e/rustml
    - https://github.com/pipehappy1/auto-diff
*/
