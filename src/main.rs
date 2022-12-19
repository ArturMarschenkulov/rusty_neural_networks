mod activation;
mod network;

use crate::network::*;
use ndarray as nd;

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
    let mut network = Network::new();
    network.gradient_descent(
        &train_pixels,
        &train_labels.map(|x| (*x as u64)),
        0.01,
        5000,
    );

    let prediction = network.make_prediction(&test_pixels);
    let accuracy = get_accuracy(&prediction, &test_labels.map(|x| (*x as u64)));
    print!("Accuracy: {}", accuracy);
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
