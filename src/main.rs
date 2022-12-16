mod t;

use ndarray as nd;

fn one_hot(y: &nd::Array1<u64>) -> nd::Array2<f64> {
    let y_size = y.len_of(nd::Axis(0));
    let y_max = *y.iter().max().unwrap() as usize;
    let mut one_hot_y = nd::Array2::zeros((y_max + 1, y_size));
    for (i, j) in y.iter().enumerate() {
        one_hot_y[[*j as usize, i]] = 1.0;
    }
    one_hot_y
}

#[derive(Clone)]
pub struct Activation<'a> {
    pub function: &'a dyn Fn(f64) -> f64,
    pub derivative: &'a dyn Fn(f64) -> f64,
}

pub const RELU: Activation = Activation {
    function: &|x| x.max(0.0),
    derivative: &|x| if x > 0.0 { 1.0 } else { 0.0 },
};
pub const LEAKY_RELU: Activation = Activation {
    function: &|x| if x > 0.0 { x } else { 0.01 * x },
    derivative: &|x| if x > 0.0 { 1.0 } else { 0.01 },
};
pub const IDENTITY: Activation = Activation {
    function: &|x| x,
    derivative: &|_| 1.0,
};
pub const SIGMOID: Activation = Activation {
    function: &|x| 1.0 / (1.0 + std::f64::consts::E.powf(-x)),
    derivative: &|x| x * (1.0 - x),
};
pub const TANH: Activation = Activation {
    function: &|x| x.tanh(),
    derivative: &|x| 1.0 - (x.powi(2)),
};

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
fn softmax(z: &nd::Array2<f64>) -> nd::Array2<f64> {
    let e = z.map(|x| x.exp());
    let sum = e.sum_axis(nd::Axis(0));
    e / sum
}

fn gradient_descent(x: &nd::Array2<f64>, y: &nd::Array1<u64>, alpha: f64, iterations: usize) -> () {
    let (mut w_1, mut b_1, mut w_2, mut b_2) = init_params();
    let mut max_accuracy = 0.0;
    let mut curr_milestone = 0.0;
    let milestones = [0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99];
    for i in 0..iterations {
        let (z_1, a_1, z_2, a_2) = forward_prop(&w_1, &b_1, &w_2, &b_2, x);
        let (dw_1, db_1, dw_2, db_2) = backward_prop(&z_1, &a_1, &z_2, &a_2, &w_1, &w_2, x, y);
        (w_1, b_1, w_2, b_2) =
            update_params(&w_1, &b_1, &w_2, &b_2, &dw_1, &db_1, &dw_2, &db_2, alpha);

        let prediction = get_predictions(&a_2);
        let accuracy = get_accuracy(&prediction, y);
        if accuracy > max_accuracy {
            max_accuracy = accuracy;
        }

        if accuracy > curr_milestone {
            println!("Iteration: {}", i);
            println!("Accuracy: {}", accuracy);
            println!("Max Accuracy: {}", max_accuracy);
            println!("---------------------");
            // curr_milestone = milestones[curr_milestone as usize];
        }

        // if i % 100 == 0 {
        //     println!("Iteration: {}", i);
        //     let prediction = get_predictions(&a_2);
        //     println!("Accuracy: {}", get_accuracy(&prediction, y));
        // }
    }
    // (w_1, b_1, w_2, b_2)
}

fn forward_prop(
    w_1: &nd::Array2<f64>,
    b_1: &nd::Array2<f64>,
    w_2: &nd::Array2<f64>,
    b_2: &nd::Array2<f64>,
    x: &nd::Array2<f64>,
) -> (
    nd::Array2<f64>,
    nd::Array2<f64>,
    nd::Array2<f64>,
    nd::Array2<f64>,
) {
    let act = LEAKY_RELU;
    // 0th layer (input)
    let z_1 = w_1.dot(x) + b_1;
    let a_1 = z_1.map(|x| (act.function)(*x));

    // 1rst layer (hidden)
    let z_2 = w_2.dot(&a_1) + b_2;
    let a_2 = softmax(&z_2);

    (z_1, a_1, z_2, a_2)
}

fn backward_prop(
    z_1: &nd::Array2<f64>,
    a_1: &nd::Array2<f64>,
    z_2: &nd::Array2<f64>,
    a_2: &nd::Array2<f64>,
    w_1: &nd::Array2<f64>,
    w_2: &nd::Array2<f64>,
    x: &nd::Array2<f64>,
    y: &nd::Array1<u64>,
) -> (nd::Array2<f64>, f64, nd::Array2<f64>, f64) {
    let m = 784.0;
    let act = LEAKY_RELU;

    // 1rst layer (hidden)
    let dz_2 = a_2 - one_hot(y);
    let dw_2 = dz_2.dot(&a_1.t()) / m as f64;
    let db_2 = dz_2.sum() / m;

    // 0th layer (input)
    let dz_1 = w_2.t().dot(&dz_2) * z_1.map(|x| (act.derivative)(*x));
    let dw_1 = dz_1.dot(&x.t()) / m as f64;
    let db_1 = dz_1.sum() / m;

    (dw_1, db_1, dw_2, db_2)
}
fn update_params(
    w_1: &nd::Array2<f64>,
    b_1: &nd::Array2<f64>,
    w_2: &nd::Array2<f64>,
    b_2: &nd::Array2<f64>,
    dw_1: &nd::Array2<f64>,
    db_1: &f64,
    dw_2: &nd::Array2<f64>,
    db_2: &f64,
    alpha: f64,
) -> (
    nd::Array2<f64>,
    nd::Array2<f64>,
    nd::Array2<f64>,
    nd::Array2<f64>,
) {
    let w_1 = w_1 - dw_1 * alpha;
    let b_1 = b_1 - db_1 * alpha;
    let w_2 = w_2 - dw_2 * alpha;
    let b_2 = b_2 - db_2 * alpha;
    (w_1, b_1, w_2, b_2)
}

fn init_params() -> (
    nd::Array2<f64>,
    nd::Array2<f64>,
    nd::Array2<f64>,
    nd::Array2<f64>,
) {
    use rand::Rng;

    let mut rng = rand::thread_rng();
    let range = -0.5..=0.5;

    // 0th layer (input)
    let weight_1 = nd::Array2::<f64>::zeros((10, 784)).map(|_| rng.gen_range(range.clone()));
    let bias_1 = nd::Array2::<f64>::zeros((10, 1)).map(|_| rng.gen_range(range.clone()));

    // 1rst layer (hidden)
    let weight_2 = nd::Array2::<f64>::zeros((10, 10)).map(|_| rng.gen_range(range.clone()));
    let bias_2 = nd::Array2::<f64>::zeros((10, 1)).map(|_| rng.gen_range(range.clone()));

    (weight_1, bias_1, weight_2, bias_2)
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
    println!("mnist_pics: {:?}", mnist_pics);
    println!("mnist_pics: {:?}", mnist_pics.slice(nd::s![..=0, ..]));
    let dev_size: i32 = 1000;

    // slice `mnist_pics` from 0 to 1000
    let mnist_pics_test = mnist_pics.slice(nd::s![0..dev_size, ..]);
    let data_test = mnist_pics_test.t();

    let y_test = data_test.slice(nd::s![0, ..]);

    println!("data_test: {:?}", data_test);
    println!("y_test: {:?}", y_test);
    let x_test = data_test.slice(nd::s![1..785, ..]);
    println!("x_test: {:?}", x_test);
    let x_test = x_test.map(|x| *x / 255.0);

    let mnist_pics_train = mnist_pics.slice(nd::s![dev_size.., ..]);
    let data_train = mnist_pics_train.t();
    let y_train = data_train.slice(nd::s![0, ..]);
    let x_train = data_train.slice(nd::s![1..785, ..]);
    let x_train = x_train.map(|x| *x / 255.0);

    println!("y_train: {:?}", y_train);
    gradient_descent(&x_train, &y_train.map(|x| (*x as u64)), 0.10, 5000);
}
