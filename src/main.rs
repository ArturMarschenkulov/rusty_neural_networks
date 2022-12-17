mod t;

use ndarray as nd;
use rand::distributions::weighted::alias_method::Weight;

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
// pub const SOFTMAX: Activation = Activation {
//     function: &|x| x.exp() / x.exp().sum(),
//     derivative: &|x| x * (1.0 - x),
// };

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

struct WeightAndBias {
    weight: nd::Array2<f64>,
    bias: nd::Array2<f64>,
}
struct ZAndA {
    z: nd::Array2<f64>,
    a: nd::Array2<f64>,
}
fn gradient_descent(x: &nd::Array2<f64>, y: &nd::Array1<u64>, alpha: f64, iterations: usize) -> () {
    let (mut wb_1, mut wb_2, mut wb_3) = init_params();
    let mut max_accuracy = 0.0;
    let mut curr_milestone = 0.0;
    let milestones = [0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99];
    for i in 0..iterations {
        // let a_0 = x.map(|x| *x);

        let (perceptrons_1, perceptrons_2, mut perceptrons_3) =
            forward_prop(&wb_1, &wb_2, &wb_3, x);
        // perceptrons_3.a = softmax(&perceptrons_3.z);

        let (dwb_1, dwb_2, dwb_3) = backward_prop(
            (&perceptrons_1, &wb_1.weight),
            (&perceptrons_2, &wb_2.weight),
            (&perceptrons_3, &wb_3.weight),
            x,
            y,
        );
        (wb_1, wb_2, wb_3) = update_params(&wb_1, &wb_2, &wb_3, &dwb_1, &dwb_2, &dwb_3, alpha);

        let prediction = get_predictions(&perceptrons_3.a);
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
    l_1: &WeightAndBias,
    l_2: &WeightAndBias,
    l_3: &WeightAndBias,
    x: &nd::Array2<f64>,
) -> (ZAndA, ZAndA, ZAndA) {
    let act = LEAKY_RELU;
    let act_2 = IDENTITY;
    // 1th layer (hidden)
    let perceptrons_u_1 = l_1.weight.dot(x) + l_1.bias.clone();
    let perceptrons_a_1 = perceptrons_u_1.map(|x| (act.function)(*x));

    // 2th layer (hidden)
    let perceptrons_u_2 = l_2.weight.dot(&perceptrons_a_1) + l_2.bias.clone();
    let perceptrons_a_2 = perceptrons_u_2.map(|x| (act.function)(*x));

    // 3th layer (output)
    let perceptrons_u_3 = l_3.weight.dot(&perceptrons_a_2) + l_3.bias.clone();
    let perceptrons_a_3 = perceptrons_u_3.map(|x| (act_2.function)(*x));
    let perceptrons_a_3 = softmax(&perceptrons_u_3);

    (
        ZAndA {
            z: perceptrons_u_1,
            a: perceptrons_a_1,
        },
        ZAndA {
            z: perceptrons_u_2,
            a: perceptrons_a_2,
        },
        ZAndA {
            z: perceptrons_u_3,
            a: perceptrons_a_3,
        },
    )
}

fn backward_prop(
    zaw_1: (&ZAndA, &nd::Array2<f64>),
    zaw_2: (&ZAndA, &nd::Array2<f64>),
    zaw_3: (&ZAndA, &nd::Array2<f64>),
    x: &nd::Array2<f64>,
    y: &nd::Array1<u64>,
) -> (WeightAndBias, WeightAndBias, WeightAndBias) {
    let m = 784.0;
    let act = LEAKY_RELU;

    // 3th layer (hidden)
    let dz_3 = zaw_3.0.a.clone() - one_hot(y);
    let dw_3 = dz_3.dot(&zaw_2.0.a.t()) / m as f64;
    let db_3 = dz_3
        .sum_axis(nd::Axis(1))
        .into_shape((dz_3.shape()[0], 1))
        .unwrap()
        / m;

    // 2th layer (hidden)
    let dz_2 = zaw_3.1.t().dot(&dz_3) * zaw_2.0.z.map(|x| (act.derivative)(*x));
    let dw_2 = dz_2.dot(&zaw_1.0.a.t()) / m as f64;
    let db_2 = dz_2
        .sum_axis(nd::Axis(1))
        .into_shape((dz_2.shape()[0], 1))
        .unwrap()
        / m;

    // 1th layer (input)
    let dz_1 = zaw_2.1.t().dot(&dz_2) * zaw_1.0.z.map(|x| (act.derivative)(*x));
    let dw_1 = dz_1.dot(&x.t()) / m as f64;
    let db_1 = dz_1
        .sum_axis(nd::Axis(1))
        .into_shape((dz_1.shape()[0], 1))
        .unwrap()
        / m;

    (
        WeightAndBias {
            weight: dw_1,
            bias: db_1,
        },
        WeightAndBias {
            weight: dw_2,
            bias: db_2,
        },
        WeightAndBias {
            weight: dw_3,
            bias: db_3,
        },
    )
}

struct Perceptron {
    weights: nd::Array2<f64>,
    bias: nd::Array2<f64>,
    value: nd::Array2<f64>,
}
fn update_params(
    l_1: &WeightAndBias,
    l_2: &WeightAndBias,
    l_3: &WeightAndBias,
    d_1: &WeightAndBias,
    d_2: &WeightAndBias,
    d_3: &WeightAndBias,
    alpha: f64,
) -> (WeightAndBias, WeightAndBias, WeightAndBias) {
    // 1th layer (hidden)
    let weight_1 = l_1.weight.clone() - d_1.weight.clone() * alpha;
    let bias_1 = l_1.bias.clone() - d_1.bias.clone() * alpha;

    // 2th layer (hidden)
    let weight_2 = l_2.weight.clone() - d_2.weight.clone() * alpha;
    let bias_2 = l_2.bias.clone() - d_2.bias.clone() * alpha;

    // 3th layer (output)
    let weight_3 = l_3.weight.clone() - d_3.weight.clone() * alpha;
    let bias_3 = l_3.bias.clone() - d_3.bias.clone() * alpha;

    (
        WeightAndBias {
            weight: weight_1,
            bias: bias_1,
        },
        WeightAndBias {
            weight: weight_2,
            bias: bias_2,
        },
        WeightAndBias {
            weight: weight_3,
            bias: bias_3,
        },
    )
}

fn init_params() -> (WeightAndBias, WeightAndBias, WeightAndBias) {
    use rand::Rng;

    let mut rng = rand::thread_rng();
    let range = -0.5..=0.5;

    // (n, m) m would be the amount of neurons on the left, while n would be the amount of neurons on the right

    let layer_0_num = 784;
    let layer_1_num = 10;
    let layer_2_num = 50;
    let layer_3_num = 10;

    // 1th layer (hidden)
    let weight_1 =
        nd::Array2::<f64>::zeros((layer_1_num, layer_0_num)).map(|_| rng.gen_range(range.clone()));
    let bias_1 = nd::Array2::<f64>::zeros((layer_1_num, 1)).map(|_| rng.gen_range(range.clone()));

    // 2th layer (output)
    let weight_2 =
        nd::Array2::<f64>::zeros((layer_2_num, layer_1_num)).map(|_| rng.gen_range(range.clone()));
    let bias_2 = nd::Array2::<f64>::zeros((layer_2_num, 1)).map(|_| rng.gen_range(range.clone()));

    // 3th layer (output)
    let weight_3 =
        nd::Array2::<f64>::zeros((layer_3_num, layer_2_num)).map(|_| rng.gen_range(range.clone()));
    let bias_3 = nd::Array2::<f64>::zeros((layer_3_num, 1)).map(|_| rng.gen_range(range.clone()));

    (
        WeightAndBias {
            weight: weight_1,
            bias: bias_1,
        },
        WeightAndBias {
            weight: weight_2,
            bias: bias_2,
        },
        WeightAndBias {
            weight: weight_3,
            bias: bias_3,
        },
    )
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
    let y_test = data_test.slice(nd::s![0, ..]);
    let x_test = data_test.slice(nd::s![1..n, ..]);
    // println!("x_test: {:?}", x_test.shape());
    // return;
    let x_test = x_test.map(|x| *x / 255.0);

    let mnist_pics_train = mnist_pics.slice(nd::s![dev_size..m as i32, ..]);
    let data_train = mnist_pics_train.t();
    let y_train = data_train.slice(nd::s![0, ..]);
    let x_train = data_train.slice(nd::s![1..n, ..]);
    let x_train = x_train.map(|x| *x / 255.0);

    println!("x_train: {:?}", x_train.shape());
    println!("y_train: {:?}", y_train);
    gradient_descent(&x_train, &y_train.map(|x| (*x as u64)), 0.01, 5000);
}

/* Here are some learning resources.
    https://www.kaggle.com/code/wwsalmon/simple-mnist-nn-from-scratch-numpy-no-tf-keras/notebook
    https://www.youtube.com/watch?v=w8yWXqWQYmU
    https://www.youtube.com/watch?v=9RN2Wr8xvro&t=463s
*/
