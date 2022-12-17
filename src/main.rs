// mod t;

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

#[derive(Clone)]
struct WeightAndBias {
    weight: nd::Array2<f64>,
    bias: nd::Array2<f64>,
}

#[derive(Clone)]
struct ZAndA {
    z: nd::Array2<f64>,
    a: nd::Array2<f64>,
}
fn gradient_descent(x: &nd::Array2<f64>, y: &nd::Array1<u64>, alpha: f64, iterations: usize) -> () {
    let mut wbs = init_params();
    let mut max_accuracy = 0.0;
    let mut curr_milestone = 0.0;
    let milestones = [0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99];
    for i in 0..iterations {
        // let a_0 = x.map(|x| *x);

        let perceptrons = forward_prop(&wbs, x);
        // perceptrons_3.a = softmax(&perceptrons_3.z);

        // zip wbs and perceptrons together into a vec of tuples. However wbs should only include its `weight` field
        let zipped = wbs
            .iter()
            .zip(perceptrons.iter())
            .map(|(wb, perceptron)| (perceptron.clone(), wb.weight.clone()))
            .collect::<Vec<_>>();

        assert_eq!(zipped.len(), 3);

        let dwbs = backward_prop(&zipped, x, y);
        assert_eq!(dwbs.len(), 3);
        let zipped_lbs = wbs
            .iter()
            .zip(dwbs.iter())
            .map(|(wb, dwb)| (wb.clone(), dwb.clone()))
            .collect::<Vec<_>>();
        assert_eq!(zipped_lbs.len(), 3);
        wbs = update_params(&zipped_lbs, alpha);
        assert_eq!(wbs.len(), 3);

        let prediction = get_predictions(&perceptrons.last().unwrap().a);
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

fn forward_prop(wbs: &Vec<WeightAndBias>, x: &nd::Array2<f64>) -> Vec<ZAndA> {
    let act = LEAKY_RELU;

    assert_eq!(wbs.len(), 3);
    let l_1 = &wbs[0];
    let l_2 = &wbs[1];
    let l_3 = &wbs[2];

    // 1th layer (hidden)
    let perceptrons_u_1 = l_1.weight.dot(x) + l_1.bias.clone();
    let perceptrons_a_1 = perceptrons_u_1.map(|x| (act.function)(*x));

    // 2th layer (hidden)
    let perceptrons_u_2 = l_2.weight.dot(&perceptrons_a_1) + l_2.bias.clone();
    let perceptrons_a_2 = perceptrons_u_2.map(|x| (act.function)(*x));

    // 3th layer (output)
    let perceptrons_u_3 = l_3.weight.dot(&perceptrons_a_2) + l_3.bias.clone();
    // let perceptrons_a_3 = perceptrons_u_3.map(|x| (act.function)(*x));
    let perceptrons_a_3 = softmax(&perceptrons_u_3);

    vec![
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
    ]
}

fn backward_prop(
    zaws: &Vec<(ZAndA, nd::Array2<f64>)>,
    x: &nd::Array2<f64>,
    y: &nd::Array1<u64>,
) -> Vec<WeightAndBias> {
    let m = 784.0;
    let act = LEAKY_RELU;

    assert_eq!(zaws.len(), 3);
    let zaw_1 = &zaws[0];
    let zaw_2 = &zaws[1];
    let zaw_3 = &zaws[2];

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

    vec![
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
    ]
}

struct Perceptron {
    weights: nd::Array2<f64>,
    bias: nd::Array2<f64>,
    value: nd::Array2<f64>,
}
fn update_params(lds: &Vec<(WeightAndBias, WeightAndBias)>, alpha: f64) -> Vec<WeightAndBias> {
    assert_eq!(lds.len(), 3);
    let l_1 = &lds[0].0;
    let l_2 = &lds[1].0;
    let l_3 = &lds[2].0;
    let d_1 = &lds[0].1;
    let d_2 = &lds[1].1;
    let d_3 = &lds[2].1;

    // 1th layer (hidden)
    let weight_1 = l_1.weight.clone() - d_1.weight.clone() * alpha;
    let bias_1 = l_1.bias.clone() - d_1.bias.clone() * alpha;

    // 2th layer (hidden)
    let weight_2 = l_2.weight.clone() - d_2.weight.clone() * alpha;
    let bias_2 = l_2.bias.clone() - d_2.bias.clone() * alpha;

    // 3th layer (output)
    let weight_3 = l_3.weight.clone() - d_3.weight.clone() * alpha;
    let bias_3 = l_3.bias.clone() - d_3.bias.clone() * alpha;

    vec![
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
    ]
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

fn init_params() -> Vec<WeightAndBias> {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let range = -0.5..=0.5;

    let layers_num = [784, 10, 50, 10];

    let mut wbs = Vec::new();

    let mut i = 0;
    while i < layers_num.len() - 1 {
        let w = nd::Array2::<f64>::zeros((layers_num[i + 1], layers_num[i]))
            .map(|_| rng.gen_range(range.clone()));
        let b =
            nd::Array2::<f64>::zeros((layers_num[i + 1], 1)).map(|_| rng.gen_range(range.clone()));
        wbs.push(WeightAndBias { weight: w, bias: b });

        i += 1;
    }
    wbs
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
    let _y_test = data_test.slice(nd::s![0, ..]);
    let x_test = data_test.slice(nd::s![1..n, ..]);
    // println!("x_test: {:?}", x_test.shape());
    // return;
    let _x_test = x_test.map(|x| *x / 255.0);

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
