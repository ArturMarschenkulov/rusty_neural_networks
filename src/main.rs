mod t;

use ndarray as nd;

fn gradient_descent(x: &nd::Array2<f64>, y: &nd::Array1<f64>, alpha: f64, iterations: usize) -> () {
    let (mut w_1, mut b_1, mut w_2, mut b_2) = init_params();

    for i in 0..iterations {
        let (z_1, a_1, z_2, a_2) = forward_prop(&w_1, &b_1, &w_2, &b_2, x);
        let (dw_1, db_1, dw_2, db_2) = back_prop(&z_1, &a_1, &z_2, &a_2, &w_1, &w_2, x, y);
        (w_1, b_1, w_2, b_2) =
            update_params(&w_1, &b_1, &w_2, &b_2, &dw_1, &db_1, &dw_2, &db_2, alpha);
        if i % 2 == 0 {
            println!("Iteration: {}", i);
            let prediction = get_predictions(&a_2);
            println!("Accuracy: {}", get_accuracy(&prediction, y));
        }
    }
    // (w_1, b_1, w_2, b_2)
}

fn argmax(a: &nd::Array2<f64>) -> nd::Array1<f64> {
    let mut res = nd::Array1::zeros(a.len_of(nd::Axis(1)));
    for (i, j) in a.genrows().into_iter().enumerate() {
        let mut max = 0.0;
        let mut max_index = 0;
        for (k, l) in j.iter().enumerate() {
            if *l > max {
                max = *l;
                max_index = k;
            }
        }
        res[i] = max_index as f64;
    }
    res
}

fn get_predictions(a_2: &nd::Array2<f64>) -> nd::Array1<f64> {
    argmax(&a_2)
}
fn get_accuracy(predictions: &nd::Array1<f64>, y: &nd::Array1<f64>) -> f64 {
    // give an array of true and false
    let bools = predictions
        .iter()
        .zip(y.iter())
        .map(|(x, y)| x == y)
        .collect::<Vec<_>>();

    // count the number of true
    let true_count = bools.iter().filter(|x| **x).count();
    true_count as f64 / y.len() as f64
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
    let z_1 = w_1.dot(x) + b_1;
    let a_1 = z_1.map(|x| relu(*x));
    let z_2 = w_2.dot(&a_1) + b_2;
    let a_2 = z_2.map(|x| relu(*x));
    (z_1, a_1, z_2, a_2)
}

fn back_prop(
    z_1: &nd::Array2<f64>,
    a_1: &nd::Array2<f64>,
    z_2: &nd::Array2<f64>,
    a_2: &nd::Array2<f64>,
    w_1: &nd::Array2<f64>,
    w_2: &nd::Array2<f64>,
    x: &nd::Array2<f64>,
    y: &nd::Array1<f64>,
) -> (nd::Array2<f64>, f64, nd::Array2<f64>, f64) {
    let m = 785.0;
    let one_hot_y = one_hot(y);
    let dz_2 = a_2 - one_hot_y;
    let dw_2 = dz_2.dot(&a_1.t()) / m as f64;
    let db_2 = dz_2.sum() / m;

    let dz_1 = w_2.t().dot(&dz_2).map(|x| relu_deriv(*x));
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

fn one_hot(y: &nd::Array1<f64>) -> nd::Array2<f64> {
    let mut one_hot_y = nd::Array2::zeros((10, y.len_of(nd::Axis(0))));
    for (i, j) in y.iter().enumerate() {
        one_hot_y[[*j as usize, i]] = 1.0;
    }
    one_hot_y
}

fn relu(x: f64) -> f64 {
    if x > 0.0 {
        x
    } else {
        0.0
    }
}
fn relu_deriv(x: f64) -> f64 {
    if x > 0.0 {
        1.0
    } else {
        0.0
    }
}
#[derive(Debug, Clone)]
struct MnistPic {
    label: u8,
    pixels: Vec<f64>,
}
impl MnistPic {
    fn new(label: u8, pixels: &Vec<f64>) -> Self {
        assert_eq!(pixels.len(), 784, "Pixel vector must be 784 long");
        MnistPic {
            label,
            pixels: pixels.clone(),
        }
    }
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

    let w_1 = nd::Array2::<f64>::zeros((10, 784)).map(|_| rng.gen_range(range.clone()));
    let b_1 = nd::Array2::<f64>::zeros((10, 1)).map(|_| rng.gen_range(range.clone()));
    let w_2 = nd::Array2::<f64>::zeros((10, 10)).map(|_| rng.gen_range(range.clone()));
    let b_2 = nd::Array2::<f64>::zeros((10, 1)).map(|_| rng.gen_range(range.clone()));
    (w_1, b_1, w_2, b_2)
}
fn to_data(records: Vec<csv::StringRecord>) -> Vec<Vec<f64>> {
    let mut res: Vec<Vec<f64>> = Vec::new();
    for i in 0..records.len() {
        let mut row: Vec<f64> = Vec::new();
        for j in 0..records[0].len() {
            row.push(records[i][j].parse::<f64>().unwrap());
        }
        res.push(row);
    }
    res;
    records
        .into_iter()
        .map(|record| {
            record
                .into_iter()
                .map(|field| field.parse::<f64>().unwrap())
                .collect()
        })
        .collect()
}

fn get_mnist_data() -> Vec<MnistPic> {
    use std::fs::File;
    use std::io::BufReader;

    let file = File::open("mnist_test.csv").unwrap();
    let mut rdr = csv::Reader::from_reader(BufReader::new(file));

    let data = rdr
        .records()
        .map(|result| result.unwrap())
        .collect::<Vec<_>>();

    let data = to_data(data);
    let mnist_pics = data
        .iter()
        .map(|d| MnistPic::new(d[0] as u8, &d[1..].to_vec()))
        .collect::<Vec<_>>();
    mnist_pics
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

    // let data = to_data(data);
    // let mnist_pics = data
    //     .iter()
    //     .map(|d| MnistPic::new(d[0] as u8, &d[1..].to_vec()))
    //     .collect::<Vec<_>>();

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
    gradient_descent(&x_train, &y_train.to_owned(), 0.10, 5000);
}
