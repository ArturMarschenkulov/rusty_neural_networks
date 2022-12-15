mod matrix;
mod t;

use matrix::*;

fn relu(z: &Matrix) -> Vec<f64> {
    let mut a: Vec<f64> = Vec::new();
    // for i in 0..z.rows {
    //     for j in 0..z.cols {
    //         a.push(z.data[i][j].max(0.0));
    //     }
    // }
    // let a: Vec<f64> = z.iter().map(|&z| z.max(0.0)).collect();
    // let cache = z.to_vec();
    a
}

fn softmax(z: &[f64]) -> Vec<f64> {
    let exp_z: Vec<f64> = z.iter().map(|&z| z.exp()).collect();
    let sum_exp_z: f64 = exp_z.iter().sum();
    exp_z.iter().map(|&z| z / sum_exp_z).collect()
}

fn transpose(x: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    let mut data_t: Vec<Vec<f64>> = Vec::new();
    for i in 0..x[0].len() {
        let mut row: Vec<f64> = Vec::new();
        for j in 0..x.len() {
            row.push(x[j][i]);
        }
        data_t.push(row);
    }
    data_t
}

fn transpose_(x: &Vec<MnistPic>) -> Vec<Vec<f64>> {
    let mut data_t: Vec<Vec<f64>> = Vec::new();
    for i in 0..x[0].pixels.len() {
        let mut row: Vec<f64> = Vec::new();
        for j in 0..x.len() {
            row.push(x[j].pixels[i]);
        }
        data_t.push(row);
    }
    data_t
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
fn forward_prop(
    w_1: &Matrix,
    b_1: &Matrix,
    w_2: &Matrix,
    b_2: &Matrix,
    x: &Matrix,
) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    // don't use outside libraries
    // Z1 = W1.dot(X) + b1
    let z_1: Matrix = w_1.dot_multiply(x).add(b_1);
    assert_eq!(z_1.rows, w_1.rows);
    assert_eq!(z_1.cols, x.cols);
    let a_1 = relu(&z_1);
    unimplemented!()
}

fn gradient_descent(x: &Matrix, y: &Vec<f64>, alpha: f64, iterations: usize) {
    let (w_1, b_1, w_2, b_2) = init_params();

    // for i in 0..iterations {
    //     let (z_1, a_1, z_2, a_2) = forward_prop(x, &w_1, &b_1, &w_2, &b_2);
    //     let (dz_2, dw_2, db_2, dz_1, dw_1, db_1) = back_prop(x, y, &z_1, &a_1, &z_2, &a_2);
    //     let (w_1, b_1, w_2, b_2) =
    //         update_params(&w_1, &b_1, &w_2, &b_2, &dw_1, &db_1, &dw_2, &db_2, alpha);
    // }
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

fn init_params() -> (Matrix, Matrix, Matrix, Matrix) {
    use rand::Rng;

    let mut rng = rand::thread_rng();
    let range = -0.5..=0.5;

    let w_1: Matrix = Matrix::new(
        10,
        784,
        (0..=10)
            .map(|_| (0..=784).map(|_| rng.gen_range(range.clone())).collect())
            .collect::<Vec<Vec<f64>>>(),
    );
    let b_1: Matrix = Matrix::new(
        10,
        1,
        (0..=10)
            .map(|_| (0..=1).map(|_| rng.gen_range(range.clone())).collect())
            .collect::<Vec<Vec<f64>>>(),
    );
    let w_2 = Matrix::new(
        10,
        10,
        (0..=10)
            .map(|_| (0..=10).map(|_| rng.gen_range(range.clone())).collect())
            .collect::<Vec<Vec<f64>>>(),
    );
    let b_2 = Matrix::new(
        10,
        1,
        (0..=10)
            .map(|_| (0..=1).map(|_| rng.gen_range(range.clone())).collect())
            .collect::<Vec<Vec<f64>>>(),
    );
    (w_1, b_1, w_2, b_2)
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

    let data = rdr
        .records()
        .map(|result| result.unwrap())
        .collect::<Vec<_>>();

    let data = to_data(data);
    let mnist_pics = data
        .iter()
        .map(|d| MnistPic::new(d[0] as u8, &d[1..].to_vec()))
        .collect::<Vec<_>>();

    let mut res = ndarray::Array2::<f64>::zeros((mnist_pics.len(), 784));
    for (i, pic) in mnist_pics.iter().enumerate() {
        for (j, pixel) in pic.pixels.iter().enumerate() {
            res[[i, j]] = *pixel;
        }
    }
    res
}


fn main() {
    // t::main();
    // return;
    use ndarray as nd;

    let mnist_pics = get_mnist_data_2();
    let dev_size: i32 = 1000;

    let ss: nd::Array2<f32> = nd::array!([1., 2., 4.0], [6., 9., 2.3], [4., 6., 9.9]);
    println!("{:?}", ss);
    let ss = ss.slice(nd::s![..=0, ..]);
    println!("{:?}", ss);

    // slice `mnist_pics` from 0 to 1000
    let mnist_pics_test = mnist_pics.slice(nd::s![0..dev_size, ..]);
    let data_test = mnist_pics_test.t();
    let y_test = data_test.slice(nd::s![..=0, ..]);
    let x_test = data_test.slice(nd::s![1..1000, ..]);
    let x_test = x_test.map(|x| *x / 255.0);

    let mnist_pics_train = mnist_pics.slice(nd::s![dev_size.., ..]);
    let data_train = mnist_pics_train.t();
    let y_train = data_test.slice(nd::s![..=0, ..]);
    let x_train = data_test.slice(nd::s![1..1000, ..]);
    let x_train = x_train.map(|x| *x / 255.0);
}
