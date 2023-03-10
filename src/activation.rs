use ndarray as nd;

#[derive(Debug, Clone, PartialEq)]
pub enum Activation {
    Relu,
    LeakyRelu,
    Identity,
    Sigmoid,
    Tanh,
    Softmax,
}
impl Activation {
    pub fn forward(&self, x: &nd::Array2<f64>) -> nd::Array2<f64> {
        match self {
            Activation::Relu => x.map(|x| if x > &0.0 { *x } else { 0.0 }),
            Activation::LeakyRelu => x.map(|x| if x > &0.0 { *x } else { 0.01 * *x }),
            Activation::Identity => x.map(|x| *x),
            Activation::Sigmoid => x.map(|x| 1.0 / (1.0 + std::f64::consts::E.powf(-*x))),
            Activation::Tanh => x.map(|x| x.tanh()),
            Activation::Softmax => softmax(x),
        }
    }
    pub fn backward(&self, x: &nd::Array2<f64>) -> nd::Array2<f64> {
        match self {
            Activation::Relu => x.map(|x| if x > &0.0 { 1.0 } else { 0.0 }),
            Activation::LeakyRelu => x.map(|x| if x > &0.0 { 1.0 } else { 0.01 }),
            Activation::Identity => x.map(|_| 1.0),
            Activation::Sigmoid => self.forward(x),
            Activation::Tanh => x.map(|x| 1.0 - (x.powi(2))),
            Activation::Softmax => softmax_derivative(x),
        }
    }
}

// pub fn softmax_2(input: &nd::Array2<f64>) -> nd::Array2<f64> {
//     let mut output = input.clone();
//     for col in output.axis_iter_mut(nd::Axis(1)) {
//         let max = col.fold(f64::NEG_INFINITY, |max, &x| max.max(x));
//         let sum = col.mapv(|x| (x - max).exp()).sum();
//         nd::Zip::from(col).and(&sum).apply(|x, &sum| *x / sum);
//     }
//     output
// }

pub fn softmax(z: &nd::Array2<f64>) -> nd::Array2<f64> {
    let e = z.map(|x| x.exp());
    let sum = e.sum_axis(nd::Axis(0));
    e / sum
}

pub fn softmax_derivative(z: &nd::Array2<f64>) -> nd::Array2<f64> {
    let e = z.map(|x| x.exp());
    let sum = e.sum_axis(nd::Axis(0));
    let s = e / sum;
    nd::Array::eye(z.len()) * s.clone() - s.t().dot(&s)
}
