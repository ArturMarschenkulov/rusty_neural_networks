use crate::activation::*;
use ndarray as nd;

#[derive(Debug, Clone, PartialEq)]
struct Layer {
    weights: nd::Array2<f64>,
    biases: nd::Array2<f64>,
    activation: Activation,
}

impl Layer {
    fn foreprop(&self, last_activated_neurons: &nd::Array2<f64>) -> Neurons {
        let z = &self.weights.dot(last_activated_neurons) + &self.biases;
        let a = self.activation.forward(&z);
        Neurons { z, a }
    }
    fn backprop(&self, previous_layer: &Layer) -> Layer {
        unimplemented!()
    }
    // fn backward_prop(
    //     &self,
    //     x: &nd::Array2<f64>,
    //     y: &nd::Array2<f64>,
    //     neurons: &Neurons,
    // ) -> (nd::Array2<f64>, nd::Array2<f64>) {
    //     let delta = (y - &neurons.activated) * self.activation.backward(&neurons.unactivated);
    //     let weights_gradient = delta.dot(&x.t());
    //     let biases_gradient = delta.sum_axis(nd::Axis(1)).insert_axis(nd::Axis(1));

    //     (weights_gradient, biases_gradient)
    // }

    // fn backward_prop(&self, last_cost: nd::Array2<f64>) -> Layer {}
}

#[derive(Clone)]
struct Neurons {
    z: nd::Array2<f64>, // unactivated neurons
    a: nd::Array2<f64>, // activated neurons
}
pub struct Network {
    layers_num: usize,
    layers: Vec<Layer>,
}
fn zip<'a, 'b, A, B>(a: &'a [A], b: &'b [B]) -> Vec<(&'a A, &'b B)> {
    a.iter().zip(b.iter()).collect()
}
impl Network {
    pub fn new() -> Self {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let range = -0.5..=0.5;

        // For right now, this array has to begin with 784 and end with 10. They represent the inpuit and output layer.
        // Later this will be refactored to be more flexible.
        let layers_num = [784, 10, 10];

        let mut layers = Vec::new();

        let mut i = 0;
        while i < layers_num.len() - 1 {
            let weights = nd::Array2::<f64>::zeros((layers_num[i + 1], layers_num[i]))
                .map(|_| rng.gen_range(range.clone()));
            let biases = nd::Array2::<f64>::zeros((layers_num[i + 1], 1))
                .map(|_| rng.gen_range(range.clone()));
            layers.push(Layer {
                weights,
                biases,
                activation: if i == layers_num.len() - 2 {
                    Activation::Softmax
                } else {
                    Activation::LeakyRelu
                },
            });

            i += 1;
        }

        Network {
            layers_num: layers_num.len(),
            layers,
        }
    }

    fn foreprop(&self, layers: &Vec<Layer>, input_data: &nd::Array2<f64>) -> Vec<Neurons> {
        let neurons_a_0 = input_data; // That is the input

        let mut neurons_vec = Vec::<Neurons>::new();
        let mut last_activated_neurons = &neurons_a_0.clone();
        for layer in layers {
            let new_neurons = layer.foreprop(last_activated_neurons);
            neurons_vec.push(new_neurons);
            last_activated_neurons = &neurons_vec.last().unwrap().a;
        }
        neurons_vec
    }

    pub fn gradient_descent(
        &mut self,
        input_data: &nd::Array2<f64>,
        labels: &nd::Array2<u64>,
        alpha: f64,
        iterations: usize,
    ) {
        let labels = &labels.map(|x| *x as f64);
        let example_num = input_data.len_of(nd::Axis(1));
        let data_inputs = input_data.len_of(nd::Axis(0));
        let multiplier = data_inputs as f32 / example_num as f32;
        println!("Example num: {}", example_num);
        println!("Data inputs: {}", data_inputs);
        println!("Multiplier: {}", multiplier);
        let mut max_accuracy = 0.0;
        let mut accuracy_change = 0.0;
        let curr_milestone = 0.0;
        let mut max_accuracy_iteration = 0;
        for i in 0..iterations {
            let neurons_vec = self.foreprop(&self.layers, input_data);

            let cost_layers = self.backprop(&neurons_vec, &self.layers, input_data, labels);

            self.layers = self.update_params(&zip(&self.layers, &cost_layers), alpha);

            let prediction = get_predictions(&neurons_vec.last().unwrap().a);
            let accuracy = get_accuracy(&prediction, labels);
            if accuracy > max_accuracy {
                accuracy_change = accuracy - max_accuracy;
                max_accuracy = accuracy;
                max_accuracy_iteration = i;
            }

            if accuracy > curr_milestone {
                println!("Iteration: {}", i);
                println!("Accuracy: {}", accuracy);
                println!(
                    "Max Accuracy: {} since {} (d {} by {})",
                    max_accuracy,
                    max_accuracy_iteration,
                    i - max_accuracy_iteration,
                    accuracy_change,
                );
                println!("---------------------");
                // curr_milestone = milestones[curr_milestone as usize];
            }

            // if accuracy > 0.90 {
            //     break;
            // }

            // if i % 100 == 0 {
            //     println!("Iteration: {}", i);
            //     let prediction = get_predictions(&perceptrons.last().unwrap().activated);
            //     println!("Accuracy: {}", get_accuracy(&prediction, y));
            // }
        }
    }

    pub fn make_prediction(&self, x: &nd::Array2<f64>) -> nd::Array1<u64> {
        let perceptrons = self.foreprop(&self.layers, x);
        get_predictions(&perceptrons.last().unwrap().a)
    }

    fn backprop(
        &self,
        neurons: &Vec<Neurons>,
        layers: &Vec<Layer>,
        input_data: &nd::Array2<f64>,
        labels: &nd::Array2<f64>,
    ) -> Vec<Layer> {
        use std::collections::VecDeque;
        let num_samples = input_data.shape()[0];
        let num_layers = layers.len();
        let num_neurons = neurons.len();

        assert_eq!(num_neurons, num_layers);

        let mut last_cost: Option<nd::Array2<f64>> = None;
        let mut result_layers: VecDeque<Layer> = VecDeque::with_capacity(num_layers);
        for i in (0..num_layers).rev() {
            let is_last_layer = i == layers.len() - 1;
            let is_first_layer = i == 0;

            let previous_layer = if is_first_layer {
                // If this is the first layer, use the input data as the
                // "previous" layer
                input_data.t()
            } else {
                // Otherwise, use the activations of the previous layer
                neurons[i - 1].a.t()
            };

            let cost = if is_last_layer {
                // If this is the last layer, the cost is just the difference
                // between the activations and the labels
                neurons.last().unwrap().a.clone() - labels
            } else {
                // Otherwise, the cost is the dot product between the weights of
                // the next layer and the last cost, multiplied by the derivative
                // of the activation function
                calculate_cost(&layers[i], &neurons[i], &layers[i + 1], &last_cost.unwrap())
            };

            // Calculate the gradient for the weights and biases of this layer
            let d_weights = calculate_d_weights(&cost, &previous_layer.to_owned(), num_samples);
            let d_biases = calculate_d_biases(&cost, num_samples);

            // Update the last cost for the next iteration
            last_cost = Some(cost);

            // Push the gradients for this layer to the result vectorF
            result_layers.push_front(Layer {
                weights: d_weights,
                biases: d_biases,
                activation: layers[i].activation.clone(),
            });
        }

        // convert a `DequeVec` into a `Vec`
        result_layers.into_iter().collect()
    }

    fn update_params(&self, lds: &[(&Layer, &Layer)], alpha: f64) -> Vec<Layer> {
        let mut layers = Vec::new();
        for (_, ld) in lds.iter().enumerate() {
            // assert_eq!(ld.0.weight.shape(), ld.1.weight.shape());
            // assert_eq!(ld.0.bias.shape(), ld.1.bias.shape());

            let weights = ld.0.weights.clone() - ld.1.weights.clone() * alpha;
            let biases = ld.0.biases.clone() - ld.1.biases.clone() * alpha;
            layers.push(Layer {
                weights,
                biases,
                activation: ld.0.activation.clone(),
            });
        }
        layers
    }
}

fn calculate_cost(
    curr_layer: &Layer,
    neurons: &Neurons,
    next_layer: &Layer,
    last_cost: &nd::Array2<f64>,
) -> nd::Array2<f64> {
    let next_layer_weights = &next_layer.weights;
    let activation_derivative = &curr_layer.activation.backward(&neurons.z);

    next_layer_weights.t().dot(last_cost) * activation_derivative.clone()
}
fn calculate_d_biases(cost: &nd::Array2<f64>, num_samples: usize) -> nd::Array2<f64> {
    cost.sum_axis(nd::Axis(1))
        .into_shape((cost.shape()[0], 1))
        .unwrap()
        / num_samples as f64
}
fn calculate_d_weights(
    cost: &nd::Array2<f64>,
    previous_layer: &nd::Array2<f64>,
    num_samples: usize,
) -> nd::Array2<f64> {
    cost.dot(previous_layer) / num_samples as f64
}
pub fn one_hot(y: &nd::Array1<u64>) -> nd::Array2<f64> {
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
pub fn get_accuracy(predictions: &nd::Array1<u64>, labels: &nd::Array2<f64>) -> f64 {
    let labels = get_predictions(labels);
    // give an array of true and false
    let bools = predictions
        .iter()
        .zip(labels.iter())
        .map(|(x, y)| x == &(*y as u64))
        .collect::<Vec<_>>();

    // count the number of true
    let true_count = bools.iter().filter(|x| **x).count();
    true_count as f64 / labels.len() as f64
}
