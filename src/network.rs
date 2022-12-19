use crate::activation::*;
use ndarray as nd;

#[derive(Debug, Clone, PartialEq)]
struct Layer {
    weights: nd::Array2<f64>,
    biases: nd::Array2<f64>,
    activation: Activation,
}

impl Layer {
    fn forward_prop(&self, x: &nd::Array2<f64>) -> Neurons {
        let neurons_u = &self.weights.dot(x) + &self.biases;
        let neurons_a = self.activation.forward(&neurons_u);

        Neurons {
            unactivated: neurons_u,
            activated: neurons_a,
        }
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
    unactivated: nd::Array2<f64>,
    activated: nd::Array2<f64>,
}
pub struct Network {
    layers_num: usize,
    layers: Vec<Layer>,
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

    fn forward_prop(&self, layers: &Vec<Layer>, x: &nd::Array2<f64>) -> Vec<Neurons> {
        let neurons_a_0 = x; // That is the input

        let mut neurons_vec = Vec::<Neurons>::new();
        let mut last_activated_neurons = &neurons_a_0.clone();
        for layer in layers {
            let new_neurons = layer.forward_prop(last_activated_neurons);
            neurons_vec.push(new_neurons);
            last_activated_neurons = &neurons_vec.last().unwrap().activated;
        }
        neurons_vec
    }

    pub fn gradient_descent(
        &mut self,
        x: &nd::Array2<f64>,
        y: &nd::Array1<u64>,
        alpha: f64,
        iterations: usize,
    ) {
        let example_num = x.len_of(nd::Axis(1));
        let data_inputs = x.len_of(nd::Axis(0));
        let multiplier = data_inputs as f32 / example_num as f32;
        println!("Example num: {}", example_num);
        println!("Data inputs: {}", data_inputs);
        println!("Multiplier: {}", multiplier);
        let mut max_accuracy = 0.0;
        let curr_milestone = 0.0;
        let mut max_accuracy_iteration = 0;
        for i in 0..iterations {
            let neurons_vec = self.forward_prop(&self.layers, x);

            let neurons_vec_and_layers: Vec<(&Neurons, &Layer)> = self
                .layers
                .iter()
                .zip(neurons_vec.iter())
                .map(|(layer, neurons)| (neurons, layer))
                .collect();

            let cost_layers = self.backward_prop(&neurons_vec_and_layers, x, y);

            let layers_and_cost_layers: Vec<(&Layer, &Layer)> = self
                .layers
                .iter()
                .zip(cost_layers.iter())
                .map(|(layer, cost_layer)| (layer, cost_layer))
                .collect();
            self.layers = self.update_params(&layers_and_cost_layers, alpha);

            let prediction = get_predictions(&neurons_vec.last().unwrap().activated);
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
        let perceptrons = self.forward_prop(&self.layers, x);
        get_predictions(&perceptrons.last().unwrap().activated)
    }

    fn backward_prop(
        &self,
        neurons_and_layers: &Vec<(&Neurons, &Layer)>,
        x: &nd::Array2<f64>,
        y: &nd::Array1<u64>,
    ) -> Vec<Layer> {
        use std::collections::VecDeque;
        let m = 784.0;
        let neurons = neurons_and_layers
            .iter()
            .map(|&(neurons, _)| neurons.clone())
            .collect::<Vec<Neurons>>();

        let weights = neurons_and_layers
            .iter()
            .map(|(_, layer)| layer.weights.clone())
            .collect::<Vec<nd::Array2<f64>>>();

        let mut last_cost = neurons.last().unwrap().activated.clone() - one_hot(y);

        let mut layers: VecDeque<Layer> = VecDeque::new();
        for (i, _) in neurons_and_layers.iter().enumerate().rev() {
            let is_last = i == neurons_and_layers.len() - 1;
            let is_first = i == 0;

            // let activation = Activation::LeakyRelu;

            let cost = if is_last {
                last_cost.clone()
            } else {
                weights[i + 1].t().dot(&last_cost).clone()
                    * neurons_and_layers[i]
                        .1
                        .activation
                        .backward(&neurons[i].unactivated)
                        .clone()
            };
            let d_weights = if is_first {
                cost.dot(&x.t()) / m as f64
            } else {
                cost.dot(&neurons[i - 1].activated.t()) / m as f64
            };

            let d_biases = cost
                .sum_axis(nd::Axis(1))
                .into_shape((cost.shape()[0], 1))
                .unwrap()
                / m;

            last_cost = cost;
            layers.push_front(Layer {
                weights: d_weights,
                biases: d_biases,
                activation: neurons_and_layers[i].1.activation.clone(),
            });
        }

        // convert a `DequeVec` into a `Vec`
        layers.into_iter().collect()
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
pub fn get_accuracy(predictions: &nd::Array1<u64>, y: &nd::Array1<u64>) -> f64 {
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
