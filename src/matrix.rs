#[derive(Debug, Clone)]
pub struct Matrix {
    pub rows: usize,
    pub cols: usize,
    data: Vec<Vec<f64>>,
    // data: Vec<64>,
}
impl Matrix {
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Matrix {
            data: vec![vec![0.0; cols]; rows],
            rows,
            cols,
        }
    }
    pub fn new(rows: usize, cols: usize, data: Vec<Vec<f64>>) -> Self {
        assert_eq!(data.len(), rows, "Data must be the same size as rows");
        assert_eq!(data[0].len(), cols, "Data must be the same size as cols");
        Matrix { rows, cols, data }
    }
    pub fn from_vec(rows: usize, cols: usize, data: Vec<f64>) -> Self {
        let data: Vec<Vec<f64>> = data.iter().map(|x| vec![*x]).collect();
        Matrix { rows, cols, data }
    }
    // pub fn from_vec(data: &Vec<f64>) -> Self {
    //     Matrix {
    //         rows: data.len(),
    //         cols: 1,
    //         data: vec![data.clone()],
    //     }
    // }
}
impl Matrix {
    pub fn multiply(&self, other: &Matrix) -> Self {
        assert_eq!(self.cols, other.rows, "Matrices must be compatible");
        let mut res = Matrix::zeros(self.rows, other.cols);
        for i in 0..self.rows {
            for j in 0..other.cols {
                for k in 0..self.cols {
                    res.data[i][j] += self.data[i][k] * other.data[k][j];
                }
            }
        }
        res
    }
    pub fn add(&self, other: &Matrix) -> Self {
        assert_eq!(self.rows, other.rows, "Matrices must be compatible");
        assert_eq!(self.cols, other.cols, "Matrices must be compatible");
        let mut res = Matrix::zeros(self.rows, self.cols);
        for i in 0..self.rows {
            for j in 0..self.cols {
                res.data[i][j] = self.data[i][j] + other.data[i][j];
            }
        }
        res
    }
    pub fn dot_multiply(&self, other: &Matrix) -> Self {
        assert_eq!(self.rows, other.rows, "Matrices must be compatible");
        assert_eq!(self.cols, other.cols, "Matrices must be compatible");
        let mut res = Matrix::zeros(self.rows, self.cols);
        for i in 0..self.rows {
            for j in 0..self.cols {
                res.data[i][j] = self.data[i][j] * other.data[i][j];
            }
        }
        res
    }
    pub fn dot(&self, other: &Matrix) -> f64 {
        assert_eq!(self.rows, other.rows, "Matrices must be compatible");
        assert_eq!(self.cols, other.cols, "Matrices must be compatible");
        let mut res = 0.0;
        for i in 0..self.rows {
            for j in 0..self.cols {
                res += self.data[i][j] * other.data[i][j];
            }
        }
        res
    }
    pub fn subtract(&self, other: &Matrix) -> Self {
        assert_eq!(self.rows, other.rows, "Matrices must be compatible");
        assert_eq!(self.cols, other.cols, "Matrices must be compatible");
        let mut res = Matrix::zeros(self.rows, self.cols);
        for i in 0..self.rows {
            for j in 0..self.cols {
                res.data[i][j] = self.data[i][j] - other.data[i][j];
            }
        }
        res
    }
    pub fn map(&self, func: fn(f64) -> f64) -> Self {
        let mut res = Matrix::zeros(self.rows, self.cols);
        for i in 0..self.rows {
            for j in 0..self.cols {
                res.data[i][j] = func(self.data[i][j]);
            }
        }
        res
    }
    pub fn transpose(&self) -> Self {
        let mut res = Matrix::zeros(self.cols, self.rows);
        for i in 0..self.rows {
            for j in 0..self.cols {
                res.data[j][i] 
                = self.data[i][j];
            }
        }
        res
    }
}
