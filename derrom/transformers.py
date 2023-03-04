import numpy as np


class base_transformer:
    """
    Base class to define the methods to be implemented.
    """
    def __init__(self):
        pass
    
    def transform(self):
        raise NotImplementedError
        
    def setup(self):
        raise NotImplementedError
        
        
class polynomial_features(base_transformer):
    """
    Polynomial features transformation.
    
    Generates a feature vector with all monomials up to the specified order/degree. This includes mixed monomials for orders > 1. 
    
    
    Notes:
    ------
        * The size of the resulting feature vector scales with the power of order/degree.
        * The generated features are not orthogonal.
        * Polynomial (opposed to monomial features) are constructed by the linear combination in the regression step.
        * The computation is implemented by iteratively computing the outer product (Tensor) product with the linear feature vector and then reshaping the resultung upper triangle (to avoid redundant features) back to a vector. Hence, more memory is allocated internally than one might infer from the resultung feature vector size.
        * The monomial features correspond the basis functions of a discrete Volterra series.
        
   
    
    
    Parameters:
    -----------
    order : int
        Polynomial degree, to which features are to be generated.
    """
    
    
    def __init__(self, order=2):
        self.order = order
    
    def __build_VAR_p_Vec(self, VAR_vec, order):
        VAR_p_Vec = [VAR_vec]
        VARp = VAR_vec
        for p in range(1,order):
            VARp = np.outer(VAR_vec,VARp)
            VARp = VARp[ np.triu_indices(VARp.shape[0], m=VARp.shape[1]) ]
            VAR_p_Vec.append(VARp)
        return np.concatenate(VAR_p_Vec, axis=0)
    
    def transform(self,DE_state_matrix):
        """
        Carries out the polynomial features transformation.
        
        
        Parameters:
            DE_state_matrix : 2D numpy.ndarray
                Delay embedded state matrix with the features in the rows, which is to be transformed. May only contain one row.
        """
        if DE_state_matrix.shape[0] == 1:
            return self.__build_VAR_p_Vec(DE_state_matrix[0], self.order).reshape(1,-1)
          
        else:        
            nCols = self.__build_VAR_p_Vec(DE_state_matrix[0], self.order).size
            nRows = DE_state_matrix.shape[0]
          
            poly_features = np.zeros((nRows,nCols)) 
          
            for k in range( DE_state_matrix.shape[0] ):
                poly_features[k] = self.__build_VAR_p_Vec(DE_state_matrix[k], self.order)

            return poly_features
        
    def setup(self, n_DE_features):
        """
        Not required for this transformation.
        """
        pass
        
        
        
class ELM_features(base_transformer):
    """
    Extreme Learning Machine (ELM) features transformation.
    
    Generates a feature vector, which concatenates the linear delay-embedded feature vector and the ELM transformation thereoff.
    The ELM transform can be understood as sinle-layer feed-forward network. The inputs are subject to random linear projection, which are then combined with a random bias and fed into a (nonlinear) activation function.
    
    The weights of the random projection matrix are drawn from a normal distribution whose parameters can be specified. The elements of the bias vector are drawn from a uniform distribution whose parameters can be specified.
    
    Parameters:
    -----------
    ELM_nodes : int
        Number of ELM nodes/neurons.
    ELM_weights_mean : float
        Mean value for the randomly drawn projection weights.
    ELM_weights_std : float
        Standard deviation of the randomly drawn projection weights.
    ELM_bias_low : float
        Lower bound for the randomly drawn biases.
    ELM_bias_high : float
        Upper bound for the randomly drawn biases.
    seed : int
        Seed value for the random generator.
    activation_function : callable
        Activation function for the ELM neurons. Must act element-wise on 1D numpy.ndarrays.
    """
    
    def __init__(self, ELM_nodes = 200, ELM_weights_mean = 0.0, ELM_weights_std = 1.0, ELM_bias_low = -1.0, ELM_bias_high = 1.0,  seed=817, activation_function=np.tanh):
        self.rng = np.random.default_rng(seed=seed)
        self.ELM_nodes = ELM_nodes
        self.ELM_weights_mean = ELM_weights_mean
        self.ELM_weights_std = ELM_weights_std
        self.ELM_bias_low = ELM_bias_low
        self.ELM_bias_high = ELM_bias_high
        self.projection_matrix = None
        self.bias_vector = None
        self.activation_function = activation_function
        
    def setup(self, n_DE_features):
        """
        Generates the random projection matrix and the random bias vector and thereby defines the number of ELM nodes/neurons. 
        
        Note that the standard deviation of the randomly drawn projection matrix is scaled with the square root of number of ELM roots. This ensures that the mean input a neuron receives is of the order of one.
        
        Parameters:
        -----------
            n_DE_features : int
                Number of delay embedded features, i.e., size the delay embedded feature vector. This is set by derrom's fit method.
        """
        self.projection_matrix = self.rng.normal(self.ELM_weights_mean, self.ELM_weights_std/np.sqrt(n_DE_features), (n_DE_features, self.ELM_nodes))
        self.bias_vector = self.rng.uniform(self.ELM_bias_low, self.ELM_bias_high, self.ELM_nodes)
    
    def transform(self,DE_state_matrix):
        """
        Carries out the ELM features transformation.
        
        
        Parameters:
            DE_state_matrix : 2D numpy.ndarray
                Delay embedded state matrix with the features in the rows, which is to be transformed. May only contain one row.
        """
        projected_data = self.activation_function( DE_state_matrix @ self.projection_matrix + self.bias_vector )
        state = np.concatenate( [DE_state_matrix,projected_data], axis=1)

        return state
