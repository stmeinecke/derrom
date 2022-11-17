 
from sklearn.decomposition import NMF
class NNMF(base_dim_reducer):     
        
    def __init__(self, max_iter=200, alpha_W=1e-4):
        self.max_iter = max_iter
        self.alpha_W = alpha_W
        pass
    
    def train(self, data_matrix, rdim):
        data_matrix -= data_matrix.min()
        self.model = NMF(n_components=rdim, init='random', random_state=0, max_iter=self.max_iter, alpha_H = 'same', alpha_W=self.alpha_W, solver='cd')
        
        self.model.fit(data_matrix)
        
    def reduce(self, data_matrix, rdim):
        data_matrix -= data_matrix.min()
        return self.model.transform(data_matrix)
        
    def reconstruct(self, reduced_data_matrix):
        return self.model.inverse_transform(reduced_data_matrix)
