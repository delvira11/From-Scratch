import numpy as np

def PCA_fit_transform(self, X, n_components):
    # Estandarizar datos
    X = X.copy()
    mean = np.mean(X, axis = 0)
    scale = np.std(X, axis = 0)
    X_std = (X - mean) / scale

    # Matriz de correlacion (covarianzas estandarizadas)    
    cov_mat = np.cov(X_std.T)

    # Valores y vectores propios
    eig_vals, eig_vecs = np.linalg.eig(cov_mat) 
    eig_vecs = eig_vecs.T

   # Ordenar vectores propios en orden descendente (matriz transformación)
    paired_eig = [(eig_vals[i], eig_vecs[i, :]) for i in range(len(eig_vals))]
    paired_eig.sort(key=lambda x: x[0], reverse=True)
    sorted_vals = np.array([i[0] for i in paired_eig])
    sorted_vecs = np.array([i[1] for i in paired_eig])
    
    # Matriz de transformación
    w = sorted_vecs[:n_components, :]

    # Cálculo de componentes
    result = (w @ X_std.T).T 
    
    # Componentes
    return result