import pandas as pd
from sklearn import preprocessing
from sklearn.manifold import MDS
from sklearn.manifold import TSNE


class TSMODEL:

    """Summary
    
    Attributes:
        proj_data_MDS (TYPE): Description
        proj_data_TSNE (TYPE): Description
        X (TYPE): Description
        X_scaled (TYPE): Description
    """
    
    def __init__(self, data_frame):
        """Init method
        
        Args:
            data_frame (TYPE): Description
        """
        self.X = data_frame[data_frame.columns[1:8]]
        self.X_scaled = preprocessing.scale(self.X)

    def fit_model_MDS(self, n_components):
        """Summary
        
        Args:
            n_components (TYPE): Description
        """
        embedding = MDS(n_components=n_components)
        X_MDS = embedding.fit_transform(self.X_scaled)

        self.proj_data_MDS = pd.DataFrame(X_MDS)

        dummy_dict = {'row_1': 1, 'row_2': 2}
        return dummy_dict

    def fit_model_TSNE(self, **kwargs):
        """Summary
        
        Args:
            **kwargs: Description
        """
        X_proj_TSNE = TSNE(**kwargs).fit_transform(self.X_scaled)
        self.proj_data_TSNE = pd.DataFrame(X_proj_TSNE)

        dummy_dict = {'row_1': 3, 'row_2': 4}
        return dummy_dict