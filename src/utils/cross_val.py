class EraCV:
    """Select validation eras and train on previous eras

    provides train/test indices to split data in train/test splits. In
    each split, one or more eras are used as a validation set while the
    specified number of immediately preceding eras are used as a
    training set.
    """

    def __init__(self, eras):
        self.eras = eras
        self.unique_eras = self._era_to_int(eras.unique())
        self.eras_int = self._era_to_int(eras)
    
    def _era_to_int(self, eras):
        return [int(era[3:]) for era in eras]

    def get_valid_indices(self, valid_start, valid_n_eras):
        self.valid_eras = self.unique_eras[self.unique_eras.index(valid_start):\
                                      self.unique_eras.index(valid_start)+\
                                      valid_n_eras]
        valid_bool = [era in self.valid_eras for era in self.eras_int] 
        self.valid_indices = np.where(valid_bool)

    def get_train_indices(self, valid_start:int, train_n_eras:int):
        train_n_eras = 0 if (train_n_eras is None) else train_n_eras
        self.train_eras = [era for era in self.unique_eras if era <\
                           valid_start][-train_n_eras:]
        train_bool = [era in self.train_eras for era in self.eras_int]
        self.train_indices = np.where(train_bool)

    def get_splits(self, valid_start:int, valid_n_eras:int,
                   train_n_eras:int = None):
        self.get_valid_indices(valid_start, valid_n_eras)
        self.get_train_indices(valid_start, train_n_eras)
        return self.train_indices[0], self.valid_indices[0]
