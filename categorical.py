from sklearn import preprocessing,linear_model
import pandas as pd
class CategoricalFeatures:
    def __init__(self, df, categorical_features, encoding_type, handle_na=False):
        """
        df: pandas dataframe
        categorical features: list of column names
        encoding type: label, binary, ohe
        """
        self.df = df
        self.handle_na = handle_na
        self.cat_feats = categorical_features
        self.enc_type = encoding_type
        self.label_encoders = dict()
        self.label_binarizer = dict()
        self.ohe = None
        if self.handle_na == True:
            for c in self.cat_feats:
                self.df.loc[:, c] = self.df.loc[:,c].astype(str).fillna("-9999999")
        self.output_df = self.df.copy(deep=True)
    
    def _label_encoding(self):
        for c in self.cat_feats:
            lbl = preprocessing.LabelEncoder()
            lbl.fit(self.df[c].values)
            self.output_df.loc[:, c] = lbl.transform(self.df[c].values)
            self.label_encoders[c] = lbl
        return self.output_df
    
    def _label_binarizer(self):
        for c in self.cat_feats:
            lbl = preprocessing.LabelBinarizer()
            lbl.fit(self.df[c].values)
            val = lbl.transform(df[c].values)
            self.output_df = self.output_df.drop(c, axis=1)
            for i in range(val.shape[1]):
                new_col_name = c+f"__bin__{i}"
                self.output_df[new_col_name] = val[:,i]
            self.label_binarizer[c] = lbl
        return self.output_df
    def one_hot(self):
        ohe = preprocessing.OneHotEncoder()
        ohe.fit(self.df[self.cat_feats].values)
        self.ohe = ohe
        return ohe.transform(self.df[self.cat_feats].values)
    
    def fit_transform(self):
        if self.enc_type == 'label':
            return self._label_encoding()
        elif self.enc_type == 'binary':
            return self._label_binarizer()
        elif self.enc_type == 'ohe':
            return self.one_hot()
        else:
            return Exception("Encoding type not understood")
        
    def transform(self, dataframe):
        if self.handle_na == True:
            for c in self.cat_feats:
                dataframe.loc[:,c] = dataframe.loc[:,c].astype(str).fillna("-999999")
        if self.enc_type == "label":
            for c,lbl in self.label_encoders:
                dataframe.loc[:,c] = lbl.transform(dataframe.loc[:,c].values)
                return dataframe
        elif self.enc_type == "binary":
            for c,lbl in self.label_binarizer:
                val = lbl.transform(dataframe.loc[:,c].values)
                dataframe = dataframe.drop([c],axis=1)
                for i in range(val.shape[1]):
                    new_col_name = c+f"__bin__{i}"
                    dataframe[new_col_name] = va[:,i] 
                return dataframe
        elif self.enc_type == 'ohe':
            return self.ohe.transform(dataframe[self.cat_feats].values)
        else:
            return Exception("Encoding type not understood")
    
if __name__ == "__main__": 
    df = pd.read_csv(r'C:\Deep Learning CourseEra Course\kaggle\cat_enc_challange\input\train.csv')
    df_test = pd.read_csv(r'C:\Deep Learning CourseEra Course\kaggle\cat_enc_challange\input\test.csv')
    sample = pd.read_csv(r'C:\Deep Learning CourseEra Course\kaggle\cat_enc_challange\input\sample_submission.csv')
    df_test['target'] = -1

    full_data = pd.concat([df,df_test])


    cols = [c for c in df.columns if c not in ['id','target']]
    print(cols)
    # cat_feats = CategoricalFeatures(df,cols,'label',True)
    cat_feats = CategoricalFeatures(full_data,
                                    categorical_features = cols,
                                    encoding_type='label',
                                    handle_na=True)
    output_df = cat_feats.fit_transform()
    
    one_hot = CategoricalFeatures(output_df,
                                  categorical_features=cols,
                                  encoding_type='ohe',
                                  handle_na=True)
    one_df = one_hot.fit_transform()
    print(one_df.shape)

    x = one_df[:len(df),:]
    x_test = one_df[len(df):,:]

    clf = linear_model.LogisticRegression()
    clf.fit(x,df.target.values)
    preds = clf.predict(x_test)[:,1]

    sample.loc[:,"target"] = preds
    sample.to_csv("submission.csv", index=False)
