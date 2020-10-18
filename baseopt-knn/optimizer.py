import bayesmark.random_search as rs
from bayesmark import np_util
from bayesmark.abstract_optimizer import AbstractOptimizer
from bayesmark.experiment import experiment_main


from sklearn.neighbors import KNeighborsRegressor
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import copy


class BaseOptimizer(AbstractOptimizer):
    # Unclear what is best package to list for primary_import here.
    primary_import = "bayesmark"

    def __init__(self, api_config, random=np_util.random):
        """Build wrapper class to use random search function in benchmark.

        Settings for `suggest_dict` can be passed using kwargs.

        Parameters
        ----------
        api_config : dict-like of dict-like
            Configuration of the optimization variables. See API description.
        """
        AbstractOptimizer.__init__(self, api_config)
        self.name='BaseOptKNN'
        self.random = random
        self.scored_suggestions = None  # keep scored suggestions returned by the external environment
        self.column_list = None  # make sure order of columns in train and test sets is the same
        self.n_samples = 500
        self.new_suggestions = None  # keep suggestions for next iteration
        self.new_suggestions_with_score = None  # keep suggestions for next iteration, for debug purpose
        self.n_suggestions = None

        self.current_iter = 0  # current iteration
        self.knn_iter_count = 4  # number of iterations before we switch from random to kNN regressor

    def suggest(self, n_suggestions=1):
        """Get suggestion.

        Parameters
        ----------
        n_suggestions : int
            Desired number of parallel suggestions in the output

        Returns
        -------
        next_guess : list of dict
            List of `n_suggestions` suggestions to evaluate the objective
            function. Each suggestion is a dictionary where each key
            corresponds to a parameter being optimized.
        """
        self.current_iter += 1
        print(f"In {self.name}'s suggest(), iter:{self.current_iter}")

        if self.n_suggestions is None:
            self.n_suggestions = n_suggestions

        if self.current_iter <= self.knn_iter_count:
            x_guess = rs.suggest_dict([], [], self.api_config, n_suggestions=self.n_suggestions, random=self.random)
            return x_guess

        return self.new_suggestions

    def observe(self, X, y):
        """Feed an observation back.

        Parameters
        ----------
        X : list of dict-like
            Places where the objective function has already been evaluated.
            Each suggestion is a dictionary where each key corresponds to a
            parameter being optimized.
        y : array-like, shape (n,)
            Corresponding values where objective has been evaluated
        """
        print(f"In {self.name}'s observe(), iter:{self.current_iter}")

        for idx, sug in enumerate(X):
            suggestion = copy.deepcopy(sug)
            suggestion['score'] = y[idx]

            if self.scored_suggestions is None:
                self.scored_suggestions = [suggestion]
            else:
                self.scored_suggestions.extend([suggestion])

        if self.current_iter >= self.knn_iter_count:
            self._update_state()

    def _update_state(self):
        # convert scored suggestions to data frame for later steps
        df = pd.DataFrame(self.scored_suggestions)

        # prepare training data
        # get target y_train
        y_train = df['score'].values
        df.drop(['score'], inplace=True, axis=1)

        # get and keep the column list of train set
        self.column_list = df.columns

        # get predictors for train set
        X_train = df.values

        # train the pipeline
        if df.shape[0] < 5:
            pipeline = Pipeline([('scaler', StandardScaler()), ('est', KNeighborsRegressor(n_jobs=-1, n_neighbors=df.shape[0]))])
        else:
            pipeline = Pipeline([('scaler', StandardScaler()), ('est', KNeighborsRegressor(n_jobs=-1, n_neighbors=5))])

        pipeline = pipeline.fit(X_train, y_train)

        # create a random test set
        df_test = self._create_random_testset()

        # get predicted scores for the test set
        y_pred = pipeline.predict(df_test.values)

        # sort scores in ascending order
        df_test['score'] = y_pred
        df_test.sort_values(by=['score'], inplace=True, ascending=True)
        df_test = df_test.drop_duplicates('score')
        # remove 'score' column
        df_test = df_test.drop(['score'], axis=1)

        if self.n_suggestions > df_test.shape[0]:
            # we have fewer than self.n_suggestions points
            # create additional random points to make up
            x_guess = rs.suggest_dict([], [], self.api_config, n_suggestions=self.n_suggestions - df_test.shape[0], random=self.random)
            df_x_guess = pd.DataFrame(x_guess)
            df_new_suggestion = pd.concat([df_test, df_x_guess], axis=0)
        else:
            # we have more than self.n_suggestions points
            if self.n_suggestions > 4:
                # take the top 4 rows
                df_top = df_test.head(4)
                # remove the top 4 rows
                df_rest = df_test.iloc[4:]
                # sample without replacement 4 rows
                df_bot = df_rest.sample(self.n_suggestions - 4)
                # merge to create new data frame
                df_new_suggestion = pd.concat([df_top, df_bot], axis=0)
            else:
                df_new_suggestion = df_test.head(self.n_suggestions)

        # create new suggestions
        self.new_suggestions = df_new_suggestion.to_dict('records')

    def _create_random_testset(self):
        x_guess = rs.suggest_dict([], [], self.api_config, n_suggestions=self.n_samples, random=self.random)
        df = pd.DataFrame(x_guess)
        # make sure that the order of columns in test set is the same as train set
        ret_df = df[self.column_list]
        return ret_df


if __name__ == "__main__":
    experiment_main(BaseOptimizer)
