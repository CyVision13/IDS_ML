import numpy as np
import random
from sklearn.base import clone
from sklearn.metrics import accuracy_score


class DGWA:
    def __init__(self, classifier, X_train, y_train, X_val, y_val,
                 population_size=20, max_iter=50, feature_count=None,
                 verbose=False, use_exploration_ops=True):  # ✅ New flag
        self.classifier = classifier
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.pop_size = population_size
        self.max_iter = max_iter
        self.feature_count = feature_count or X_train.shape[1]
        self.verbose = verbose
        self.use_exploration_ops = use_exploration_ops  # ✅ Save flag

        # Initialize population: binary vectors
        self.population = np.random.randint(2, size=(self.pop_size, self.feature_count))
        self.fitness = np.zeros(self.pop_size)

        # Alpha, Beta, Delta
        self.alpha_pos = None
        self.beta_pos = None
        self.delta_pos = None
        self.alpha_score = -np.inf
        self.beta_score = -np.inf
        self.delta_score = -np.inf

    def fitness_function(self, wolf):
        """Evaluate fitness = classification accuracy using selected features"""
        selected_features = np.where(wolf == 1)[0]
        if len(selected_features) == 0:
            return 0  # no features selected → worst fitness

        clf = clone(self.classifier)
        clf.fit(self.X_train[:, selected_features], self.y_train)
        preds = clf.predict(self.X_val[:, selected_features])
        return accuracy_score(self.y_val, preds)

    def update_leaders(self):
        """Update alpha, beta, delta wolves and their fitness"""
        for i in range(self.pop_size):
            fit = self.fitness[i]
            if fit > self.alpha_score:
                self.delta_score, self.delta_pos = self.beta_score, self.beta_pos
                self.beta_score, self.beta_pos = self.alpha_score, self.alpha_pos
                self.alpha_score, self.alpha_pos = fit, self.population[i].copy()
            elif fit > self.beta_score:
                self.delta_score, self.delta_pos = self.beta_score, self.beta_pos
                self.beta_score, self.beta_pos = fit, self.population[i].copy()
            elif fit > self.delta_score:
                self.delta_score, self.delta_pos = fit, self.population[i].copy()

    def exploration_operators(self, wolf):
        """Apply exploration operators (Algorithm 4)"""

        new_wolf = wolf.copy()
        length = len(wolf)

        # 1. Gene swapping between two random positions
        idx1, idx2 = random.sample(range(length), 2)
        new_wolf[idx1], new_wolf[idx2] = new_wolf[idx2], new_wolf[idx1]

        # 2. Odd-even position swapping
        for i in range(1, length, 2):
            if i + 1 < length:
                new_wolf[i], new_wolf[i + 1] = new_wolf[i + 1], new_wolf[i]

        # 3. Even-odd position swapping
        for i in range(0, length - 1, 2):
            new_wolf[i], new_wolf[i + 1] = new_wolf[i + 1], new_wolf[i]

        # 4. Segment reversal (reverse a random segment)
        start = random.randint(0, length - 2)
        end = random.randint(start + 1, length - 1)
        new_wolf[start:end + 1] = new_wolf[start:end + 1][::-1]

        # 5. Gene flipping (bit flip at random position)
        flip_idx = random.randint(0, length - 1)
        new_wolf[flip_idx] = 1 - new_wolf[flip_idx]

        return new_wolf

    def exploitation_operator(self):
        """Update positions based on alpha, beta, delta using majority voting"""
        # Majority voting of alpha, beta, delta at each gene position
        leaders = np.vstack([self.alpha_pos, self.beta_pos, self.delta_pos])
        majority_vote = (np.sum(leaders, axis=0) >= 2).astype(int)
        return majority_vote

    def optimize(self):
        """Main DGWA optimization loop"""
        # Evaluate initial fitness
        for i in range(self.pop_size):
            self.fitness[i] = self.fitness_function(self.population[i])
        self.update_leaders()

        for iteration in range(self.max_iter):
            new_population = []

            for wolf in self.population:
                if self.use_exploration_ops:
                    # Full DGWA: combine exploration + exploitation
                    candidate = self.exploration_operators(wolf)
                    exploitation_pos = self.exploitation_operator()
                    updated_wolf = np.where(np.random.rand(self.feature_count) < 0.5, candidate, exploitation_pos)
                else:
                    # MGWA: only use exploitation
                    updated_wolf = self.exploitation_operator()

                new_population.append(updated_wolf)

            self.population = np.array(new_population)

            # Evaluate new fitness
            for i in range(self.pop_size):
                self.fitness[i] = self.fitness_function(self.population[i])

            self.update_leaders()

            if self.verbose:
                print(f"Iteration {iteration + 1}/{self.max_iter}, Best fitness: {self.alpha_score:.4f}")

        return self.alpha_pos, self.alpha_score
