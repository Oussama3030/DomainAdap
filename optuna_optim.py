import os
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import optuna
from optuna.trial import TrialState
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import precision_recall_curve, auc
from tqdm import tqdm
from torch.autograd import Function
import pandas as pd
from sklearn.preprocessing import StandardScaler

optuna.logging.set_verbosity(optuna.logging.DEBUG)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DIR = os.getcwd()
LOG_INTERVAL = 10

# Define a gradient reversal layer
class GradReverse(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None

def grad_reverse(x, alpha):
    return GradReverse.apply(x, alpha)

def get_activation_function(name):
    activations = {
        "relu": nn.ReLU(),
        "leaky_relu": nn.LeakyReLU(0.1),
        "elu": nn.ELU(),
        "selu": nn.SELU(),
        "tanh": nn.Tanh(),
        "gelu": nn.GELU()
    }
    return activations[name]

def define_model(trial):

    # Output dimension of the feature extractor
    feature_out_dim = trial.suggest_categorical("feature_out_size", [32, 64, 128, 256, 512])
    
    # Number of hidden layers
    feature_n_layers = trial.suggest_int("feature_n_layers", 1, 3)
    class_n_layers = trial.suggest_int("class_n_layers", 1, 3)
    domain_n_layers = trial.suggest_int("domain_n_layers", 1, 3)
    
    # Feature extractor
    feature_extractor = FeatureExtractor(feature_out_dim, feature_n_layers, trial) 

    # Class predictor
    class_predictor = ClassPredictor(feature_out_dim, class_n_layers, trial)
    
    # Domain discriminator
    domain_discriminator = DomainDiscriminator(feature_out_dim, domain_n_layers, trial)
    
    return feature_extractor, class_predictor, domain_discriminator


class FeatureExtractor(nn.Module):
    def __init__(self, output_size, n_layers, trial=None):
        super(FeatureExtractor, self).__init__()
        
        # Get activation function from trial
        activation_name = trial.suggest_categorical("feature_activation", 
                                                  ["relu", "leaky_relu", "elu", "selu", "tanh", "gelu"])
        self.activation = get_activation_function(activation_name)

        # Get dropout rate from trial
        dropout_rate = trial.suggest_float("feature_dropout", 0.1, 0.5)

        # Create layers
        self.layers = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        
        # Input dimension
        input_dim = 7
        
        # Dynamically create hidden layers
        for i in range(n_layers):
            # Suggest layer size
            hidden_size = trial.suggest_categorical(f"feature_hidden_size_{i+1}", [64, 128, 256, 512])
            
            # Add layer and dropout
            self.layers.append(nn.Linear(input_dim, hidden_size))
            self.dropouts.append(nn.Dropout(dropout_rate))
            
            # Update input dim for next layer
            input_dim = hidden_size
        
        # Output layer
        self.layer_out = nn.Linear(input_dim, output_size)

    def forward(self, x):
        # Pass through hidden layers
        for layer, dropout in zip(self.layers, self.dropouts):
            x = self.activation(layer(x))
            x = dropout(x)
        
        # Output layer
        x = self.layer_out(x)
        return x


class ClassPredictor(nn.Module):
    def __init__(self, input_size, n_layers, trial=None):
        super(ClassPredictor, self).__init__()
        
        # Get activation function from trial
        activation_name = trial.suggest_categorical("class_activation", 
                                                  ["relu", "leaky_relu", "elu", "selu", "tanh", "gelu"])
        self.activation = get_activation_function(activation_name)
        
        # Get dropout rate from trial
        dropout_rate = trial.suggest_float("class_dropout", 0.1, 0.5)

        # Create layers
        self.layers = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        
        # First layer input dimension
        current_dim = input_size
        
        # Dynamically create hidden layers
        for i in range(n_layers):
            # Suggest layer size
            hidden_size = trial.suggest_categorical(f"class_hidden_size_{i+1}", [64, 128, 256, 512])
            
            # Add layer and dropout
            self.layers.append(nn.Linear(current_dim, hidden_size))
            self.dropouts.append(nn.Dropout(dropout_rate))
            
            # Update input dim for next layer
            current_dim = hidden_size
        
        # Output layer
        self.layer_out = nn.Linear(current_dim, 1)

    def forward(self, x):
        # Pass through hidden layers
        for layer, dropout in zip(self.layers, self.dropouts):
            x = self.activation(layer(x))
            x = dropout(x)
        
        # Output layer
        x = self.layer_out(x)
        return x


class DomainDiscriminator(nn.Module):
    def __init__(self, input_size, n_layers, trial=None):
        super(DomainDiscriminator, self).__init__()
        self.alpha = 0
        
        # Get activation function from trial
        activation_name = trial.suggest_categorical("domain_activation", 
                                                  ["relu", "leaky_relu", "elu", "selu", "tanh", "gelu"])
        self.activation = get_activation_function(activation_name)
        
        # Get dropout rate from trial
        dropout_rate = trial.suggest_float("domain_dropout", 0.1, 0.5)
        
        # Create layers
        self.layers = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        
        # First layer input dimension
        current_dim = input_size
        
        # Dynamically create hidden layers
        for i in range(n_layers):
            # Suggest layer size
            hidden_size = trial.suggest_categorical(f"domain_hidden_size_{i+1}", [64, 128, 256, 512])
            
            # Add layer and dropout
            self.layers.append(nn.Linear(current_dim, hidden_size))
            self.dropouts.append(nn.Dropout(dropout_rate))
            
            # Update input dim for next layer
            current_dim = hidden_size
        
        # Output layer
        self.layer_out = nn.Linear(current_dim, 1)
    
    def forward(self, x):
        x = grad_reverse(x, self.alpha)
        
        # Pass through hidden layers
        for layer, dropout in zip(self.layers, self.dropouts):
            x = self.activation(layer(x))
            x = dropout(x)
        
        # Output layer
        x = self.layer_out(x)
        return x


def calculate_alpha(current_step, total_steps, gamma=10):

    progress = float(current_step) / total_steps
    return 2.0 / (1.0 + np.exp(-gamma * progress)) - 1.0

        
def load_data(trial):

    BATCHSIZE = trial.suggest_categorical("batch_size", [32, 64, 128, 256])

    training_samples_mc = np.load('/Users/oussamabenchikhi/o2workdir/PID/ML/Data/DANN_training_samples_mc.npy')
    mapped_targets_mc = np.load('/Users/oussamabenchikhi/o2workdir/PID/ML/Data/DANN_mapped_targets_mc.npy')

    training_samples_real = np.load('/Users/oussamabenchikhi/o2workdir/PID/ML/Data/DANN_training_samples_raw.npy')

    # Split Monte Carlo data into training and testing sets with stratification to maintain class balance
    training_input_mc, testing_input_mc, training_target_mc, testing_target_mc = train_test_split(training_samples_mc,
                                                                                            mapped_targets_mc,
                                                                                            test_size=0.3,
                                                                                            stratify=mapped_targets_mc,
                                                                                            random_state=42)
    
    training_input_real, testing_input_real = train_test_split(training_samples_real, 
                                                           test_size=0.3, 
                                                           random_state=42)

    # Extract only the first 3 features for model input (dE/dx, p, TOF)
    # training_input_mc = training_input_mc[:, :3]
    # testing_input_mc = testing_input_mc[:, :3]

    # training_input_real = training_input_real[:, :3]
    # testing_input_real = testing_input_real[:, :3]

    mask_training = (training_input_mc[:, 1] > 0.75) & (training_input_mc[:, 1] < 1.5)
    mask_testing = (testing_input_mc[:, 1] > 0.75) & (testing_input_mc[:, 1] < 1.5)

    training_input_mc = training_input_mc#[mask_training]
    testing_input_mc = testing_input_mc#[mask_testing]
    training_input_real = training_input_real#[(training_input_real[:, 1] > 0.75) & (training_input_real[:, 1] < 1.5)]
    testing_input_real = testing_input_real#[(testing_input_real[:, 1] > 0.75) &(testing_input_real[:, 1] < 1.5)]

    training_target_mc = training_target_mc#[mask_training]
    testing_target_mc = testing_target_mc#[mask_testing]

    # Configure DEVICE for computation (GPU if available, otherwise CPU)
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Convert Monte Carlo data to PyTorch tensors and move to DEVICE
    training_input_mc = torch.from_numpy(training_input_mc).type(torch.float).to(DEVICE)
    training_target_mc = torch.from_numpy(training_target_mc).type(torch.float).to(DEVICE)

    testing_input_mc = torch.from_numpy(testing_input_mc).type(torch.float).to(DEVICE)
    testing_target_mc = torch.from_numpy(testing_target_mc).type(torch.float).to(DEVICE)

    # Convert real data to PyTorch tensors and move to DEVICE
    training_input_real = torch.from_numpy(training_input_real).type(torch.float).to(DEVICE)

    testing_input_real = torch.from_numpy(testing_input_real).type(torch.float).to(DEVICE)

    # Create TensorDatasets for Monte Carlo data
    train_dataset_mc = TensorDataset(training_input_mc, training_target_mc)
    test_dataset_mc = TensorDataset(testing_input_mc, testing_target_mc)

    train_dataset_real = TensorDataset(training_input_real)
    test_dataset_real = TensorDataset(testing_input_real)

    # Create DataLoaders for batch processing of Monte Carlo data
    train_loader_mc = DataLoader(train_dataset_mc, batch_size=BATCHSIZE, shuffle=True, drop_last=True)
    test_loader_mc = DataLoader(test_dataset_mc, batch_size=BATCHSIZE, shuffle=True, drop_last=True)

    train_loader_real = DataLoader(train_dataset_real, batch_size=BATCHSIZE, shuffle=True, drop_last=True)
    test_loader_real = DataLoader(test_dataset_real, batch_size=BATCHSIZE, shuffle=True, drop_last=True)

    return train_loader_mc, test_loader_mc, train_loader_real, test_loader_real

def set_model_mode(mode='train', models=None):
    for model in models:
        if mode == 'train':
            model.train()
        else:
            model.eval()

def objective(trial):
    try:
        # Generate the model
        feature_extractor, class_predictor, domain_discriminator = define_model(trial)

        feature_extractor.to(DEVICE)
        class_predictor.to(DEVICE)
        domain_discriminator.to(DEVICE)

        lr_feature = trial.suggest_float("lr_feature", 1e-7, 1e-2, log=True)
        lr_class = trial.suggest_float("lr_class", 1e-7, 1e-2, log=True)
        lr_domain = trial.suggest_float("lr_domain", 1e-7, 1e-2, log=True)

        optimizer_feature = optim.Adam(feature_extractor.parameters(), lr=lr_feature)
        optimizer_class = optim.Adam(class_predictor.parameters(), lr=lr_class)
        optimizer_domain = optim.Adam(domain_discriminator.parameters(), lr=lr_domain)

        # Generate the data
        train_loader_mc, test_loader_mc, train_loader_real, test_loader_real = load_data(trial)

        training_target_mc = train_loader_mc.dataset.tensors[1]

        # Calculate the pos_weight (neg/pos ratio)
        num_negatives = (training_target_mc == 0).sum().item()
        num_positives = (training_target_mc == 1).sum().item()

        pos_weight = num_negatives / num_positives
            
        pos_weight_tensor = torch.tensor([pos_weight]).to(DEVICE)

        # Generate the loss
        label_loss_fn = nn.BCEWithLogitsLoss()
        domain_loss_fn = nn.BCEWithLogitsLoss()

        EPOCHS = 30
        total_steps = EPOCHS * min(len(train_loader_mc), len(train_loader_real))

        set_model_mode('train', [feature_extractor, class_predictor, domain_discriminator])
        best_combined_score = 0.0

        for epoch in range(EPOCHS):

            batches = zip(train_loader_mc, train_loader_real)
            n_batches = min(len(train_loader_mc), len(train_loader_real))

            batches_test = zip(test_loader_mc, test_loader_real)
            n_batches_test = min(len(test_loader_mc), len(test_loader_real))

            for batch_idx, ((source_x, source_labels), (target_x,)) in enumerate(tqdm(batches, leave=False, total=n_batches)):

                # as suggested in the paper
                p = float(batch_idx + epoch * n_batches) / total_steps
                lambd = 2. / (1. + np.exp(-10. * p)) - 1
                
                # Update the alpha value in the domain discriminator
                domain_discriminator.alpha = lambd

                # Skip if batch sizes don't match
                if source_x.size(0) != target_x.size(0):
                    continue
                    
                # Move data to device
                source_x = source_x.to(DEVICE)
                source_labels = source_labels.to(DEVICE)
                target_x = target_x.to(DEVICE)
                
                # Create domain labels (1 for MC, 0 for real data)
                domain_y = torch.cat([torch.ones(source_x.size(0)), 
                                    torch.zeros(target_x.size(0))]).to(DEVICE)
                
                # 1) Train feature_extractor and class_classifier on source batch only
                optimizer_feature.zero_grad()
                optimizer_class.zero_grad()
                
                # Extract features and get class predictions for source data
                source_features = feature_extractor(source_x).view(source_x.shape[0], -1)
                label_preds = class_predictor(source_features).squeeze()
                
                # Compute class loss and optimize
                label_loss = label_loss_fn(label_preds, source_labels.float())
                label_loss.backward(retain_graph=True)
                optimizer_class.step()
                optimizer_feature.step()
                
                # 2) Train feature_extractor and domain_classifier on full batch
                optimizer_feature.zero_grad()
                optimizer_domain.zero_grad()
                
                # Concatenate source and target data
                x = torch.cat([source_x, target_x])
                
                # Extract features and get domain predictions
                features = feature_extractor(x).view(x.shape[0], -1)
                domain_preds = domain_discriminator(features).squeeze()
                
                # Compute domain loss and optimize
                domain_loss = domain_loss_fn(domain_preds, domain_y.float())
                domain_loss.backward(retain_graph=True)
                optimizer_domain.step()
                optimizer_feature.step()

            # Set models to eval
            set_model_mode('eval', [feature_extractor, class_predictor, domain_discriminator])

            # Evaluation phase
            label_targets = []
            label_probabilities = []

            domain_targets = []
            domain_predictions = []
            domain_probabilities = []

            with torch.no_grad():
                for batch_idx, ((source_x, source_labels), (target_x,)) in enumerate(tqdm(batches_test, leave=False, total=n_batches_test)):

                    final_lambda = 20 #2. / (1. + np.exp(-10. * 1.0)) - 1 # 0 was 2
                    domain_discriminator.alpha = final_lambda
                                    
                    x = torch.cat([source_x, target_x]).to(DEVICE)
                    domain_y = torch.cat([torch.ones(source_x.size(0)),
                                        torch.zeros(target_x.size(0))]).to(DEVICE)
                    label_y = source_labels.to(DEVICE)  

                    features = feature_extractor(x).view(x.shape[0], -1)
                    domain_out = domain_discriminator(features).squeeze()
                    label_out = class_predictor(features[:source_x.shape[0]]).squeeze()
                    
                    # Calculate the label probabilities
                    label_probs = torch.sigmoid(label_out)

                    # Predict the domains
                    domain_preds = (torch.sigmoid(domain_out) > 0.5).float()
                    domain_probs = torch.sigmoid(domain_out)

                    # Handle potential scalar outputs for label_y and label_probs
                    label_y_np = label_y.cpu().numpy()
                    if np.isscalar(label_y_np) or label_y_np.ndim == 0:
                        label_targets.append(float(label_y_np))
                    else:
                        label_targets.extend(label_y_np)
                    
                    label_probs_np = label_probs.cpu().numpy()
                    if np.isscalar(label_probs_np) or label_probs_np.ndim == 0:
                        label_probabilities.append(float(label_probs_np))
                    else:
                        label_probabilities.extend(label_probs_np)
                    
                    # Handle potential scalar outputs for domain_y, domain_preds, and domain_probs
                    domain_y_np = domain_y.cpu().numpy()
                    if np.isscalar(domain_y_np) or domain_y_np.ndim == 0:
                        domain_targets.append(float(domain_y_np))
                    else:
                        domain_targets.extend(domain_y_np)
                    
                    domain_preds_np = domain_preds.cpu().numpy()
                    if np.isscalar(domain_preds_np) or domain_preds_np.ndim == 0:
                        domain_predictions.append(float(domain_preds_np))
                    else:
                        domain_predictions.extend(domain_preds_np)
                    
                    domain_probs_np = domain_probs.cpu().numpy()
                    if np.isscalar(domain_probs_np) or domain_probs_np.ndim == 0:
                        domain_probabilities.append(float(domain_probs_np))
                    else:
                        domain_probabilities.extend(domain_probs_np)

            # Check for NaN values before precision-recall calculation
            if np.isnan(label_probabilities).any():
                print(f"Trial {trial.number}: NaN values detected")
                return 0.0  # Return a poor score instead of failing

            # Calculate AUPRC
            precision, recall, _ = precision_recall_curve(label_targets, label_probabilities)
            auprc = auc(recall, precision)

            # Calculate domain accuracy
            domain_acc = np.mean(np.array(domain_predictions) == np.array(domain_targets)).astype(float)

            domain_confusion = 1.0 - 2.0 * abs(domain_acc - 0.5)  # A domain accuracy of 0.5 maximizes this value
            combined_score = 0.5*domain_confusion + 0.5*auprc

            print("Domain Accuracy: ", domain_acc, "auprc: ", auprc, "combined: ", combined_score)
                
            if combined_score > best_combined_score:
                best_combined_score = combined_score
    
            # Report the best metric to Optuna
            trial.report(best_combined_score, epoch)

            # Handle pruning
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        return best_combined_score
        

    except Exception as e:
        print(f"Trial {trial.number} failed with error: {str(e)}")
        return 0.0 


if __name__ == "__main__":
   
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42, n_startup_trials=10),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=15,  n_warmup_steps=10),
    )

    study.optimize(objective, n_trials=50, catch=(ValueError, RuntimeError, torch.cuda.OutOfMemoryError))
    pruned_trials = study.get_trials(states=(optuna.trial.TrialState.PRUNED,))
    complete_trials = study.get_trials(states=(optuna.trial.TrialState.COMPLETE,))

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    # Plot optimization results
    try:
        fig1 = optuna.visualization.plot_optimization_history(study)
        fig1.show()
        
        fig2 = optuna.visualization.plot_parallel_coordinate(study)
        fig2.show()
        
        fig3 = optuna.visualization.plot_param_importances(study)
        fig3.show()
        
        fig4 = optuna.visualization.plot_contour(study)
        fig4.show()
        
        fig5 = optuna.visualization.plot_slice(study)
        fig5.show()
    except:
        print("Error creating visualization.")


    print("Best model:")
    feature_extractor, class_predictor, domain_discriminator = define_model(trial)
    print(f"Feature Extractor: {feature_extractor}")
    print(f"Class Predictor: {class_predictor}")
    print(f"Domain Discriminator: {domain_discriminator}")