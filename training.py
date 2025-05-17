import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_curve, auc
from torch.nn.utils import clip_grad_norm_  

# Import the models from DANN module
from model import FeatureExtractor, ClassClassifier, DomainClassifier, GradReverse

def set_hyperparameters():
    
    params = {
        'n_epochs': 200,
        'batchsize': 128,
        'lr_feature': 7.787547434272369e-05,
        'lr_class': 1.5125556736397536e-07,
        'lr_domain': 2.1280904885895868e-05
    }
    return params


def calculate_alpha(current_step, total_steps, gamma=10):

    progress = float(current_step) / total_steps
    return 2.0 / (1.0 + np.exp(-gamma * progress)) - 1.0

def load_data(mc_file_path, mc_targets_path, real_file_path):

    # Load Monte Carlo data
    training_samples_mc = np.load(mc_file_path)
    mapped_targets_mc = np.load(mc_targets_path)
    
    # Load real experimental data
    training_samples_real = np.load(real_file_path)
    
    return training_samples_mc, mapped_targets_mc, training_samples_real

def split_data(mc_samples, mc_targets, real_samples, test_size=0.3, random_state=42):

    # Split Monte Carlo data with stratification to maintain class balance
    training_input_mc, testing_input_mc, training_target_mc, testing_target_mc = train_test_split(
        mc_samples,
        mc_targets,
        test_size=test_size,
        stratify=mc_targets,
        random_state=random_state
    )
    
    # Split real data into training and testing sets
    training_input_real, testing_input_real = train_test_split(
        real_samples, 
        test_size=test_size, 
        random_state=random_state
    )
    
    return (
        training_input_mc, testing_input_mc, training_target_mc, testing_target_mc,
        training_input_real, testing_input_real
    )

def preprocess_data(training_input_mc, testing_input_mc, training_input_real, testing_input_real, training_target_mc, testing_target_mc):
    
    training_input_mc = training_input_mc
    testing_input_mc = testing_input_mc
    training_input_real = training_input_real
    testing_input_real = testing_input_real
    training_target_mc = training_target_mc
    testing_target_mc = testing_target_mc

    # # Extract only particles with momentum between 0 and 0.75
    # mask_training = training_input_mc[:, 1] < 0.75
    # mask_testing = testing_input_mc[:, 1] < 0.75

    # training_input_mc = training_input_mc[mask_training]
    # testing_input_mc = testing_input_mc[mask_testing]
    # training_input_real = training_input_real[training_input_real[:, 1] < 0.75]
    # testing_input_real = testing_input_real[testing_input_real[:, 1] < 0.75]

    # training_target_mc = training_target_mc[mask_training]
    # testing_target_mc = testing_target_mc[mask_testing]

    # # Extract only particles with momentum between 0.75 and 1.5
    # mask_training = (training_input_mc[:, 1] >= 0.75) & (training_input_mc[:, 1] < 1.5)
    # mask_testing = (testing_input_mc[:, 1] >= 0.75) & (testing_input_mc[:, 1] < 1.5)

    # training_input_mc = training_input_mc[mask_training]
    # testing_input_mc = testing_input_mc[mask_testing]
    # training_input_real = training_input_real[(training_input_real[:, 1] >= 0.75) & (training_input_real[:, 1] < 1.5)]
    # testing_input_real = testing_input_real[(testing_input_real[:, 1] >= 0.75) & (testing_input_real[:, 1] < 1.5)]

    # training_target_mc = training_target_mc[mask_training]
    # testing_target_mc = testing_target_mc[mask_testing]

    # Extract only particles with momentum larger than 1.5
    # mask_training = (training_input_mc[:, 1] >= 1.5) 
    # mask_testing = (testing_input_mc[:, 1] >= 1.5) 

    # training_input_mc = training_input_mc[mask_training]
    # testing_input_mc = testing_input_mc[mask_testing]
    # training_input_real = training_input_real[training_input_real[:, 1] >= 1.5]
    # testing_input_real = testing_input_real[testing_input_real[:, 1] >= 1.5]

    # training_target_mc = training_target_mc[mask_training]
    # testing_target_mc = testing_target_mc[mask_testing]

    return training_input_mc, testing_input_mc, training_input_real, testing_input_real, training_target_mc, testing_target_mc#, scaler

def create_dataloaders(training_input_mc, training_target_mc, testing_input_mc, testing_target_mc,
                      training_input_real, testing_input_real, batch_size, device):

    # Convert Monte Carlo data to PyTorch tensors and move to device
    training_input_mc = torch.from_numpy(training_input_mc).type(torch.float).to(device)
    training_target_mc = torch.from_numpy(training_target_mc).type(torch.float).to(device)
    testing_input_mc = torch.from_numpy(testing_input_mc).type(torch.float).to(device)
    testing_target_mc = torch.from_numpy(testing_target_mc).type(torch.float).to(device)
    
    # Create TensorDatasets for Monte Carlo data
    train_dataset_mc = TensorDataset(training_input_mc, training_target_mc)
    test_dataset_mc = TensorDataset(testing_input_mc, testing_target_mc)
    
    # Create DataLoaders for batch processing of Monte Carlo data
    train_loader_mc = DataLoader(train_dataset_mc, batch_size=batch_size, shuffle=True)
    test_loader_mc = DataLoader(test_dataset_mc, batch_size=batch_size, shuffle=True)
    
    # Convert real data to PyTorch tensors and move to device
    training_input_real = torch.from_numpy(training_input_real).type(torch.float).to(device)
    testing_input_real = torch.from_numpy(testing_input_real).type(torch.float).to(device)
    
    # Create TensorDatasets for real data (no labels)
    train_dataset_real = TensorDataset(training_input_real)
    test_dataset_real = TensorDataset(testing_input_real)
    
    # Create DataLoaders for batch processing of real data
    train_loader_real = DataLoader(train_dataset_real, batch_size=batch_size, shuffle=True)
    test_loader_real = DataLoader(test_dataset_real, batch_size=batch_size, shuffle=True)
    
    return train_loader_mc, test_loader_mc, train_loader_real, test_loader_real

def initialize_models(device):

    feature_extractor = FeatureExtractor().to(device)
    class_classifier = ClassClassifier().to(device)
    domain_classifier = DomainClassifier().to(device)
    
    return feature_extractor, class_classifier, domain_classifier

def setup_training(models, params, training_target_mc, device):

    feature_extractor, class_classifier, domain_classifier = models
    
    # Setup separate optimizers for each component with specific optimizer types and weight decays
    feature_optimizer = optim.Adam(
        feature_extractor.parameters(), 
        lr=params['lr_feature']
                                        )
    
        # weight_decay=params['weight_decay_feature']
    
    class_optimizer = optim.Adam(
        class_classifier.parameters(), 
        lr=params['lr_class']

    )
            # weight_decay=params['weight_decay_class']
    
    domain_optimizer = optim.Adam(
        domain_classifier.parameters(), 
        lr=params['lr_domain']
                )
    
    optimizers = {
        'feature': feature_optimizer,
        'class': class_optimizer,
        'domain': domain_optimizer
    }

    # Calculate the pos_weight (neg/pos ratio)
    num_negatives = (training_target_mc == 0).sum().item()
    num_positives = (training_target_mc == 1).sum().item()

    pos_weight = num_negatives / num_positives
        
    pos_weight_tensor = torch.tensor([pos_weight]).to(device)
    
    # Loss functions for classification and domain adaptation
    class_loss_fn = nn.BCEWithLogitsLoss()
    domain_loss_fn = nn.BCEWithLogitsLoss() 
    
    return optimizers, class_loss_fn, domain_loss_fn


def set_model_mode(mode='train', models=None):

    for model in models:
        if mode == 'train':
            model.train()
        else:
            model.eval()


def save_models(models, save_paths):

    feature_extractor, class_classifier, domain_classifier = models
    feature_extractor_path, class_classifier_path, domain_classifier_path = save_paths
    
    torch.save(feature_extractor.state_dict(), feature_extractor_path)
    torch.save(class_classifier.state_dict(), class_classifier_path)
    torch.save(domain_classifier.state_dict(), domain_classifier_path)
    
    print(f"Models saved to {feature_extractor_path}, {class_classifier_path}, and {domain_classifier_path}")


def train_epoch(train_loader_mc, train_loader_real, test_loader_mc, test_loader_real, models, optimizers, 
               class_loss_fn, domain_loss_fn, epoch, n_epochs, device, gamma=10, max_grad_norm=1.0):

    feature_extractor, class_classifier, domain_classifier = models
    
    # Set models to training mode
    set_model_mode('train', [feature_extractor, class_classifier, domain_classifier])
    
    # Zip MC and real data loaders to iterate through both simultaneously
    batches = zip(train_loader_mc, train_loader_real)
    n_batches = min(len(train_loader_mc), len(train_loader_real))

    batches_test = zip(test_loader_mc, test_loader_real)
    n_batches_test = min(len(test_loader_mc), len(test_loader_real))

    total_steps = n_epochs * n_batches
    
    # Initialize metrics for this epoch
    total_domain_loss = 0
    total_class_loss = 0
    total_label_accuracy = 0
    total_domain_accuracy = 0
    
    # Lists to store all predictions and labels for evaluation
    all_domain_labels = []
    all_domain_predictions = []
    all_domain_probabilities = []
    
    all_class_labels = []
    all_class_predictions = []
    all_class_probabilities = []
    
    for batch_idx, ((source_x, source_labels), (target_x,)) in enumerate(tqdm(batches, leave=False, total=n_batches)):

        # as suggested in the paper
        p = float(batch_idx + epoch * n_batches) / total_steps
        lambd = 2. / (1. + np.exp(-10. * p)) - 1
        
        # Update the alpha value in the domain discriminator
        domain_classifier.alpha = lambd
        
        # Skip if batch sizes don't match
        if source_x.size(0) != target_x.size(0):
            continue
            
        # Move data to device
        source_x = source_x.to(device)
        source_labels = source_labels.to(device)
        target_x = target_x.to(device)
        
        # Create domain labels (1 for MC, 0 for real data)
        domain_y = torch.cat([torch.ones(source_x.size(0)), 
                            torch.zeros(target_x.size(0))]).to(device)
        
        # 1) Train feature_extractor and class_classifier on source batch only
        optimizers['feature'].zero_grad()
        optimizers['class'].zero_grad()
        
        # Extract features and get class predictions for source data
        source_features = feature_extractor(source_x).view(source_x.shape[0], -1)
        label_preds = class_classifier(source_features).squeeze()
        
        # Compute class loss and optimize
        label_loss = class_loss_fn(label_preds, source_labels.float())
        label_loss.backward(retain_graph=True)

        optimizers['class'].step()
        optimizers['feature'].step()
        
        # 2) Train feature_extractor and domain_classifier on full batch
        optimizers['feature'].zero_grad()
        optimizers['domain'].zero_grad()
        
        # Concatenate source and target data
        x = torch.cat([source_x, target_x])
        
        # Extract features and get domain predictions
        features = feature_extractor(x).view(x.shape[0], -1)
        domain_preds = domain_classifier(features).squeeze()
        
        # Compute domain loss and optimize
        domain_loss = domain_loss_fn(domain_preds, domain_y.float())
        domain_loss.backward(retain_graph=True)
        optimizers['domain'].step()
        optimizers['feature'].step()
        
        # Calculate metrics using current model outputs
        with torch.no_grad():
            features = feature_extractor(x).view(x.shape[0], -1)
            domain_preds = domain_classifier(features).squeeze()
            source_features = feature_extractor(source_x).view(source_x.shape[0], -1)
            label_preds = class_classifier(source_features).squeeze()
            
            # Calculate class prediction accuracy
            class_probabilities = torch.sigmoid(label_preds)
            predicted_class = (class_probabilities > 0.5).float()
            total_label_accuracy += (predicted_class == source_labels).sum().item() / source_labels.size(0)
            
            # Calculate domain prediction accuracy
            domain_probabilities = torch.sigmoid(domain_preds)
            predicted_domain = (domain_probabilities > 0.5).float()
            total_domain_accuracy += (predicted_domain == domain_y).sum().item() / domain_y.size(0)
            
            # Store all predictions and labels for evaluation
            all_domain_labels.extend(domain_y.cpu().numpy())
            all_domain_predictions.extend(predicted_domain.cpu().numpy())
            all_domain_probabilities.extend(domain_probabilities.detach().cpu().numpy())
            
            all_class_labels.extend(source_labels.cpu().numpy())
            all_class_predictions.extend(predicted_class.cpu().numpy())
            all_class_probabilities.extend(class_probabilities.detach().cpu().numpy())
        
        # Track metrics for this batch
        total_domain_loss += domain_loss.item()
        total_class_loss += label_loss.item()
    
    # Calculate mean metrics for this epoch
    mean_domain_loss = total_domain_loss / n_batches
    mean_class_loss = total_class_loss / n_batches
    mean_class_accuracy = total_label_accuracy / n_batches
    mean_domain_accuracy = total_domain_accuracy / n_batches

    # Evaluate on test data
    label_targets_test = []
    label_probabilities_test = []

    domain_targets_test = []
    domain_predictions_test = []
    domain_probabilities_test = []

    # Lists to collect outputs for loss calculation
    label_out_list = []
    label_y_list = []
    domain_out_list = []
    domain_y_list = []

    # Track test losses
    test_class_loss_total = 0
    test_domain_loss_total = 0
    test_batches_count = 0

    # Set models to evaluation mode
    set_model_mode('eval', [feature_extractor, class_classifier, domain_classifier])

    with torch.no_grad():
        for batch_idx, ((source_x, source_labels), (target_x,)) in enumerate(tqdm(batches_test, leave=False, total=n_batches_test)):
            # Skip if batch sizes don't match
            if source_x.size(0) != target_x.size(0):
                continue
                
            test_batches_count += 1
            
            # Use final lambda for evaluation (at p=1.0)
            final_lambda = 2. / (1. + np.exp(-10. * 1.0)) - 1 # 20
            domain_classifier.alpha = final_lambda

            # Move data to device
            source_x = source_x.to(device)
            source_labels = source_labels.to(device)
            target_x = target_x.to(device)

            x = torch.cat([source_x, target_x])
            domain_y = torch.cat([torch.ones(source_x.size(0)), 
                                torch.zeros(target_x.size(0))]).to(device)
            label_y = source_labels

            # Extract features
            features = feature_extractor(x).view(x.shape[0], -1)
            domain_out = domain_classifier(features).squeeze()
            
            # Get label predictions for source data only
            source_features = feature_extractor(source_x).view(source_x.shape[0], -1)
            label_out = class_classifier(source_features).squeeze()

            # Calculate the label probabilities
            label_probs = torch.sigmoid(label_out)

            # Predict the domains
            domain_preds = (torch.sigmoid(domain_out) > 0.5).float()
            domain_probs = torch.sigmoid(domain_out)

            # Store the predictions and labels
            label_targets_test.extend(label_y.cpu().numpy())
            label_probabilities_test.extend(label_probs.cpu().numpy())

            domain_targets_test.extend(domain_y.cpu().numpy())
            domain_predictions_test.extend(domain_preds.cpu().numpy())
            domain_probabilities_test.extend(domain_probs.cpu().numpy())

            # Calculate losses for this batch
            current_class_loss = class_loss_fn(label_out, label_y.float()).item()
            current_domain_loss = domain_loss_fn(domain_out, domain_y.float()).item()
            
            # Accumulate test losses
            test_class_loss_total += current_class_loss
            test_domain_loss_total += current_domain_loss

    # Calculate mean test losses
    test_class_loss = test_class_loss_total / test_batches_count if test_batches_count > 0 else 0
    test_domain_loss = test_domain_loss_total / test_batches_count if test_batches_count > 0 else 0

    # Return metrics and predictions
    return {
        'domain_loss': mean_domain_loss,
        'class_loss': mean_class_loss,
        'class_accuracy': mean_class_accuracy,
        'domain_accuracy': mean_domain_accuracy,
        'all_domain_labels': all_domain_labels,
        'all_domain_predictions': all_domain_predictions,
        'all_domain_probabilities': all_domain_probabilities,
        'all_class_labels': all_class_labels,
        'all_class_predictions': all_class_predictions,
        'all_class_probabilities': all_class_probabilities,
        'label_targets_test': label_targets_test,
        'label_probabilities_test': label_probabilities_test,
        'domain_targets_test': domain_targets_test,
        'domain_predictions_test': domain_predictions_test,
        'domain_probabilities_test': domain_probabilities_test,
        'test_class_loss': test_class_loss, 
        'test_domain_loss': test_domain_loss
    }

def train_dann(params, data_loaders, models, optimizers, class_loss_fn, domain_loss_fn, device):

    train_loader_mc, test_loader_mc, train_loader_real, test_loader_real = data_loaders
    
    # Extract hyperparameters
    n_epochs = params['n_epochs']
    gamma = params.get('gamma', 10) 
    
    # Lists for tracking metrics during training
    class_losses = []
    domain_losses = []
    class_accuracies = []
    domain_accuracies = []
    test_domain_accuracies = [] 
    label_probs = []
    label_targets = []

    # New lists for tracking test losses
    test_class_losses = []
    test_domain_losses = []
    auprc_per_epoch = []
    train_auprc_per_epoch = []  
    
    # Final training metrics to return
    final_metrics = {}
    
    # Main training loop
    for epoch in tqdm(range(n_epochs), desc="Training epochs"):
        # Train for one epoch with dynamic alpha
        epoch_metrics = train_epoch(
            train_loader_mc, train_loader_real, test_loader_mc, test_loader_real, models, optimizers,
            class_loss_fn, domain_loss_fn, epoch, n_epochs, device, gamma, params.get('max_grad_norm', 1.0)
        )        

        # Track metrics
        class_losses.append(epoch_metrics['class_loss'])
        domain_losses.append(epoch_metrics['domain_loss'])
        class_accuracies.append(epoch_metrics['class_accuracy'])
        domain_accuracies.append(epoch_metrics['domain_accuracy'])
        
        # Track test losses
        test_class_losses.append(epoch_metrics['test_class_loss'])
        test_domain_losses.append(epoch_metrics['test_domain_loss'])
        
        # Calculate test domain accuracy
        test_domain_accuracy = sum(np.array(epoch_metrics['domain_predictions_test']) == 
                                   np.array(epoch_metrics['domain_targets_test'])) / len(epoch_metrics['domain_targets_test'])
        test_domain_accuracies.append(test_domain_accuracy) 
        
        label_targets = epoch_metrics['label_targets_test']
        label_probs = epoch_metrics['label_probabilities_test']

        # Calculate AUPRC for test data
        precision, recall, _ = precision_recall_curve(label_targets, label_probs)
        pr_auc = auc(recall, precision)
        auprc_per_epoch.append(pr_auc)
        
        # Calculate AUPRC for training data
        train_precision, train_recall, _ = precision_recall_curve(
            epoch_metrics['all_class_labels'], 
            epoch_metrics['all_class_probabilities']
        )
        train_pr_auc = auc(train_recall, train_precision)
        train_auprc_per_epoch.append(train_pr_auc)

        domain_confusion = 1 - 2 * np.abs(0.5 - test_domain_accuracy)
        combined_score = 0.5*domain_confusion + 0.5*pr_auc
        
        best_combined_score = 0.0
        if (combined_score > best_combined_score and 0.495 < test_domain_accuracy < 0.505):
            if (pr_auc > 0.985):

                best_combined_score = combined_score 
                model_save_paths = (
                    'DANN_FeatureExtractor_HighestE.pth',
                    'DANN_ClassClassifier_HighestE.pth',
                    'DANN_DomainClassifier_HighestE.pth'
                )

                # Save trained models
                save_models(models, model_save_paths)
                tqdm.write(f"Found new best model at epoch {epoch + 1} with combined score: {combined_score:.4f} (domain_acc: {test_domain_accuracy:.4f} and AUPRC: {pr_auc:.4f})")

        # Print epoch results
        tqdm.write(f"Epoch {epoch + 1}/{n_epochs}, Domain Loss: {epoch_metrics['domain_loss']:.4f}, "
                   f"Label Loss: {epoch_metrics['class_loss']:.4f}, Label Accuracy: {epoch_metrics['class_accuracy']:.4f}, "
                   f"Domain Accuracy: {epoch_metrics['domain_accuracy']:.4f}, Test Domain Accuracy: {test_domain_accuracy:.4f}, "
                   f"Train AUPRC: {train_pr_auc:.4f}, Test AUPRC: {pr_auc:.4f}, "
                   f"Test Domain Loss: {epoch_metrics['test_domain_loss']:.4f}, Test Class Loss: {epoch_metrics['test_class_loss']:.4f}")
        
        # Save the final epoch metrics
        if epoch == n_epochs - 1:
            final_metrics = epoch_metrics
    
    # Plot training and test losses
    plt.figure(figsize=(12, 5))
    
    # Class loss plot
    plt.subplot(1, 2, 1)
    plt.plot(class_losses, label='Train Class Loss')
    plt.plot(test_class_losses, label='Test Class Loss')
    plt.title('Class Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Domain loss plot
    plt.subplot(1, 2, 2)
    plt.plot(domain_losses, label='Train Domain Loss')
    plt.plot(test_domain_losses, label='Test Domain Loss')
    plt.title('Domain Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Plot Domain Accuracy per epoch for both training and testing
    plt.figure(figsize=(10, 6))
    plt.plot(domain_accuracies, label='Train Domain Accuracy')
    plt.plot(test_domain_accuracies, label='Test Domain Accuracy')
    plt.title("Domain Accuracy per Epoch")
    plt.grid()
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()

    # Plot AUPRC per Epoch for both training and testing
    plt.figure(figsize=(10, 6))
    plt.plot(auprc_per_epoch, label='Test AUPRC')
    plt.plot(train_auprc_per_epoch, label='Train AUPRC')
    plt.title("AUPRC per Epoch")
    plt.grid()
    plt.xlabel("Epoch")
    plt.ylabel("AUPRC")
    plt.legend()
    plt.show()
    
    # Visualize distribution of domain probabilities
    plt.hist(final_metrics['all_domain_probabilities'], bins=100)
    plt.title("Distribution of Domain Probabilities")
    plt.xlabel("Probability")
    plt.ylabel("Count")
    plt.show()

    plt.hist(final_metrics['domain_probabilities_test'], bins=250)
    plt.title("Distribution of Domain Probabilities (Test Data)")
    plt.xlabel("Probability")
    plt.ylabel("Count")
    plt.yscale('log')
    plt.show()

    # Plot precision-recall curve for testing data
    plt.figure(figsize=(10, 6))
    precision, recall, _ = precision_recall_curve(
        final_metrics['label_targets_test'], 
        final_metrics['label_probabilities_test']
    )
    pr_auc = auc(recall, precision)

    plt.plot(recall, precision, lw=2, label=f'Precision-Recall curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve (AUC = {pr_auc:.3f})')
    plt.legend(loc="best")
    plt.grid(True)
    plt.show()
    
    return {
        'class_losses': class_losses,
        'domain_losses': domain_losses,
        'class_accuracies': class_accuracies,
        'domain_accuracies': domain_accuracies,
        'test_domain_accuracies': test_domain_accuracies,
        'train_auprc_per_epoch': train_auprc_per_epoch,
        'auprc_per_epoch': auprc_per_epoch,
        'test_class_losses': test_class_losses,  
        'test_domain_losses': test_domain_losses,
        'final_metrics': final_metrics
    }

def main():
    """Main function to run the DANN training pipeline."""
    # Set file paths
    mc_file_path = '/Users/oussamabenchikhi/o2workdir/PID/ML/Data/DANN_training_samples_mc.npy'
    mc_targets_path = '/Users/oussamabenchikhi/o2workdir/PID/ML/Data/DANN_mapped_targets_mc.npy'
    real_file_path = '/Users/oussamabenchikhi/o2workdir/PID/ML/Data/DANN_training_samples_raw.npy'
    
    # Set hyperparameters
    params = set_hyperparameters()
    
    # Configure device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load data
    training_samples_mc, mapped_targets_mc, training_samples_real = load_data(
        mc_file_path, mc_targets_path, real_file_path
    )
    
    # Split data
    (training_input_mc, testing_input_mc, training_target_mc, testing_target_mc,
     training_input_real, testing_input_real) = split_data(
        training_samples_mc, mapped_targets_mc, training_samples_real
    )
    
    # Preprocess data
    (training_input_mc, testing_input_mc, 
     training_input_real, testing_input_real,
     training_target_mc, testing_target_mc) = preprocess_data( #, scaler
        training_input_mc, testing_input_mc, training_input_real, testing_input_real, training_target_mc, testing_target_mc
    )
    
    # Create DataLoaders
    data_loaders = create_dataloaders(
        training_input_mc, training_target_mc, testing_input_mc, testing_target_mc,
        training_input_real, testing_input_real, params['batchsize'], device
    )
    
    # Initialize models
    models = initialize_models(device)
    
    # Setup training
    optimizers, class_loss_fn, domain_loss_fn = setup_training(
        models, params, training_target_mc, device
    )    
    
    # Train the DANN model
    training_history = train_dann(
        params, data_loaders, models, optimizers, class_loss_fn, domain_loss_fn, device
    )
        
    # Set model save paths
    model_saving_paths = (
        'DANN_FeatureExtractor_SGD.pth',
        'DANN_ClassClassifier_SGD.pth',
        'DANN_DomainClassifier_SGD.pth'
    )
    
    # Save trained models
    save_models(models, model_saving_paths)

    # Convert the feature extractor to ONNX format
    feature_extractor, class_classifier, domain_classifier = models
    
    # # Export feature extractor
    # dummy_input_fe = torch.randn(1, 7).to(device)
    # onnx_program_fe = torch.onnx.export(feature_extractor, dummy_input_fe, dynamo=True)
    # onnx_program_fe.save("DANN_FeatureExtractor_SGD.onnx")
    # print("Feature Extractor converted to ONNX format successfully.")
    
    # # Export class classifier
    # dummy_input_cc = torch.randn(1, 128).to(device)
    # onnx_program_cc = torch.onnx.export(class_classifier, dummy_input_cc, dynamo=True)
    # onnx_program_cc.save("DANN_ClassClassifier_SGD.onnx")
    # print("Class Classifier converted to ONNX format successfully.")

    print("DANN training completed successfully.")

    
if __name__ == "__main__":
    main()