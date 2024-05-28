from model.model_class.environment import *

class GraphRegressionModel(object):
    """ Model Template for Classification """

    class GeneralModel():
        def __init__(self, configs):
            pass

    def __init__(self, configs, name="Model"):
        super().__init__()
        self.configs = configs
        self.name = self.configs.name 
        self.model = self.Model(self.configs) # create the model

        # operations
            # set seed - control randomness
        torch.manual_seed(self.configs.random_state) 

            # optimiser and criterion
        self.optimizer = AdamW(self.model.parameters(), lr=self.configs.lr)
        self.criterion = self.configs.loss
        self.validation_criterion = self.configs.validation_loss

        # automatically detect GPU device if avilable
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.configs.device = self.device
        
        # turn all the losses to GPU
        self.model.to(self.device)
        self.criterion.to(self.device)
        self.validation_criterion.to(self.device) # extra loss in case we wanted to measure it differently for early stop

    def __str__(self):
        return self.name
    
    def save(self, mark=''):
        """ Saving Mechanism - quite direct """
        mark = ' ' + mark if mark else mark
        torch.save(self.model.state_dict(), os.path.join(self.configs.rootpath, f'state/{self}{mark}.pt'))
    
    def load(self, mark=''):
        """ Loading Mechanism - quite direct """
        mark = ' ' + mark if mark else mark
        self.model.load_state_dict(torch.load(os.path.join(self.configs.rootpath, f'state/{self}{mark}.pt'), map_location=self.device))
    
    def fit(self, total_train_X_spatial, total_train_X_nonspatial, total_train_y, train_masks, total_val_X_spatial, total_val_X_nonspatial, total_val_y, val_masks, graph):
        
        # we didn't use DataSet and DataLoader, instead just a list to control replicability.
        # thus, need to deepcopy to prevent messing up the original data
        total_train_X_spatial, total_train_X_nonspatial, total_train_y = copy.deepcopy(total_train_X_spatial), copy.deepcopy(total_train_X_nonspatial), copy.deepcopy(total_train_y)

        self.model.train() # Train mode of model records gradients

        # sets scheduler and patience
        scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', factor=0.5, patience=self.configs.patience//2) if self.configs.scheduler else None
        patience = self.configs.patience

        # get a list of random seeds, using our original random seed
        np.random.seed(self.configs.random_state) 
        seeds = [np.random.randint(0, 1000000) for _ in range(self.configs.epochs)]
        
        GRAPH = torch.FloatTensor(graph).to(self.device)

        # min_loss = np.inf
        max_r2 = -np.inf
        best_epoch = 0
        for epoch in range(self.configs.epochs):


            if not patience: # end training when no more patience
                break

            epoch_loss = 0
            epoch_pred, epoch_true = [], []
            
            # shuffles all the data we will use in mini batch.
            # we choose to shuffle after every mini batch to provide randomness
            np.random.seed(seeds[epoch])
            np.random.shuffle(total_train_X_spatial)
            np.random.seed(seeds[epoch]) # reset seed so that they are shuffled in same order
            np.random.shuffle(total_train_y)
            np.random.seed(seeds[epoch])
            np.random.shuffle(total_train_X_nonspatial)

            n_batch = 0

            for mini_batch_number in tqdm(range(len(total_train_X_spatial))):

 
                # make this mini batch into a tensor, and move to GPU
                geospatial_X, non_geospatial_X, y, mask = torch.FloatTensor(total_train_X_spatial[mini_batch_number]).to(self.device), \
                            torch.FloatTensor(total_train_X_nonspatial[mini_batch_number]).to(self.device), \
                            torch.FloatTensor(total_train_y[mini_batch_number]).to(self.device), \
                            torch.BoolTensor(train_masks[mini_batch_number]).to(self.device)
                

                # edge case of last mini batch containing no data since we use //
                if len(geospatial_X) == 0:
                    break
                
                # zero the gradients
                self.optimizer.zero_grad()

                pred, true = self.model(geospatial_X[mask], non_geospatial_X[mask], GRAPH[mask][:, mask]), y[mask]

                # calculate loss
                loss = self.criterion(pred, true)

                # backpropagation
                loss.backward()
                if self.configs.grad_clip: # gradient clip
                    nn.utils.clip_grad_norm(self.model.parameters(), 2)
                self.optimizer.step()
                
                epoch_loss += loss.detach().cpu().numpy()
                epoch_pred += pred.detach().cpu().tolist()
                epoch_true += true.detach().cpu().tolist()

                n_batch += 1
            
            epoch_loss /= n_batch # calculate average loss

            # print epoch training results
                # get the predicted label

            epoch_r2 = r2_score(epoch_true, epoch_pred)
            epoch_mae = mean_absolute_error(epoch_true, epoch_pred)
            epoch_mse = mean_squared_error(epoch_true, epoch_pred)
            epoch_rmse = np.sqrt(epoch_mse)

            record = f''' Epoch {epoch+1} Train | Loss: {epoch_loss:>7.4f} | R2: {epoch_r2:>7.4f}| MSE: {epoch_mse:>7.4f} | RMSE: {epoch_rmse:>7.4f} | MAE: {epoch_mae:>7.4f} '''

            print(record)

            # Validation
                # get the validation results
            valid_loss, valid_r2 = self.eval(total_val_X_spatial, total_val_X_nonspatial, total_val_y, val_masks, graph, epoch)
                # previously set to eval, now change back to train
            self.model.train()

                # we decide to optimise early stop based on balanced accuracy
                # note that we can change this to other metrics using configs.optimised_metric to dom1 or dom2's balaccu
            if valid_r2 > max_r2:
                max_r2 = valid_r2
                best_epoch = epoch
                self.save()

            else:
                patience -= 1
            if scheduler:
                scheduler.step(valid_loss)

        return best_epoch

    def predict(self, future_X_spatial, future_X_geospatial, future_masks, graph):
        self.model.eval()

        pred_y = []

        GRAPH = torch.FloatTensor(graph).to(self.device)

        with torch.no_grad():

            for mini_batch_number in tqdm(range(len(future_X_spatial))):
 
                # make this mini batch into a tensor, and move to GPU
                geospatial_X, non_geospatial_X, mask = torch.FloatTensor(future_X_spatial[mini_batch_number]).to(self.device), \
                            torch.FloatTensor(future_X_geospatial[mini_batch_number]).to(self.device), \
                            torch.BoolTensor(future_masks[mini_batch_number]).to(self.device)

                if len(geospatial_X) == 0:
                    break

                pred = self.model(geospatial_X[mask], non_geospatial_X[mask], GRAPH[mask][:, mask])

                pred_y.extend(pred.detach().cpu().tolist())
    
        return pred_y
        

    def eval(self, total_val_X_spatial, total_val_X_nonspatial, total_val_y, val_masks, graph, epoch, evaluation_mode = False):
        
        # get the prediction
        pred_val_y = self.predict(total_val_X_spatial, total_val_X_nonspatial, val_masks, graph)

        val_y = []
        for i in range(len(total_val_y)):
            val_y.extend(total_val_y[i][val_masks[i].astype(bool)])

            # get the tensor version of the prediction and true logits
        pred_val_y_tensor = torch.FloatTensor(np.array(pred_val_y)).to(self.device)
        val_y_tensor = torch.FloatTensor(val_y).to(self.device)
        
        epoch_r2 = r2_score(val_y, pred_val_y)
        epoch_mae = mean_absolute_error(val_y, pred_val_y)
        epoch_mse = mean_squared_error(val_y, pred_val_y)
        epoch_rmse = np.sqrt(epoch_mse)

        epoch_loss = self.criterion(pred_val_y_tensor, val_y_tensor)

        record = f'''Epoch {epoch+1} Val | Loss: {epoch_loss:>7.4f} | R2: {epoch_r2:>7.4f}| MSE: {epoch_mse:>7.4f} | RMSE: {epoch_rmse:>7.4f} | MAE: {epoch_mae:>7.4f} '''

        print(record)

        if not evaluation_mode: # return the results if not in evaluation mode for early stop

            return epoch_loss, epoch_r2
            