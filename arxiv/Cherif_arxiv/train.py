
from pickletools import optimize
import torch 
from model import GCN, SAGE
from data import OGBNDataset
import torch.nn.functional as F


class Trainer_node:
    def __init__(self,model, data, train_idx, valid_idx,test_idx,evaluator,args,logger):
        self.model = model 
        self.data = data 
        self.train_idx = train_idx
        self.valid_idx = valid_idx
        self.test_idx = test_idx
    
        self.evaluator = evaluator
        self.args = args
        self.logger = logger
        #self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)


    def train(self,model,optimizer):
        model.train()

        optimizer.zero_grad()
        out = model(self.data.x, self.data.adj_t)[self.train_idx]
        loss = F.nll_loss(out, data.y.squeeze(1)[train_idx])
        # loss = torch.nn.CrossEntropyLoss()  
        # loss= loss(out, self.data.y.squeeze(1)[self.train_idx])
        loss.backward()
        optimizer.step()

        return loss.item()


    # @torch.no_grad()
    def test(self, model):
        model.eval()

        out = model(self.data.x, self.data.adj_t)
        y_pred = out.argmax(dim=-1, keepdim=True)

        # Train accuracy
        train_acc = self.evaluator.eval({'y_true': self.data.y[self.train_idx],'y_pred': y_pred[self.train_idx],})['acc']
        # Valid accuracy
        valid_acc = self.evaluator.eval({'y_true': self.data.y[self.valid_idx],'y_pred': y_pred[self.valid_idx],})['acc']

        # test_accuracy
        test_acc = self.evaluator.eval({'y_true': self.data.y[self.test_idx],'y_pred': y_pred[self.test_idx],})['acc']

        return train_acc, valid_acc, test_acc

    
    def main(self,optimizer):
       
        for run in range(self.args.runs):
            self.model.reset_parameters()
            
            for epoch in range(1, 1 + self.args.epochs):
                loss = self.train(self.model,optimizer)
                result = self.test(self.model)
                self.logger.add_result(run, result)

                if epoch % self.args.log_steps == 0:
                    train_acc, valid_acc, test_acc = result
                    print(f'Run: {run + 1:02d}, '
                        f'Epoch: {epoch:02d}, '
                        f'Loss: {loss:.4f}, '
                        f'Train: {100 * train_acc:.2f}%, '
                        f'Valid: {100 * valid_acc:.2f}% '
                        f'Test: {100 * test_acc:.2f}%')

            self.logger.print_statistics(run)
        self.logger.print_statistics()




if __name__ == "__main__":
    main_dataset = OGBNDataset()
    data = main_dataset.data
    
    args = main_dataset.args
    dataset = main_dataset.dataset
    device = main_dataset.device
    #model = GCN(data.num_features, args.hidden_channels,dataset.num_classes, args.num_layers, args.dropout).to(device)
    model = SAGE(data.num_features, args.hidden_channels,dataset.num_classes, args.num_layers, args.dropout).to(device)
    train_idx = main_dataset.train_idx
    valid_idx = main_dataset.valid_idx
    test_idx =main_dataset.test_idx
    evaluator = main_dataset.evaluator
    logger = main_dataset.logger
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    trainer = Trainer_node(model, data, train_idx, valid_idx,test_idx,evaluator,args,logger)
    trainer.main(optimizer)
    