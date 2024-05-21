from model.model_class.__template__ import *


class GNN(GraphRegressionModel):
    class Model(nn.Module):

        def __init__(self, CFG):
            super().__init__()
            self.CFG = CFG

            torch.manual_seed(self.CFG.random_state)

            if self.CFG.learnable_graph:
                self.adjacency_matrix = nn.Parameter(self.CFG.adjacency_matrix)

            # graph input layer
            
            # gcn/agcn
                
            # linear layer ()

        # TODO: mask out on template, and here.
        
        
        def forward(self, x_spatial, x_nonspatial, graph):

            mask_today_list = create_mask_today(x)

            y = torch.full([x.shape[0], 1], np.nan).to(self.CFG.device)

            y[mask_today_list[-1], :] = self.out(x_gcn)

            return y
        
    def __init__(self, CFG, name="GNN"):
        super().__init__(CFG, name=name)


class GCN_Cell(nn.Module):
    def __init__(self, CFG):
        pass
    def forward(self):
        pass


class A_GCN_Cell(nn.Module):
    def __init__(self, CFG):
        pass
    def forward(self):
        pass

class GCN_Layer(nn.Module):
    def __init__(self, CFG):
        pass
    def forward(self):
        pass

class GraphInputLayer(nn.Module):
    def __init__(self, CFG):
        pass
    def forward(self):
        pass