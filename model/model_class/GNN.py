from model.model_class.__template__ import *


class GNN(GraphRegressionModel):
    class Model(nn.Module):

        def __init__(self, CFG):
            super().__init__()
            self.CFG = CFG

            torch.manual_seed(self.CFG.random_state)

            # graph input layer
            self.input = GraphInputLayer(self.CFG)
            
            # gcn/agcn
            if self.attention_heads == 0:
                self.gcn = self.gcn = nn.Sequential(*[
                                            nn.Sequential(
                                                GCN(self.CFG), 
                                                nn.Dropout(),
                                                nn.ReLU()
                                            )
                                            for _ in range(self.CFG.n_layers)
                                        ])
            else:
                self.gcn = self.gcn = nn.Sequential(*[
                                            nn.Sequential(
                                                A_GCN(self.CFG), 
                                                nn.Dropout(),
                                                nn.ReLU()
                                            )
                                            for _ in range(self.CFG.n_layers)])
            
            
            self.linear = GraphOutputLayer(self.CFG)
        
        
        def forward(self, x_spatial, x_nonspatial, graph):

            x_spatial = self.input(x_spatial)

            for layer in self.gcn:
                x_spatial = layer(x_spatial, graph)

            x_combined = torch.cat((x_spatial, x_nonspatial), dim=-1) # TODO: make x_spatial down to 1 dim?
            out = self.linear(x_combined)

            return out
        
    def __init__(self, CFG, name="GNN"):
        super().__init__(CFG, name=name)


class GCN(nn.Module):
    def __init__(self, CFG):
        super().__init__()
        self.CFG = CFG
    
        torch.manual_seed(self.CFG.random_state)
        self.gcn = nn.Linear(self.CFG.hidden_dim, self.CFG.hidden_dim, bias = True)

    def forward(self, x, graph):
        return self.gcn(graph @ x)



class A_GCN(nn.Module): # TODO: double check with attention is all you need
    def __init__(self, CFG):
        super().__init__()
        self.CFG = CFG
    
        torch.manual_seed(self.CFG.random_state)

        self.dim_per_head = self.CFG.hidden_dim // self.CFG.attention_heads

        self.QLinear = nn.Linear(self.CFG.hidden_dim, self.CFG.hidden_dim, bias = False)
        self.KLinear = nn.Linear(self.CFG.hidden_dim, self.CFG.hidden_dim, bias = False) 
        self.VLinear = nn.Linear(self.CFG.hidden_dim, self.CFG.hidden_dim, bias = False) 
        self.OutLinear = nn.Linear(self.CFG.hidden_dim, self.CFG.hidden_dim, bias = False) 
        self.softmax = nn.Softmax(dim = -1)
        self.relu = nn.ReLU()

        self.forward1 = nn.Linear(self.CFG.hidden_dim, self.CFG.hidden_dim, bias = False) 
        self.forward2 = nn.Linear(self.CFG.hidden_dim, self.CFG.hidden_dim, bias = False) 

        self.layer_norm1 = nn.LayerNorm(self.CFG.hidden_dim)
        self.dropout = nn.Dropout(self.CFG.dropout)
        self.layer_norm2 = nn.LayerNorm(self.CFG.hidden_dim)

    def forward(self, x, graph):
        Q = self.dropout(self.QLinear(x))
        K = self.dropout(self.KLinear(x))
        V = self.dropout(self.VLinear(x))

        attention_out_list = []
        for j in range(self.CFG.attention_heads):

            Q_tmp = Q[:, j*self.dim_per_head:(j+1)*self.dim_per_head]
            K_tmp = K[:, j*self.dim_per_head:(j+1)*self.dim_per_head]
            V_tmp = V[:, j*self.dim_per_head:(j+1)*self.dim_per_head]

            attention_out_tmp = self.softmax((Q_tmp @ torch.transpose(K_tmp, 0, 1)) * graph / np.sqrt(self.dim_per_head)) @ V_tmp

            attention_out_list.append(attention_out_tmp)

        out = self.dropout(self.relu(torch.cat(attention_out_list, dim = 1)))

        attention_out = self.dropout(self.OutLinear(out))

        x_add_attention = x + attention_out

        x_add_attention_normalised = self.layer_norm1(x_add_attention)

        forward_output = self.forward2(self.dropout(self.relu(self.forward1(x_add_attention_normalised))))

        x_add_forward = forward_output + x_add_attention

        x_add_forward_normalised = self.layer_norm2(x_add_forward)

        return x_add_forward_normalised

class GraphInputLayer(nn.Module):
    def __init__(self, CFG):
        super().__init__()
        self.CFG = CFG

        torch.manual_seed(self.CFG.random_state)
        self.input = nn.Sequential(
            nn.Linear(self.CFG.spatial_input_dim, self.CFG.hidden_size),
            nn.ReLU(),
            nn.Dropout(self.CFG.dropout)
        )

    def forward(self, x):
        return self.input(x)
    
class GraphOutputLayer(nn.Module):
    def __init__(self, CFG):
        super().__init__()
        self.CFG = CFG

        torch.manual_seed(self.CFG.random_state)
        self.output = nn.Sequential(
            nn.Linear(self.CFG.hidden_size + self.CFG.nonspatial_input_dim, 1),
        )

    def forward(self, x):
        return self.output(x)